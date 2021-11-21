import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
import skimage.morphology as morpho

from .utils import mvsecLoadRectificationMaps, mvsecRectifyEvents, mvsecCumulateSpikesIntoFrames
from .indices import *
from network.metrics import lin_to_log_depths, depth_to_disparity


def load_MVSEC(root: str, scenario:str, split:str, num_frames_per_depth_map, warmup_chunks, train_chunks,
               transform=None, normalize=False, learn_on='LIN', load_test_only=False):
    """
        Load a split of MVSEC in only one function.

        Sequences and indices follow those presented in Tulyakov et al. (ICCV 2019),
            "Learning an Event Sequence Embedding for Dense Event-Based Deep Stereo"
    """

    # Indices for validation and test sets vary depending on the split
    # Training and validation/test sequences also vary with the split
    if split == '1':
        training_sequences = ['2', '3']
        valtest_sequence = '1'
        valid_indices = SPLIT1_VALID_INDICES
        test_indices = SPLIT1_TEST_INDICES
    elif split == '2':
        training_sequences = ['1', '3']
        valtest_sequence = '2'
        valid_indices = SPLIT2_VALID_INDICES
        test_indices = SPLIT2_TEST_INDICES
    elif split == '3':
        training_sequences = ['1', '2']
        valtest_sequence = '3'
        valid_indices = SPLIT3_VALID_INDICES
        test_indices = SPLIT3_TEST_INDICES

    # load all indoor_flying sequences and create datasets
    # return all three train, valid, and test sets for training
    if not load_test_only:
        dataset1 = MVSEC_sequence(root=root,
                                  scenario=scenario, split=split, sequence=training_sequences[0],
                                  num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks, train_chunks=train_chunks,
                                  normalize=normalize, transform=transform, learn_on=learn_on)
        dataset2 = MVSEC_sequence(root=root,
                                  scenario=scenario, split=split, sequence=training_sequences[1],
                                  num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks, train_chunks=train_chunks,
                                  normalize=normalize, transform=transform, learn_on=learn_on)
        dataset3 = MVSEC_sequence(root=root,
                                  scenario=scenario, split=split, sequence=valtest_sequence,
                                  num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks, train_chunks=train_chunks,
                                  normalize=normalize, transform=transform, learn_on=learn_on)

        train_set = torch.utils.data.ConcatDataset(datasets=[dataset1, dataset2])
        valid_set = torch.utils.data.Subset(dataset3, valid_indices)
        test_set = torch.utils.data.Subset(dataset3, test_indices)

        return train_set, valid_set, test_set

    # only return test set for evaluation
    else:
        dataset3 = MVSEC_sequence(root=root,
                                  scenario=scenario, split=split, sequence=valtest_sequence,
                                  num_frames_per_depth_map=num_frames_per_depth_map, warmup_chunks=warmup_chunks,
                                  train_chunks=train_chunks,
                                  normalize=normalize, transform=transform, learn_on=learn_on)

        test_set = torch.utils.data.Subset(dataset3, test_indices)

        return test_set


class MVSEC_sequence(Dataset):
    """
    Neuromorphic dataset class to hold MVSEC data.

    Raw events are initially represented in Adress Event Representation (AER) format, that is, as a list of tuples
    (X, Y, T, P).
    An MVSEC sequence (e.g. 'indoor_flying3') is cut into frames on which we accumulate spikes occurring during a
    certain time interval dt to constitute a spike frame.
    In our paper, dt = 50 ms, which happens to correspond to the frequency of ground truth depth maps provided by the
    LIDAR.

    One chunk corresponds to a duration of 50 ms, containing 'num_frames_per_depth_map' frames for one single label.
    Essentially, the sequence is translated in 2 input tensors (left/right) of shape [# chunks, # of frames, 2 (ON/OFF), W, H].
    Corresponding ground-truth depth maps are contained in a tensor of shape [# of chunks, W, H].
    Dataloaders finally add the batch dimension.

    Warmup chunks are chronologically followed by train chunks. Warmup chunks can be used for training recurrent models;
    the idea is to deactivate automatic differentiation and perform inference on warmup chunks before train chunks, so
    that hidden states within the model reach a steady state. Then activate autodiff back before forward passing train
    chunks.

    Therefore, in our paper, we used 1 train chunk of 1 frame (of 50 ms) per depth ground truth.

    'transform' can be used for data augmentation techniques, whose methods we provide in this data_augmentation.py file
    """

    @staticmethod
    def get_wh():
        return 346, 260

    def __init__(self, root: str, scenario: str, split: str, sequence: str,
                 num_frames_per_depth_map=1, warmup_chunks=5, train_chunks=5,
                 transform=None, normalize=False, learn_on='LIN'):

        print("\n#####################################")
        print("# LOADING AND PREPROCESSING DATASET #")
        print("#####################################\n")

        self.root = root
        self.num_frames_per_depth_map = num_frames_per_depth_map

        self.N_warmup = warmup_chunks
        self.N_train = train_chunks

        self.transform = transform

        # load the data
        datafile = self.root + '{}/{}{}_data.hdf5'.format(scenario, scenario, sequence)
        data = h5py.File(datafile, 'r')
        datafile_gt = self.root + '{}/{}{}_gt.hdf5'.format(scenario, scenario, sequence)
        data_gt = h5py.File(datafile_gt, 'r')

        # get the ground-truth depth maps (i.e. our labels) and their timestamps
        Ldepths_rect = np.array(data_gt['davis']['left']['depth_image_rect'])  # RECTIFIED / LEFT
        Ldepths_rect_ts = np.array(data_gt['davis']['left']['depth_image_rect_ts'])

        # remove depth maps occurring during take-off and landing of the drone (bad data)
        start_idx, end_idx = SEQUENCES_FRAMES[scenario]['split' + split][scenario + sequence]  # e.g., 'indoor_flying'/'split1'/'indoor_flying1'
        Ldepths_rect = Ldepths_rect[start_idx:end_idx, :, :]
        Ldepths_rect_ts = Ldepths_rect_ts[start_idx:end_idx]

        # fill holes (i.e., dead pixels) in the groundtruth with mathematical morphology's closing operation
        # yL has shape (num_labels, 1, 260, 346)
        for i in range(len(Ldepths_rect)):
            filled = morpho.area_closing(Ldepths_rect[i], area_threshold=24)
            Ldepths_rect[i] = filled

        # pixels with zero value get a NaN value because they are invalid
        Ldepths_rect[Ldepths_rect == 0] = np.nan

        # convert linear (metric) depth to log depth or disparity if required
        if learn_on == 'LOG':
            Ldepths_rect = lin_to_log_depths(Ldepths_rect)
        elif learn_on == 'DISP':
            Ldepths_rect = depth_to_disparity(Ldepths_rect)
        elif learn_on == 'LIN':
            pass
        else:
            raise ValueError("'learn_on' argument should either be 'LIN' for metric depth, "
                             "'LOG' for log depth, "
                             "or 'DISP' for disparity.")

        # shape of each depth map: (H, W) --> (1, H, W)
        Ldepths_rect = np.expand_dims(Ldepths_rect, axis=1)

        # get the events
        Levents = np.array(data['davis']['left']['events'])  # EVENTS: X Y TIME POLARITY
        Revents = np.array(data['davis']['right']['events'])

        # remove events occurring during take-off and landing of the drone as well
        Levents = Levents[(Levents[:, 2] > Ldepths_rect_ts[0] - 0.05) & (Levents[:, 2] < Ldepths_rect_ts[-1])]
        Revents = Revents[(Revents[:, 2] > Ldepths_rect_ts[0] - 0.05) & (Revents[:, 2] < Ldepths_rect_ts[-1])]

        # rectify the spatial coordinates of spike events and get rid of events falling outside of the 346x260 fov
        Lx_path = self.root + '{}/{}_calib/{}_left_x_map.txt'.format(scenario, scenario, scenario)
        Ly_path = self.root + '{}/{}_calib/{}_left_y_map.txt'.format(scenario, scenario, scenario)
        Rx_path = self.root + '{}/{}_calib/{}_right_x_map.txt'.format(scenario, scenario, scenario)
        Ry_path = self.root + '{}/{}_calib/{}_right_y_map.txt'.format(scenario, scenario, scenario)
        Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
        rect_Levents = np.array(mvsecRectifyEvents(Levents, Lx_map, Ly_map))
        rect_Revents = np.array(mvsecRectifyEvents(Revents, Rx_map, Ry_map))

        # convert data to a sequence of frames
        xL, yL = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts, num_frames_per_depth_map=num_frames_per_depth_map)
        xR, _ = mvsecCumulateSpikesIntoFrames(rect_Revents, Ldepths_rect, Ldepths_rect_ts, num_frames_per_depth_map=num_frames_per_depth_map)

        # normalize nonzero values in the input data to have zero mean and unit variance
        if normalize:
            nonzero_mask_L = xL > 0  # LEFT
            mL = xL[nonzero_mask_L].mean()
            sL = xL[nonzero_mask_L].std()
            xL[nonzero_mask_L] = (xL[nonzero_mask_L] - mL) / sL

            nonzero_mask_R = xR > 0  # RIGHT
            mR = xR[nonzero_mask_R].mean()
            sR = xR[nonzero_mask_R].std()
            xR[nonzero_mask_R] = (xR[nonzero_mask_R] - mR) / sR

        assert xL.shape == xR.shape

        # store the (N_warmup + N_train) first chunks and labels for warmup and initialization
        self.first_data_left = xL[: 1 + 2 * (
                    self.N_warmup + self.N_train)]  # shape: (1+(2*N_warmup+N_train), nfpdm, 2, 260, 346)
        self.first_data_right = xR[: 1 + 2 * (self.N_warmup + self.N_train)]
        self.first_labels = yL[: 1 + 2 * (self.N_warmup + self.N_train)]  # shape: (1+(2*N_warmup+N_train), 1, 260, 346)

        self.data_left = xL[self.N_warmup + self.N_train:]  # shape: (n_chunks - N_warmup, nfpdm, 2, 260, 346)
        self.data_right = xR[self.N_warmup + self.N_train:]
        self.labels = yL[self.N_warmup + self.N_train:]  # shape: (n_chunks - N_warmup, 1, 260, 346)

        # close hf5py file properly
        data.close()

    def __len__(self):
        return self.data_left.shape[0]

    def __getitem__(self, index):

        if index - self.N_train - self.N_warmup - 1 >= 0:  # index = 13
            init_pots = self.labels[index - self.N_train - self.N_warmup]  # 3
            warmup_chunks_left = self.data_left[
                                 index - self.N_train - self.N_warmup + 1: index - self.N_train + 1]  # 4 5 6 7 8
            warmup_chunks_right = self.data_right[index - self.N_train - self.N_warmup + 1: index - self.N_train + 1]
            train_chunks_left = self.data_left[index - self.N_train + 1: index + 1]  # 9 10 11 12 13
            train_chunks_right = self.data_right[index - self.N_train + 1: index + 1]
            groundtruth = self.labels[index]  # 13

        elif index - self.N_train - self.N_warmup - 1 < 0:  # e.g. 2 - 5 - 5 = -8
            init_pots = self.first_labels[index]  # -8 (2)
            warmup_chunks_left = self.first_data_left[
                                 index + 1: index + 1 + self.N_warmup]  # -7 -6 -5 -4 -3 (3 4 5 6 7)
            warmup_chunks_right = self.first_data_right[index + 1: index + 1 + self.N_warmup]
            train_chunks_left = self.first_data_left[
                                index + 1 + self.N_warmup: index + 1 + self.N_warmup + self.N_train]  # -2 -1 0 1 2 (8 9 10 11 12)
            train_chunks_right = self.first_data_right[
                                 index + 1 + self.N_warmup: index + 1 + self.N_warmup + self.N_train]
            groundtruth = self.first_labels[index + self.N_warmup + self.N_train]  # 2 (12)

        data = init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth
        # init_pots, label: (1, H, W)
        # warmup_chunks: (N_warmup, nfpdm, 2, H, W)
        # train_chunks: (N_train, nfpdm, 2, H, W)

        if self.transform:
            data = self.transform(data)

        return data

    def show(self):
        # TODO: implement show method
        pass