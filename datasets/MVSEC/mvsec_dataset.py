import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data.dataset import Dataset
import spikingjelly

from .utils import mvsecLoadRectificationMaps, mvsecRectifyEvents, mvsecCumulateSpikesIntoFrames, \
    mvsecSpikesAndDepth
from network.metrics import lin_to_log_depths


SEQUENCES_FRAMES = {'indoor_flying': {'indoor_flying1': (140, 400), # first: (140, 400)  # other papers: (140, 1200)
                                      'indoor_flying2': (160, 1580),
                                      'indoor_flying3': (125, 1815),
                                      'indoor_flying4': (90, 360)}
                    }


class MVSEC(Dataset):
    """
    Neuromorphic dataset class to hold MVSEC data.

    We use Adress Event Representation (AER) format to represent events.
    An MVSEC sequence (e.g. 'indoor_flying_4') is cut into frames on which we accumulate spikes occurring during a
    certain time interval dt.

    Essentially, the sequence is translated in 2 input tensors (left/right) of shape [# of frames, 2 (ON/OFF), W, H]
    Corresponding round-truth depth maps are contained in a tensor of shape [# of frames, 2 (ON/OFF), W, H]
    """

    @staticmethod
    def get_wh():
        return 346, 260

    def __init__(self, root: str, scenario: str, case: str, start_end=(0, -1), num_frames_per_depth_map=1, mirror_time=False,
                 take_log=True, show_sequence=False):
        print("\n#####################################")
        print("# LOADING AND PREPROCESSING DATASET #")
        print("#####################################\n")

        self.root = root
        self.num_frames_per_depth_map = num_frames_per_depth_map

        # load the data
        datafile = self.root + '{}/{}{}_data.hdf5'.format(scenario, scenario, case)
        data = h5py.File(datafile, 'r')
        datafile_gt = self.root + '{}/{}{}_gt.hdf5'.format(scenario, scenario, case)
        data_gt = h5py.File(datafile_gt, 'r')

        # get the ground-truth depth maps (i.e. our labels) and their timestamps
        Ldepths_rect = np.array(data_gt['davis']['left']['depth_image_rect'])  # RECTIFIED / LEFT
        Rdepths_rect = np.array(data_gt['davis']['right']['depth_image_rect'])  # RECTIFIED / RIGHT
        Ldepths_rect_ts = np.array(data_gt['davis']['left']['depth_image_rect_ts'])
        Rdepths_rect_ts = np.array(data_gt['davis']['right']['depth_image_rect_ts'])

        # remove depth maps occurring during take-off and landing of the drone (bad data)
        start_idx, end_idx = SEQUENCES_FRAMES[scenario][scenario+case]
        Ldepths_rect = Ldepths_rect[start_idx:end_idx, :, :]
        Rdepths_rect = Rdepths_rect[start_idx:end_idx, :, :]
        Ldepths_rect_ts = Ldepths_rect_ts[start_idx:end_idx]
        Rdepths_rect_ts = Rdepths_rect_ts[start_idx:end_idx]

        # replace nan values with 255.
        Ldepths_rect = np.nan_to_num(Ldepths_rect, nan=0)
        Rdepths_rect = np.nan_to_num(Rdepths_rect, nan=0)

        # shape of each depth map: (H, W) --> (1, H, W)
        Ldepths_rect = np.expand_dims(Ldepths_rect, axis=1)
        Rdepths_rect = np.expand_dims(Rdepths_rect, axis=1)

        # convert linear (metric) to normalized log depths if required
        if take_log:
            Ldepths_rect = lin_to_log_depths(Ldepths_rect)
            Rdepths_rect = lin_to_log_depths(Rdepths_rect)

        # get the events
        Levents = np.array(data['davis']['left']['events'])  # EVENTS: X Y TIME POLARITY
        Revents = np.array(data['davis']['right']['events'])  # EVENTS: X Y TIME POLARITY

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

        # show the sequence
        if show_sequence:
            mvsecSpikesAndDepth(Ldepths_rect, rect_Levents)

        # convert data to a sequence of frames
        xL, yL = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts,
                                               num_frames_per_depth_map=num_frames_per_depth_map)
        xR, _ = mvsecCumulateSpikesIntoFrames(rect_Revents, Rdepths_rect, Rdepths_rect_ts,
                                               num_frames_per_depth_map=num_frames_per_depth_map)

        assert xL.shape == xR.shape

        # only keep a subset of the sequence
        start_chunk_idx = start_end[0]
        end_chunk_idx = start_end[1]
        xL = xL[start_chunk_idx:end_chunk_idx]
        yL = yL[start_chunk_idx:end_chunk_idx]
        xR = xR[start_chunk_idx:end_chunk_idx]

        # use temporal mirroring for data augmentation and smooth training
        if mirror_time:
            xL_mirr = np.flip(xL, axis=0)  # reverse the order of spike frames
            xL_mirr = np.flip(xL_mirr, axis=2)  # reverse spike polarities because we are going backward in time
            xL_final = np.concatenate((xL, xL_mirr), axis=0)  # concatenate original and mirrored frame sequences

            yL_mirr = np.flip(yL, axis=0)  # reverse the order of labels
            yL_final = np.concatenate((yL, yL_mirr), axis=0)  # concatenate original and mirrored label sequences

            xR_mirr = np.flip(xR, axis=0)
            xR_mirr = np.flip(xR_mirr, axis=2)
            xR_final = np.concatenate((xR, xR_mirr), axis=0)
        else:
            xL_final = xL
            xR_final = xR
            yL_final = yL

        self.data_left = xL_final  # currently (n_chunks, nfpdm, 2, 260, 346), should have shape (N, num_chunks, N*nfpdm, 2, 260, 346)
        self.data_right = xR_final

        self.labels = yL_final  # only use left depth map as ground-truth

        # close hf5py file properly
        data.close()

    def __len__(self):
        """
        :return: the number of frames into which we split the dataset
        """
        return self.data_left.shape[0]

    def __getitem__(self, index):
        """
        :param index:
        :return: the left and right spike frames and their associated ground-truth for the input index. Consequently,
                 frame_left and frame_right have shape [2 (polarity), W, H] and label has shape [H, W]
        """
        frame_left = self.data_left[index]
        frame_right = self.data_right[index]
        label = self.labels[index]
        return frame_left, frame_right, label

    def show(self):
        # TODO: les 2 premières frames de chaque chunk ont des pixels a 0.5, tandis que les suivantes en ont à 1 et à 2
        #  determiner pourquoi !! Indice: normalisation de Wei (ligne 100 de ce fichier).
        i = 0
        for chunk in self.data_left:
            i += 1
            j = 0
            for frame in chunk:
                j += 1
                f = frame[0]
                #plt.imshow(f)
                #plt.show()
                cv2.imshow("cumulated ON event frames", f)
                cv2.waitKey(int(1000 / (20 * self.num_frames_per_depth_map)))
        cv2.destroyAllWindows()


class shuffled_MVSEC(Dataset):
    """
    Designed for a training experiment where the training data is not fed sequentially anymore.

    ONLY MONOCULAR AT THE MOMENT !!!

    Motivations (discord message from Ulysse):
    --------------------------------------------
    Hello, I've got yet another idea to help the network generalize. It is certainly far less elegant than what we've
    done sor far, but I think we have to give it a try. I got this idea thinking about all other papers we have seen
    so far, including the latest paper I told you about. The basic idea is fairly simple: shuffle the dataset. I know
    it's counterintuitive but let me explain the reasons, and how I propose to do it.

    Lately, we've seen that our model is able to learn and render the scene very well while training. But once in
    eval mode, we can notice that it's just learning by heart and tuning its weights to the last input data and
    labels it has seen. In my opinion, this is mainly caused by our training on the dataset sequentially. Let's
    consider a random input data and label in the sequential dataset. In a surrounding of say, +- 5 chunks of 50ms
    each, input data and labels are all very similar to each other. If our model has a short-term memory of around
    500ms, it's learning this subsequence very well, but it and only it ! And because we feed in the training data
    sequentially, the model learns a sort of sliding archetypal sequence of 500ms.

    So to overcome this, I propose to train on the shuffled dataset in a "smart" way. Training on a large amount of
    shuffled data just like an image classification or most other deep learning models might be the key to
    generalization. I don't like it for sequential datasets like MVSEC, but it's what I believe.

    What I propose to do is:
        1) initialize the output potentials with the groundtruth preceding the warmup data
        2) forward pass on warmup data without building any computational graph. The aim here is to let intermediate
        neurons reach a good steady state before doing the "real" inference. Warmup data could be constituted of
        e.g., 10 chunks of 2 frames for a total of 5*50=250ms
        3) forward pass on a long sample of the dataset while building a graph. The sample should be as long as possible
         to capture long-term dependencies. It could be constituted of 10 chunks of 2 frames, so for a total length of
         500ms
        4) calculate the loss and update weights
        5) go to 1) and start again with new data from the shuffled dataset

    It is frustrating to think that we could bypass the main difficulty we've had so far (i.e., generalization) by just
    not caring anymore about the temporal structure of the dataset. We've been trying to exploit it as much as possible
    until now, but I think we have to take a step back, and open-mindedly try conventional, yet less elegant,
    approaches. Of course, I won't give up on our own way of feeding the training data. But I'm very very excited to
    see what this does !
    """

    @staticmethod
    def get_wh():
        return 346, 260

    def __init__(self, root: str, scenario: str, case: str, start_end=(0, -1),
                 num_frames_per_depth_map=1, warmup_chunks=5, train_chunks=5,
                 mirror_time=False, take_log=True, show_sequence=False):
        print("\n#####################################")
        print("# LOADING AND PREPROCESSING DATASET #")
        print("#####################################\n")

        self.root = root
        self.num_frames_per_depth_map = num_frames_per_depth_map

        self.N_warmup = warmup_chunks
        self.N_train = train_chunks

        # load the data
        datafile = self.root + '{}/{}{}_data.hdf5'.format(scenario, scenario, case)
        data = h5py.File(datafile, 'r')
        datafile_gt = self.root + '{}/{}{}_gt.hdf5'.format(scenario, scenario, case)
        data_gt = h5py.File(datafile_gt, 'r')

        # get the ground-truth depth maps (i.e. our labels) and their timestamps
        Ldepths_rect = np.array(data_gt['davis']['left']['depth_image_rect'])  # RECTIFIED / LEFT
        Ldepths_rect_ts = np.array(data_gt['davis']['left']['depth_image_rect_ts'])

        # remove depth maps occurring during take-off and landing of the drone (bad data)
        start_idx, end_idx = SEQUENCES_FRAMES[scenario][scenario+case]
        Ldepths_rect = Ldepths_rect[start_idx:end_idx, :, :]
        Ldepths_rect_ts = Ldepths_rect_ts[start_idx:end_idx]

        # replace nan values with 255.
        Ldepths_rect = np.nan_to_num(Ldepths_rect, nan=0)

        # shape of each depth map: (H, W) --> (1, H, W)
        Ldepths_rect = np.expand_dims(Ldepths_rect, axis=1)

        # convert linear (metric) to normalized log depths if required
        if take_log:
            Ldepths_rect = lin_to_log_depths(Ldepths_rect)

        # get the events
        Levents = np.array(data['davis']['left']['events'])  # EVENTS: X Y TIME POLARITY

        # remove events occurring during take-off and landing of the drone as well
        Levents = Levents[(Levents[:, 2] > Ldepths_rect_ts[0] - 0.05) & (Levents[:, 2] < Ldepths_rect_ts[-1])]

        # rectify the spatial coordinates of spike events and get rid of events falling outside of the 346x260 fov
        Lx_path = self.root + '{}/{}_calib/{}_left_x_map.txt'.format(scenario, scenario, scenario)
        Ly_path = self.root + '{}/{}_calib/{}_left_y_map.txt'.format(scenario, scenario, scenario)
        Rx_path = self.root + '{}/{}_calib/{}_right_x_map.txt'.format(scenario, scenario, scenario)
        Ry_path = self.root + '{}/{}_calib/{}_right_y_map.txt'.format(scenario, scenario, scenario)
        Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
        rect_Levents = np.array(mvsecRectifyEvents(Levents, Lx_map, Ly_map))

        # show the sequence
        if show_sequence:
            mvsecSpikesAndDepth(Ldepths_rect, rect_Levents)

        # convert data to a sequence of frames
        xL, yL = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts,
                                               num_frames_per_depth_map=num_frames_per_depth_map)

        # only keep a subset of the sequence
        start_chunk_idx = start_end[0]
        end_chunk_idx = start_end[1]
        xL = xL[start_chunk_idx:end_chunk_idx]
        yL = yL[start_chunk_idx:end_chunk_idx]

        # use temporal mirroring for data augmentation and smooth training
        if mirror_time:
            xL_mirr = np.flip(xL, axis=0)  # reverse the order of spike frames
            xL_mirr = np.flip(xL_mirr, axis=2)  # reverse spike polarities because we are going backward in time
            xL_final = np.concatenate((xL, xL_mirr), axis=0)  # concatenate original and mirrored frame sequences

            yL_mirr = np.flip(yL, axis=0)  # reverse the order of labels
            yL_final = np.concatenate((yL, yL_mirr), axis=0)  # concatenate original and mirrored label sequences
        else:
            xL_final = xL
            yL_final = yL

        # store the (N_warmup + N_train) first chunks and labels for warmup and initialization
        self.first_data = xL_final[: 1 + 2*(self.N_warmup + self.N_train)]  # shape: (1+(2*N_warmup+N_train), nfpdm, 2, 260, 346)
        self.first_labels = yL_final[: 1 + 2*(self.N_warmup + self.N_train)]  # shape: (1+(2*N_warmup+N_train), 1, 260, 346)

        self.data = xL_final[self.N_warmup + self.N_train:]  # shape: (n_chunks - N_warmup, nfpdm, 2, 260, 346)
        self.labels = yL_final[self.N_warmup + self.N_train:]  # shape: (n_chunks - N_warmup, 1, 260, 346)

        # close hf5py file properly
        data.close()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        if index - self.N_train - self.N_warmup - 1 >= 0:  # index = 13
            init_pots = self.labels[index - self.N_train - self.N_warmup]  # 3
            warmup_chunks = self.data[index - self.N_train - self.N_warmup + 1: index - self.N_train + 1]  # 4 5 6 7 8
            train_chunks = self.data[index - self.N_train + 1: index + 1]  # 9 10 11 12 13
            groundtruth = self.labels[index]  # 13
            return init_pots, warmup_chunks, train_chunks, groundtruth

        elif index - self.N_train - self.N_warmup - 1 < 0:  # e.g. 2 - 5 - 5 = -8
            init_pots = self.first_labels[index]  # -8 (2)
            warmup_chunks = self.first_data[index + 1: index + 1 + self.N_warmup]  # -7 -6 -5 -4 -3 (3 4 5 6 7)
            train_chunks = self.first_data[index + 1 + self.N_warmup: index + 1 + self.N_warmup + self.N_train]  # -2 -1 0 1 2 (8 9 10 11 12)
            groundtruth = self.first_labels[index + self.N_warmup + self.N_train]  # 2 (12)
            return init_pots, warmup_chunks, train_chunks, groundtruth

    def show(self):
        i = 0
        for chunk in self.data_left:
            i += 1
            j = 0
            for frame in chunk:
                j += 1
                f = frame[0]
                cv2.imshow("cumulated ON event frames", f)
                cv2.waitKey(int(1000 / (20 * self.num_frames_per_depth_map)))
        cv2.destroyAllWindows()

