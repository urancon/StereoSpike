import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data.dataset import Dataset
import spikingjelly

from .utils import mvsecLoadRectificationMaps, mvsecRectifyEvents, mvsecCumulateSpikesIntoFrames, \
    mvsecSpikesAndDepth, mvsecToVideo
from network.metrics import lin_to_log_depths


SEQUENCES_FRAMES = {'indoor_flying': {'indoor_flying_1': (140, 400),  # (140, 1200)
                                      'indoor_flying_2': (160, 1580),
                                      'indoor_flying_3': (125, 1815),
                                      'indoor_flying_4': (90, 360)}
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

    def __init__(self, root: str, num_frames_per_depth_map=1, normalization=None, take_log=True, show_sequence=False):
        print("\n#####################################")
        print("# LOADING AND PREPROCESSING DATASET #")
        print("#####################################\n")

        self.root = root
        self.normalization = normalization
        self.num_frames_per_depth_map = num_frames_per_depth_map

        # load the data
        datafile = self.root + 'indoor_flying/indoor_flying1_data'
        data = h5py.File(datafile + '.hdf5', 'r')
        datafile_gt = self.root + 'indoor_flying/indoor_flying1_gt'
        data_gt = h5py.File(datafile_gt + '.hdf5', 'r')

        # get the ground-truth depth maps (i.e. our labels) and their timestamps
        Ldepths_rect = np.array(data_gt['davis']['left']['depth_image_rect'])  # RECTIFIED / LEFT
        Rdepths_rect = np.array(data_gt['davis']['right']['depth_image_rect'])  # RECTIFIED / RIGHT
        Ldepths_rect_ts = np.array(data_gt['davis']['left']['depth_image_rect_ts'])
        Rdepths_rect_ts = np.array(data_gt['davis']['right']['depth_image_rect_ts'])

        # convert linear (metric) to normalized log depths if required
        if take_log:
            Ldepths_rect = lin_to_log_depths(Ldepths_rect)
            Rdepths_rect = lin_to_log_depths(Rdepths_rect)

        # remove depth maps occurring during take-off and landing of the drone (bad data)
        start_idx, end_idx = SEQUENCES_FRAMES['indoor_flying']['indoor_flying_1']
        Ldepths_rect = Ldepths_rect[start_idx:end_idx, :, :]
        Rdepths_rect = Rdepths_rect[start_idx:end_idx, :, :]
        Ldepths_rect_ts = Ldepths_rect_ts[start_idx:end_idx]
        Rdepths_rect_ts = Rdepths_rect_ts[start_idx:end_idx]

        # replace nan values with 255.
        Ldepths_rect = np.nan_to_num(Ldepths_rect, nan=0)
        Rdepths_rect = np.nan_to_num(Rdepths_rect, nan=0)

        # get the events
        Levents = np.array(data['davis']['left']['events'])  # EVENTS: X Y TIME POLARITY
        Revents = np.array(data['davis']['right']['events'])  # EVENTS: X Y TIME POLARITY

        # remove events occurring during take-off and landing of the drone as well
        Levents = Levents[(Levents[:, 2] > Ldepths_rect_ts[0] - 0.05) & (Levents[:, 2] < Ldepths_rect_ts[-1])]
        Revents = Revents[(Revents[:, 2] > Ldepths_rect_ts[0] - 0.05) & (Revents[:, 2] < Ldepths_rect_ts[-1])]

        # rectify the spatial coordinates of spike events and get rid of events falling outside of the 346x260 fov
        Lx_path = self.root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_x_map.txt'
        Ly_path = self.root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_y_map.txt'
        Rx_path = self.root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_x_map.txt'
        Ry_path = self.root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_y_map.txt'
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

        if self.normalization is not None and self.normalization != 'frequency':
            raise NotImplementedError(self.normalization)
            #self.data_left = normalize_frame(xL, self.normalization)
            #self.data_right = normalize_frame(xR, self.normalization)
        else:
            self.data_left = xL
            self.data_right = xR

        self.labels = yL  # only use left depth map as ground-truth

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
        return [frame_left, frame_right], label

    def show(self):
        # TODO: les 2 premières frames de chaque chunk ont des pixels a 0.5, tandis que les suivantes en ont à 1 et à 2
        #  determiner pourquoi !! Indice: normalisation de Wei (ligne 100 de ce fichier).
        i = 0
        for chunk in self.data_left:
            i += 1
            j = 0
            for frame in chunk:
                j +=1
                f = frame[0]
                #plt.imshow(f)
                #plt.show()
                cv2.imshow("cumulated ON event frames", f)
                cv2.waitKey(int(1000 / (20 * self.num_frames_per_depth_map)))
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(MVSEC.get_wh())
    dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/', num_frames_per_depth_map=5, normalization='max')
    print(len(dataset))
    x, y = dataset[0]
    print(x[0].shape)
    print(y.shape)

