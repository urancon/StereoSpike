import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import spikingjelly
from spikingjelly.datasets.utils import EventsFramesDatasetBase, integrate_events_to_frames, normalize_frame

from .utils import mvsecLoadRectificationMaps, mvsecRectifyEvents, mvsecCumulateSpikesIntoFrames


class MVSEC(EventsFramesDatasetBase):
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

    def __init__(self, root: str, num_frames_per_depth_map=1, normalization='max'):
        self.normalization = normalization

        # load the data
        datafile = root + 'indoor_flying/indoor_flying4_data'
        data = h5py.File(datafile + '.hdf5', 'r')
        datafile_gt = root + 'indoor_flying/indoor_flying4_gt'
        data_gt = h5py.File(datafile_gt + '.hdf5', 'r')

        # get the ground-truth depth maps (i.e. our labels) and their timestamps
        Ldepths_rect = np.array(data_gt['davis']['left']['depth_image_rect'])  # RECTIFIED / LEFT
        Rdepths_rect = np.array(data_gt['davis']['right']['depth_image_rect'])  # RECTIFIED / RIGHT
        Ldepths_rect_ts = np.array(data_gt['davis']['left']['depth_image_rect_ts'])
        Rdepths_rect_ts = np.array(data_gt['davis']['right']['depth_image_rect_ts'])

        # get the events
        Levents = np.array(data['davis']['left']['events'])  # EVENTS: X Y TIME POLARITY
        Revents = np.array(data['davis']['right']['events'])  # EVENTS: X Y TIME POLARITY

        # rectify the spatial coordinates of spike events
        Lx_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_x_map.txt'
        Ly_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_y_map.txt'
        Rx_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_x_map.txt'
        Ry_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_y_map.txt'
        Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
        rect_Levents = np.array(mvsecRectifyEvents(Levents, Lx_map, Ly_map))
        rect_Revents = np.array(mvsecRectifyEvents(Revents, Rx_map, Ry_map))

        # convert data to a sequence of frames
        xL, yL = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts,
                                               num_frames_per_depth_map=num_frames_per_depth_map)
        xR, _ = mvsecCumulateSpikesIntoFrames(rect_Revents, Rdepths_rect, Rdepths_rect_ts,
                                               num_frames_per_depth_map=num_frames_per_depth_map)

        assert xL.shape == xR.shape

        if self.normalization is not None and self.normalization != 'frequency':
            self.data_left = normalize_frame(xL, self.normalization)
            self.data_right = normalize_frame(xR, self.normalization)
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


if __name__ == "__main__":
    print(MVSEC.get_wh())
    dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/', num_frames_per_depth_map=5, normalization='max')
    print(len(dataset))
    x, y = dataset[0]
    print(x[0].shape)
    print(y.shape)

