import h5py
import cv2
import numpy as np
import gc
from tqdm import tqdm

from torch.utils.data.dataset import Dataset

import spikingjelly
from spikingjelly.datasets.utils import EventsFramesDatasetBase, integrate_events_to_frames, normalize_frame

from mvsec_utils import mvsecLoadRectificationMaps, mvsecRectifyEvents, mvsecFloatToInt, mvsecCumulateSpikesIntoFrames


class MVSEC(EventsFramesDatasetBase):

    @staticmethod
    def get_wh():
        return 346, 260

    def __init__(self, root: str, use_frame=True, frames_num=10, split_by='number', normalization='max'):
        # TODO: write some docstring about data format (shape of tensors, meaning of dimensions, etc.)
        # TODO: remove unused arguments

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
        Levents = data['davis']['left']['events']  # EVENTS: X Y TIME POLARITY
        Levents = np.array(Levents)
        Revents = data['davis']['right']['events']  # EVENTS: X Y TIME POLARITY
        Revents = np.array(Revents)

        # rectify the spatial coordinates of spike events
        Lx_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_x_map.txt'
        Ly_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_y_map.txt'
        Rx_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_x_map.txt'
        Ry_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_y_map.txt'
        Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
        rect_Levents = np.array(mvsecRectifyEvents(Levents, Lx_map, Ly_map))
        rect_Revents = np.array(mvsecRectifyEvents(Revents, Rx_map, Ry_map))

        # convert the timestamps from float to integer (needed for compatibility with spikingjelly)
        rect_Levents = mvsecFloatToInt(rect_Levents)
        rect_Revents = mvsecFloatToInt(rect_Revents)

        # save the data to Address Event Representation (AER) over the whole sequence
        #frames_num = len(Ldepths_rect_ts)
        """
        self.data = {'left':
                         {'t': rect_Levents[:, 2], 'x': rect_Levents[:, 0], 'y': rect_Levents[:, 1],
                          'p': rect_Levents[:, 3]},
                     'right':
                         {'t': rect_Revents[:, 2], 'x': rect_Revents[:, 0], 'y': rect_Revents[:, 1],
                          'p': rect_Revents[:, 3]},
                     }
        self.labels = {'left':
                           {'t': Ldepths_rect_ts, 'maps': Ldepths_rect},
                       'right':
                           {'t': Rdepths_rect_ts, 'maps': Rdepths_rect}
                       }
        """

        xL, yL = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts, num_frames_per_depth_map=2)
        xR, yR = mvsecCumulateSpikesIntoFrames(rect_Revents, Rdepths_rect, Rdepths_rect_ts, num_frames_per_depth_map=2)

        # close hf5py file as it is no longer useful
        data.close()

        del Levents
        del Revents
        del rect_Levents
        del rect_Revents

        gc.collect()

        self.data = np.stack((xL, xR))  # too memory consuming for num_frames_per_depth_map > 5 !!!! # data[0] contains 'left' data, data[1] contains 'right' data
        self.labels = np.stack((yL, yR))  # shape [2 (left/right), # of frames, H, W]

    def __len__(self):
        """
        :return: the number of frames into which we split the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, index):
        frame = self.data[:, index] # TODO: currently not grabbing the right thing
        label = self.labels[:, index]

        # TODO: normalize grabbed frame (use spikingjelly's code)
        #if self.normalization is not None and self.normalization != 'frequency':
        #    frame = normalize_frame(frame, self.normalization)

        return frame, label


if __name__ == "__main__":
    print(MVSEC.get_wh())
    dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/')
    print(len(dataset))
    x, y = dataset[0]
    print(x.shape)
    print(y.shape)
