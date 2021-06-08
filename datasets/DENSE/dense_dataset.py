import os
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset


def splitAERinto(AER_data, num_frames: int):
    """
    Split an AER data list into num_frames AER lists by time binning of events.

    :param AER_data: A list or array of (t, x, y, p) quadruplets
    :param num_frames: The number of temporal bins to split the data into
    :return: A list of lists or array of (t, x, y, p) quadruplets. I.e., a list of AER_data like objects
    """

    first_ts = AER_data[0][0]
    last_ts = AER_data[-1][0]
    window_length = (last_ts - first_ts) / num_frames

    split_list = []

    start_ts = first_ts
    end_ts = first_ts + window_length

    for f in range(num_frames):
        filt_AER = AER_data[(AER_data[:, 0] > start_ts) & (AER_data[:, 0] < end_ts)]
        start_ts = end_ts
        end_ts = end_ts + window_length
        split_list.append(filt_AER)

    return split_list


def AERtoEVFrame(AER_data, AER_format='TXYP'):
    """
    Converts data in Address Event Representation (AER) to a 2-channel event frame or spike histogram.

    :param AER_data: A list or array of (t, x, y, p) quadruplets
    :return: an event frame, whose first channel contains cumulated ON spikes and second channels contains cumulated
     OFF spikes
    """

    # create new 2 channel frame to accumulate spikes on
    frame = np.zeros((2, 260, 346), dtype='float')

    # cumulate events on one frame, depending of the input format
    if AER_format == 'TXYP':
        for event in AER_data:
            event_X = int(event[1])
            event_Y = int(event[2])
            event_POL = int(event[3])

            if event_POL == 1:
                frame[0, event_Y, event_X] += 1  # register ON event on channel 0
            elif event_POL == -1:
                frame[1, event_Y, event_X] += 1  # register OFF event on channel 1

        return frame

    elif AER_format == 'XYTP':
        for event in AER_data:
            event_X = int(event[0])
            event_Y = int(event[1])
            event_POL = int(event[3])

            if event_POL == 1:
                frame[0, event_Y, event_X] += 1
            elif event_POL == -1:
                frame[1, event_Y, event_X] += 1

        return frame


def EVFrametoColorFrame(EVFrame):
    """
    Returns an RGB version of an event frame/histogram, where:
        - RED = ON events
        - BLUE = ON events
        - PINK = ON & OFF events

    :param EVFrame: a 2-channel event frame of shape: (2, H, W).
    :return: color_frame: a 3-channel RGB event frame of shape (H, W, 3), to be directly displayed with cv2.imshow()
    """
    frame_ON = EVFrame[0]
    frame_OFF = EVFrame[1]

    color_frame = np.zeros((260, 346, 3), dtype='uint8')

    ON_mask = (frame_ON > 0) & (frame_OFF == 0)
    OFF_mask = (frame_ON == 0) & (frame_OFF > 0)
    ON_OFF_mask = (frame_ON > 0) & (frame_OFF > 0)

    color_frame[ON_mask] = [255, 0, 0]
    color_frame[OFF_mask] = [0, 0, 255]
    color_frame[ON_OFF_mask] = [255, 25, 255]

    return color_frame


class DENSE(Dataset):
    """
    A dataset class to contain DENSE dataset, a synthetic neuromorphic dataset mimicking MVSEC and containing perfect
    grountruth on outdoor scenes, created with the CARLA simiulator.

     available at http://rpg.ifi.uzh.ch/E2DEPTH.html

    It is a monocular dataset, meaning that while keeping the same API as the MVSEC dataset class, but with
    self.data_right = None

    self.xL and self.xR have shape (num_chunks, num_frames_per_depth_map, 2, 260, 346)
    self.yL has shape (num_chunks, 260, 346)
    """

    def __init__(self, root: str, start_end=(0, -1), num_frames_per_depth_map=1, mirror_time=False, take_log=True):
        print("\n#####################################")
        print("# LOADING AND PREPROCESSING DATASET #")
        print("#####################################\n")

        self.FPS = 30
        self.root = root
        self.num_frames_per_depth_map = num_frames_per_depth_map

        data_path = root + "events/data/"
        label_path = root + "depth/data/"

        xL = []
        yL = []

        # load the data
        data_list = sorted([elem for elem in os.listdir(data_path) if '.npy' in elem])
        label_list = sorted([elem for elem in os.listdir(label_path) if '.npy' in elem])

        # split the data into chunks of num_frames_per_depth_maps frames of spikes
        for AER_filename, label_filename in zip(data_list, label_list):

            AER_data = np.load(root + "events/data/" + AER_filename)
            split_AER_list = splitAERinto(AER_data, num_frames=num_frames_per_depth_map)

            chunk = []
            for AER in split_AER_list:
                frame = AERtoEVFrame(AER, AER_format='TXYP')
                chunk.append(frame)

            label = np.load(root + "depth/data/" + label_filename)
            xL.append(chunk)
            yL.append(label)

        # convert to numpy array for easier manipulation
        xL = np.array(xL)
        yL = np.array(yL)

        # only keep a subset of the sequence
        start_chunk_idx = start_end[0]
        end_chunk_idx = start_end[1]
        xL = xL[start_chunk_idx:end_chunk_idx]
        yL = yL[start_chunk_idx:end_chunk_idx]

        if mirror_time:
            xL_mirr = np.flip(xL, axis=0)  # reverse the order of spike frames
            xL_mirr = np.flip(xL_mirr, axis=1)  # reverse spike polarities because we are going backward in time
            self.data_left = np.concatenate((xL, xL_mirr), axis=0)  # concatenate original and mirrored frame sequences

            self.data_right = None

            yL_mirr = np.flip(yL, axis=0)  # reverse the order of labels
            self.labels = np.concatenate((yL, yL_mirr), axis=0)  # concatenate original and mirrored label sequences

        else:
            self.data_left = xL
            self.data_right = None
            self.labels = yL

    def __len__(self):
        return len(self.data_left)

    def __getitem__(self, index):
        chunk_left = self.data_left[index]
        chunk_right = np.zeros(0)
        label = self.labels[index]
        return [chunk_left, chunk_right], label

    def show(self):
        for chunk in self.data_left:
            for event_frame in chunk:
                color_frame = EVFrametoColorFrame(event_frame)
                cv2.imshow("Cumulated event frame", color_frame)
                cv2.waitKey(int(1000 / (self.FPS * self.num_frames_per_depth_map)))
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = "/home/ulysse/Desktop/PFE CerCo/datasets/DENSE/test/test_sequence_00_town10/"
    dataset = DENSE(root, start_end=(200, 442), num_frames_per_depth_map=1, mirror_time=True)
    print(len(dataset))
    index = 42
    frame_left = dataset[index][0][0]  # shape (2, 260, 346)
    frame_right = dataset[index][0][1]  # None (because monocular)
    label = dataset[index][1]  # shape (260, 346)
    dataset.show()
    print('done')
