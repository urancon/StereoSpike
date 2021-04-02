import h5py
import cv2
import numpy as np
from tqdm import tqdm

FRAME_SIZE = (346, 260)
FRAME_W = 346
FRAME_H = 260
LIDAR_FPS = 20
DAVIS_FPS = 50


def mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path):
    """
    Loads the rectification maps for further calibration of DAVIS' spike events coordinates.

    :param Lx_path: path of the .txt file containing the mapping of the x coordinate for the left DAVIS camera
    :param Ly_path:                     ..                              y        ..          left
    :param Rx_path:                     ..                              x        ..          right
    :param Ry_path:                     ..                              y        ..          right
    :return: all corresponding mapping matrices in the form of a numpy array
    """
    print("\nloading rectification maps...")
    Lx_map = np.loadtxt(Lx_path)
    Ly_map = np.loadtxt(Ly_path)
    Rx_map = np.loadtxt(Rx_path)
    Ry_map = np.loadtxt(Ry_path)
    return Lx_map, Ly_map, Rx_map, Ry_map


def mvsecRectifyEvents(events, x_map, y_map):
    """
    Rectifies the spatial coordinates of the input spike events in accordance to the given mapping matrices.
    CAUTION: make sure events and maps correspond to the same side (DAVIS/left or DAVIS/right) !

    :param events: a list of spike events to the format [X, Y, TIME, POLARITY]
    :param x_map: np.array obtained by mvsecLoadRectificationMaps() function
    :param y_map:                       ..
    :return: rectified events, in the same format as the input events
    """
    print("\nrectifying spike coordinates...")
    rect_events = []
    for event in tqdm(events):
        x = int(event[0])
        y = int(event[1])
        x_rect = x_map[y, x]
        y_rect = y_map[y, x]
        rect_events.append([x_rect, y_rect, event[2], event[3]])

    # convert to np.array and remove spikes falling outside of the Lidar field of view (fov)
    rect_events = np.array(rect_events)
    rect_events = rect_events[(rect_events[:, 0] >= 0)
                              & (rect_events[:, 0] <= 346)
                              & (rect_events[:, 1] >= 0)
                              & (rect_events[:, 1] <= 260)]
    return rect_events


def mvsecFloatToInt(events):
    """
    Converts an event array elements from floats to integers;
    first multiply the times by a large value to not lose information. DAVIS cameras have a a resolution of around 10us,
    so this implies multiplying the timestamps by more than 1e6.
    Also, rectified pixels values calculated by mvsecRectifyEvents() are floats, so it is a good thing to finally round
    them to the nearest int for later use.

    :param events: a list of spike events to the format [X, Y, TIME, POLARITY]
    :return: events whith integer spatial and temporal coordinates, in the same format as the input events
    """
    #
    events[:, 2] = events[:, 2] * 1e7
    events = np.rint(events).astype(int)
    return events


def mvsecShowDepth(Ldepths_rect, Rdepths_rect, Ldepths_raw, Rdepths_raw, Rblended, Lblended):
    """
    Reconstitutes a video file from the Lidar depth acquisitions.
    CAUTION: depth maps were processed for the sake of data visualization only !

    :param Ldepths_rect:
    :param Rdepths_rect:
    :param Ldepths_raw:
    :param Rdepths_raw:
    :param Lblended:
    :param Rblended:
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('depth.mp4', fourcc, LIDAR_FPS, (2 * 346, 3 * 260))

    for i in range(len(Ldepths_rect)):
        f_rect = np.concatenate((Ldepths_rect[i], Rdepths_rect[i]),
                                axis=1)  # concatenate left and right DMs to show them side by side
        f_raw = np.concatenate((Ldepths_raw[i], Rdepths_raw[i]), axis=1)
        f_blended = np.concatenate((Lblended[i], Rblended[i]), axis=1)

        f = np.concatenate((f_rect, f_raw),
                           axis=0)  # concatenate to show rectified maps at the top and raw at the bottom

        f = np.nan_to_num(f, copy=True, nan=0)  # small processing of the depth maps; DO NOT DO THIS FOR TRAINING A SNN
        f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX)
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        f = np.concatenate((f, f_blended), axis=0)
        f = f.astype(np.uint8)

        cv2.imshow("depth maps: Left | Right ----- Rectified (top) | Raw (bottom)", f)
        out.write(f)
        cv2.waitKey(int(1000 / LIDAR_FPS))

    out.release()


def mvsecShowBlended(Lblended, Rblended):
    """
    Shows a preview (provided by the authors of the dataset) of the sequence. Consists of the superposition of depth
    maps and events

    :param Lblended:
    :param Rblended:
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('blended.mp4', fourcc, LIDAR_FPS, (2 * 346, 260))

    for i in range(len(Lblended)):
        f = np.concatenate((Lblended[i], Rblended[i]), axis=1)  # concatenate to show left and right images side by side
        cv2.imshow("depth maps / events superposition (provided by the authors) ----- Left | Right", f)
        out.write(f)
        cv2.waitKey(int(1000 / LIDAR_FPS))

    out.release()


def mvsecSpikesAndDepth(Ldepths_rect, Levents, Lblended=None):
    """
    Reconsitutes a video file from Lidar depth acquisitions, superpose cumulated spike events between frames, and
    compares the result with the "blended" data provided by the authors of MVSEC dataset.

    :param Ldepths_rect:
    :param Levents:  a list of spike events to the format [X, Y, TIME, POLARITY]. TIME values MUST BE FLOATS !
    :param Lblended:
    :return:
    """
    print("\ncumulating spikes and editing depth maps...")
    # general variable related to the video
    currentFrame = 0
    frames = []

    # spike events variables
    listIndX = [int(spk[0]) for spk in Levents]
    listIndY = [int(spk[1]) for spk in Levents]
    listTime = [spk[2] for spk in Levents]
    listPol = [spk[3] for spk in Levents]
    listTime[:] -= listTime[0]  # set the first event to time t0=0s

    # prepare to save a video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_shape = (346, 260 * 2) if Lblended is not None else (346, 260)
    out = cv2.VideoWriter("reconstituted_vs_blended.mp4", fourcc, LIDAR_FPS, video_shape)

    frame = Ldepths_rect[currentFrame]
    frame = np.nan_to_num(frame, copy=True, nan=0)
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = frame.astype(np.uint8)
    for i in tqdm(range(len(Levents))):  # loop over spike events

        # if the time of event i does not exceed the time of the next time step
        # (the first spike event comes 0.2605787s before the first lidar acquisition for this sequence)
        if listTime[i] < listTime[0] + currentFrame * 1 / LIDAR_FPS:

            # register it and mark it on the current frame
            if listIndX[i] < 346 and listIndY[i] < 260:
                if listPol[i] == 1:
                    frame[listIndY[i], listIndX[i]] = [0, 0, 255]  # ON / BLUE
                else:
                    frame[listIndY[i], listIndX[i]] = [255, 0, 0]  # OFF / RED

        # otherwise, save the currently edited frame, go to the next, and register the event on this new frame as before
        else:
            frames.append(frame)
            currentFrame += 1
            try:
                frame = Ldepths_rect[currentFrame]
                frame = np.nan_to_num(frame, copy=True, nan=0)
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                frame = frame.astype(np.uint8)
            except IndexError as e:
                print(e)
                print("Some events remained after the last Lidar acquisition. Ignoring them...")
                break

            if listIndX[i] < 346 and listIndY[i] < 260:
                if listPol[i] == 1:
                    frame[listIndY[i], listIndX[i]] = [0, 0, 255]  # ON / BLUE
                else:
                    frame[listIndY[i], listIndX[i]] = [255, 0, 0]  # OFF / RED

    # Finally, just show all frames edited with spike events
    print("showing video sequence...\n")
    for f, j in zip(frames, range(len(frames))):
        if Lblended is not None:
            f = np.concatenate((f, Lblended[j]), axis=0)  # concatenate to show left and right images side by side
        cv2.imshow("cumulated spikes on depth map", f)
        out.write(f)
        cv2.waitKey(int(1000 / LIDAR_FPS))

    out.release()
    cv2.destroyAllWindows()
    print("saved video sequence to 'reconstituted_vs_blended.mp4'...\n")


def mvsecCumulateSpikesIntoFrames(events, depth_rect, depth_rect_ts, num_frames_per_depth_map=1):
    """
    Cumulates spikes into frames that are synchronized with the labels of the depth labels timestamps.
    Frames will have shape [2 (polarities), W, H], the first channel being for ON events and the second for OFF events.

    Note: By default, spikes are accumulated over frames of duration dt = 1/LIDAR_FPS = 50 ms. In fact,
        dt = 1 / (LIDAR_FPS * num_frames_per_depth_map)
        Equivalence Table :
        ---------------------------------------------
          dt (ms)    |     num_frames_per_depth_map |
        ---------------------------------------------
          50         |         1
          10         |         5
          5          |         10
          1          |         25

    :param events: events with their timestamps being floats (not converted to integer yet) EVENTS: X Y TIME POLARITY
    :param depth_rect_ts: depth maps
    :param depth_rect_ts: timestamps of the depth maps
    :return: a tensor of shape [# of frames, 2 (polarities), W, H] containing the cumulated spikes, and a tensor of
            shape [# of frames, W, H] containing the corresponding and synchronized depth maps
    """
    assert num_frames_per_depth_map in [1, 2, 5, 10, 25], "num_frames_per_depth_map must divide 50 ! Choose another " \
                                                          "value among [1, 2, 5, 10, 25] ..."
    print("\nCumulating spikes into frames and synchronizing with ground-truth...")
    print("Time interval of each frame: dt = " + str(50 / num_frames_per_depth_map) + " ms")

    fps = num_frames_per_depth_map * LIDAR_FPS
    chunksequence = []
    maps = []

    # remove the temporal offset on timestamps; the first spike happens before the first lidar acquisition
    first_spike_time = events[0, 2]
    events[:, 2] -= first_spike_time
    depth_rect_ts[:] -= first_spike_time

    for numchunk in tqdm(range(len(depth_rect_ts))):  # we have as many chunks as groundtruth maps

        chunk = []

        for numframe in range(num_frames_per_depth_map):  # we get num_frames_per_depth_map frames per chunk

            # events that occur between the timestamps of the current and the next frame
            start_ts = numchunk*num_frames_per_depth_map*1/fps + numframe*1/fps
            end_ts = numchunk*num_frames_per_depth_map*1/fps + numframe*1/fps + 1/fps
            filt_events = events[(events[:, 2] > start_ts) & (events[:, 2] < end_ts)]

            # create new 2 channel frame to accumulate spikes on
            frame = np.zeros((2, 260, 346), dtype='float')

            for numevent in range(len(filt_events)):

                event_X = int(filt_events[numevent, 0])
                event_Y = int(filt_events[numevent, 1])
                event_POL = filt_events[numevent, 3]

                if event_POL == 1:
                    frame[0, event_Y, event_X] += 1  # register ON event on channel 0
                else:
                    frame[1, event_Y, event_X] += 1  # register OFF event on channel 1

            chunk.append(frame)

        chunksequence.append(chunk)
        maps.append(depth_rect[numchunk])

    return np.array(chunksequence), np.array(maps)


def mvsecToVideo(images):
    """
    Produces a video file with DAVIS grayscale images.
    This allows to visualize more easily the dataset footage.

    :param images: DAVIS left or right image
    :return:
    """
    print("\nMerging frames of the sequence into a video for visualization...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('grayscale.mp4', fourcc, DAVIS_FPS, (346, 260))

    for i in tqdm(range(len(images))):
        f = images[i]
        f = np.nan_to_num(f, copy=True,
                          nan=255)  # small processing of the depth maps; DO NOT DO THIS FOR TRAINING A SNN
        f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX)
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        f = f.astype(np.uint8)

        cv2.imshow("grayscale", f)
        out.write(f)
        cv2.waitKey(int(1000 / DAVIS_FPS))

    out.release()


if __name__ == '__main__':
    # load dataset files
    root = '/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/'
    datafile = root + 'indoor_flying/indoor_flying1_data'
    data = h5py.File(datafile + '.hdf5', 'r')
    datafile_gt = root + 'indoor_flying/indoor_flying1_gt'
    data_gt = h5py.File(datafile_gt + '.hdf5', 'r')

    # raw grayscale images
    images = data['davis']['left']['image_raw']
    mvsecToVideo(images)

    # depth maps
    Ldepths_rect = data_gt['davis']['left']['depth_image_rect']  # RECTIFIED / LEFT
    Rdepths_rect = data_gt['davis']['right']['depth_image_rect']  # RECTIFIED / RIGHT
    Ldepths_raw = data_gt['davis']['left']['depth_image_raw']  # RAW / LEFT
    Rdepths_raw = data_gt['davis']['right']['depth_image_raw']  # RAW / RIGHT
    Ldepths_rect_ts = np.array(data_gt['davis']['left']['depth_image_rect_ts'])
    Rdepths_rect_ts = np.array(data_gt['davis']['right']['depth_image_rect_ts'])

    # blended: Visualization of all events from the left DAVIS that are 25ms from each left depth map superimposed on
    # the depth map. Gives a preview of what each sequence looks like. Provided by the authors of the dataset.
    Lblended = data_gt['davis']['left']['blended_image_rect']
    Rblended = data_gt['davis']['right']['blended_image_rect']

    # get the events
    Levents = np.array(data['davis']['left']['events'])  # EVENTS: X Y TIME POLARITY
    Revents = np.array(data['davis']['right']['events'])  # EVENTS: X Y TIME POLARITY

    # rectify the coordinates of spike events
    Lx_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_x_map.txt'
    Ly_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_left_y_map.txt'
    Rx_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_x_map.txt'
    Ry_path = root + 'indoor_flying/indoor_flying_calib/indoor_flying_right_y_map.txt'
    Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
    rect_Levents = mvsecRectifyEvents(Levents, Lx_map, Ly_map)
    rect_Revents = mvsecRectifyEvents(Revents, Rx_map, Ry_map)

    # show data in a video file (cumulated spikes between frames)
    mvsecToVideo(images)
    # AER_frames, sync_labels = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts, 2)
    # mvsecSpikesAndDepth(Ldepths_rect, rect_Levents, Lblended)
    # mvsecShowBlended(Lblended, Rblended)
    # mvsecShowDepth(Ldepths_rect, Rdepths_rect, Ldepths_raw, Rdepths_raw, Rblended, Lblended)
