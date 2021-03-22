import h5py
import cv2
import numpy as np
from tqdm import tqdm


FRAME_SIZE = (346, 260)
FRAME_W = 346
FRAME_H = 260
FPS = 20


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


def mvsecReadEvents(events):
    """
    Creates a .mp4 video with all cumulated spikes. Blue: OFF, Red: ON

    :param events:
    :return:
    """
    frameSize = (260, 346)
    frame = np.ones(frameSize).astype(np.uint8) * 127
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    mer = 361579  # mean event rate: events/s
    FPS = 60
    listIndX = [int(spk[0]) for spk in events]
    listIndY = [int(spk[1]) for spk in events]
    listTime = [spk[2] for spk in events]
    listPol = [spk[3] for spk in events]
    listTime[:] -= listTime[0]
    frames = []
    currentFrame = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('indoor_flying4_events.mp4', fourcc, FPS, FRAME_SIZE)
    for i in tqdm(range(len(events))):
        while listTime[i] > currentFrame * (1 / FPS):  # put all spikes that occurred (?)
            frames.append(frame)
            frame = np.ones(frameSize).astype(np.uint8) * 127
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            currentFrame += 1
        if listPol[i] == 1:  # ON / BLUE
            frame[listIndY[i], listIndX[i], 0] = 0  # R
            frame[listIndY[i], listIndX[i], 1] = 0  # G
            frame[listIndY[i], listIndX[i], 2] = 255  # B
        else:  # OFF / RED
            frame[listIndY[i], listIndX[i], 0] = 255  # R
            frame[listIndY[i], listIndX[i], 1] = 0  # G
            frame[listIndY[i], listIndX[i], 2] = 0  # B
    for f in frames:
        # cv2.imshow('events', f)
        out.write(f)
        # cv2.waitKey(int(1000 / FPS))
    out.release()


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
    out = cv2.VideoWriter('depth.mp4', fourcc, FPS, (2*346, 3*260))

    for i in range(len(Ldepths_rect)):
        f_rect = np.concatenate((Ldepths_rect[i], Rdepths_rect[i]), axis=1)  # concatenate left and right DMs to show them side by side
        f_raw = np.concatenate((Ldepths_raw[i], Rdepths_raw[i]), axis=1)
        f_blended = np.concatenate((Lblended[i], Rblended[i]), axis=1)

        f = np.concatenate((f_rect, f_raw), axis=0)  # concatenate to show rectified maps at the top and raw at the bottom

        f = np.nan_to_num(f, copy=True, nan=0)  # small processing of the depth maps; DO NOT DO THIS FOR TRAINING A SNN
        f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX)
        f = cv2.cvtColor(f,cv2.COLOR_GRAY2RGB)
        f = np.concatenate((f, f_blended), axis=0)
        f = f.astype(np.uint8)

        cv2.imshow("depth maps: Left | Right ----- Rectified (top) | Raw (bottom)", f)
        out.write(f)
        cv2.waitKey(int(1000 / FPS))

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
    out = cv2.VideoWriter('blended.mp4', fourcc, FPS, (2*346, 260))

    for i in range(len(Ldepths_rect)):
        f = np.concatenate((Lblended[i], Rblended[i]), axis=1)  # concatenate to show left and right images side by side
        cv2.imshow("depth maps / events superposition (provided by the authors) ----- Left | Right", f)
        out.write(f)
        cv2.waitKey(int(1000 / FPS))

    out.release()


def mvsecSpikesAndDepth(Ldepths_rect, Levents, Lblended):
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
    out = cv2.VideoWriter("reconstituted_vs_blended.mp4", fourcc, FPS, (346, 260*2))

    frame = Ldepths_rect[currentFrame]
    frame = np.nan_to_num(frame, copy=True, nan=0)
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = frame.astype(np.uint8)
    for i in tqdm(range(len(Levents))):  # loop over spike events

        # if the time of event i does not exceed the time of the next time step
        # (the first spike event comes 0.2605787s before the first lidar acquisition for this sequence)
        if listTime[i] < 0.2605787 + currentFrame * 1/FPS:

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
            except ValueError as e:
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
    for f, j in zip(frames, range(len(Lblended))):
        f = np.concatenate((f, Lblended[j]), axis=0)  # concatenate to show left and right images side by side
        cv2.imshow("cumulated spikes on depth map", f)
        out.write(f)
        cv2.waitKey(int(1000 / FPS))

    out.release()
    print("saved video sequence to 'reconstituted_vs_blended.mp4'...\n")


def mvsecCumulateSpikesIntoFrames(events, depth_rect, depth_rect_ts, num_frames_per_depth_map=1):
    """
    Cumulates spikes into frames that are synchronized with the labels of the depth labels timestamps.
    Frames will have shape [2 (polarities), W, H], the first channel being for ON events and the second for OFF events.

    Note: By default, spikes are accumulated over frames of duration dt = 1/FPS = 50 ms. In fact,
        dt = 1 / (FPS * num_frames_per_depth_map)
        Equivalence Table :
        ---------------------------------------------
          dt (ms)    |     num_frames_per_depth_map |
        ---------------------------------------------
          50         |         1
          10         |         5
          5          |         10
          1          |         25

    :param events: events with their timestamps being floats (not converted to integer yet)
    :param depth_rect_ts: depth maps
    :param depth_rect_ts: timestamps of the depth maps
    :return: a tensor of shape [# of frames, 2 (polarities), W, H] containing the cumulated spikes, and a tensor of
            shape [# of frames, W, H] containing the corresponding and synchronized depth maps
    """
    assert num_frames_per_depth_map in [1, 2, 5, 10, 25], "num_frames_per_depth_map must divide 50 ! Choose another " \
                                                          "value among [1, 2, 5, 10, 25] ..."
    print("\nCumulating spikes into frames and synchronizing with ground-truth...")
    print("Time interval of each frame: dt = " + str(50/num_frames_per_depth_map) + " ms")

    fps = num_frames_per_depth_map * FPS
    currentFrame = 0
    currentMap = 0
    frames = []
    maps = []

    # spike events variables
    listIndX = [int(spk[0]) for spk in events]
    listIndY = [int(spk[1]) for spk in events]
    listTime = [spk[2] for spk in events]
    listPol = [spk[3] for spk in events]
    first_spike_time = listTime[0]
    listTime[:] -= first_spike_time  # set the first event to time t0=0s
    depth_rect_ts[:] -= first_spike_time  # the first spike happens before the first lidar acquisition

    while depth_rect_ts[0] >= 0:  # create dummy timestamps and labels before the first timestamp
        depth_rect_ts = np.insert(depth_rect_ts, 0, depth_rect_ts[0] - (1 / fps))
        depth_rect = np.insert(depth_rect, 0, depth_rect[0], axis=0)

    frame = np.zeros((2, 260, 346), dtype='float')
    for i in tqdm(range(len(events))):  # loop over spike events

        # if the time of event i does not exceed the timestamp of the next lidar acquisition
        # (the first spike event comes depth_rect_ts[0] before the first lidar acquisition for this sequence)
        if listTime[i] < depth_rect_ts[0] + currentFrame * 1 / fps:

            # register it and mark it on the current frame
            if listIndX[i] < 346 and listIndY[i] < 260:
                if listPol[i] == 1:
                    frame[0, listIndY[i], listIndX[i]] += 1  # register ON event
                else:
                    frame[1, listIndY[i], listIndX[i]] += 1  # register OFF event

        # otherwise, save the currently edited frame, go to the next, and register the event on this new frame as before
        else:
            frames.append(frame)
            maps.append(depth_rect[currentMap])
            currentMap = 1 + currentFrame // num_frames_per_depth_map  # remember that the first depth map has only 1 frame corresponding to it
            currentFrame += 1
            frame = np.zeros((2, 260, 346), dtype='float')

            if listIndX[i] < 346 and listIndY[i] < 260:
                if listPol[i] == 1:
                    frame[0, listIndY[i], listIndX[i]] += 1  # register ON event
                else:
                    frame[1, listIndY[i], listIndX[i]] += 1  # register OFF event

        # break once we treated all events having a label
        # the total number of frames we should have is (1 + (len(depth_ts) - 1) * num_frames_per_depth_map), because we
        # process the first depth map differently, i.e. we cumulate all spikes occuring before the first depth map on
        # the first frame. All other depth maps give each num_frames_per_depth_map depth maps.
        if currentFrame == 1 + (len(depth_rect_ts) - 1) * num_frames_per_depth_map:
            break

    return np.array(frames), np.array(maps)


if __name__ == '__main__':
    # load dataset files
    root = '/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/'
    datafile = root + 'indoor_flying/indoor_flying4_data'
    data = h5py.File(datafile + '.hdf5', 'r')
    datafile_gt = root + 'indoor_flying/indoor_flying4_gt'
    data_gt = h5py.File(datafile_gt + '.hdf5', 'r')

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
    AER_frames, sync_labels = mvsecCumulateSpikesIntoFrames(rect_Levents, Ldepths_rect, Ldepths_rect_ts, 2)
    mvsecSpikesAndDepth(Ldepths_rect, rect_Levents, Lblended)
    #mvsecSpikesAndDepth(Ldepths_rect, Levents, Lblended)
    #mvsecShowBlended(Lblended, Rblended)
    #mvsecShowDepth(Ldepths_rect, Rdepths_rect, Ldepths_raw, Rdepths_raw, Rblended, Lblended)
    #mvsecToVideo(datafile, events, images)
    #mvsecReadEvents(events)

