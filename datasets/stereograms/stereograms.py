import numpy as np
import cv2
from torch.utils.data.dataset import Dataset

# MVSEC resolution
HEIGHT = 260
WIDTH = 346


def fuse_to_analgyph(left_rds, right_rds):
    """
    Fuses two RDS binary tensors to visualize them in Red/Cyan anaglyph.

    Procedure:
     - new image = 2 x left image + 1 x right image
     - assign a color to each pixel depending on their value:
        ~ value of 1 --> red ; pixel is seen only by one ye
        ~ value of 2 --> cyan ; pixel is seen by both eyes
        ~ value of 3 --> white ; pixel is seen by both eyes
    """
    # fusion of both images
    fused = 2*left_rds + right_rds

    red_mask = (fused == 1)
    cyan_mask = (fused == 2)
    white_mask = (fused == 3)

    # anaglyph image (RGB)
    anaglyph_frame = np.zeros((260, 346, 3), dtype='uint8')

    anaglyph_frame[red_mask] = [255, 0, 0]
    anaglyph_frame[cyan_mask] = [0, 255, 255]
    anaglyph_frame[white_mask] = [255, 255, 255]

    # cv2.imshow() reads BGR format by default
    anaglyph_frame = cv2.cvtColor(anaglyph_frame, cv2.COLOR_RGB2BGR)

    return anaglyph_frame


def generate_random_spikes(size: tuple, appearance_rate: float = 0.5):
    """
    Returns a random (H, W) binary array.

    :param size: the size of the stereogram to produce. Must be a (H, W) tuple
    :param appearance_rate: the probability of a pixel being a one, i.e., the probability to emit a spike at each pixel
    :return: a random (H, W) binary array
    """
    H, W = size
    return (np.random.rand(1, H, W) > (1 - appearance_rate)).astype('int')


def shift(frame, disparity_map, amount: int = 5):
    """
    Shifts the portion of the random binary frame corresponding to the disparity map by a certain amount. The bigger
    this amount, the bigger the resulting disparity.

    TODO: support grayscale disparity maps. The higher the disparity value, the bigger the amount of shift. The amount
     of shift could be proportional to the disparity value in the disparity map

    TODO: pour l'instant, le pattern est un carré statique de côté 40 pixels situé au centre de l'image. Utiliser un
     pattern qui bouge dans le temps !

    :param frame: a binary frame of shape (H, W)
    :param disparity_map: a binary tensor of shape (H, W)
    :param amount: integer amount of shift to apply, expressed in pixels
    :return: shifted binary frame of shape (W, H)
    """
    _, H, W = frame.shape
    shifted_frame = np.copy(frame)
    shifted_frame[0, H//2-40:H//2+40, W//2-40+amount:W//2+40+amount] = frame[0, H//2-40:H//2+40, W//2-40:W//2+40]
    return shifted_frame


def simulate_RDS(num_steps: int, disparity_map, appearance_rate: float, shift_amount: int, lifetime: int, out_path: str):
    """
    At every timestep, appearance_rate new points randomly appear in the ON channel; after lifetime timesteps, their
    disparition is significated by a spike at the same location in the OFF channel.

    For more insights on RDSs, refer to https://www.ime.usp.br/~otuyama/stereogram/basic/index.html

    :param num_steps: the number of timesteps to simulate, or the length of the produced dataset
    :param disparity_map: a disparity map of shape (H, W) to render in the stereogram
    :param appearance_rate: the number of ON spikes at each timestep (i.e., on ech frame)
    :param shift_amount: the lateral shift to apply to pixels corresponding to the disparity map. The bigger the distance
     between both eyes, the bigger the shift
    :param lifetime: the number of timesteps after which an OFF event follows an ON event at the same location
    :param out_path: path of the output file
    :return: left and right streams of spike frames, each being a list on num_steps tensors of shape (2, H, W). Channel 0
     is for ON spikes and channel 1 for OFF spikes.
    """
    size = (HEIGHT, WIDTH)

    xL = []
    xR = []

    i = 0
    ON_left_list = []
    ON_right_list = []

    for t in range(num_steps+lifetime):

        frame_ON_left = generate_random_spikes(size, appearance_rate)
        frame_ON_right = shift(frame_ON_left, disparity_map, shift_amount)

        ON_left_list.append(frame_ON_left)
        ON_right_list.append(frame_ON_right)

        # ON spikes that occurred lifetime timesteps before current ON spikes become current OFF spikes
        if t >= lifetime:
            EVframe_left = np.concatenate((frame_ON_left, ON_left_list[i]), axis=0)
            EVframe_right = np.concatenate((frame_ON_right, ON_right_list[i]), axis=0)
            i += 1
        elif t < lifetime:
            EVframe_left = np.concatenate((frame_ON_left, np.zeros_like(frame_ON_left)), axis=0)
            EVframe_right = np.concatenate((frame_ON_right, np.zeros_like(frame_ON_right)), axis=0)

        xL.append(EVframe_left)
        xR.append(EVframe_right)

    # discard the lifetime first frames of the sequence, since they have no event on their OFF channel
    xL = xL[lifetime:]
    xR = xR[lifetime:]

    return xL, xR


class NeuromorphicRDS(Dataset):
    """
    A dataset class to contain n0euromorphic Random Dot Stereograms (RDS).

    The latter are used to compare the activations of the encoder layers if our binocular network with what is observed
    experimentally in the early layers of the visual cortex.
    """

    def __init__(self, num_steps, disparity_map, appearance_rate, shift_amount, lifetime, out_path):
        xL, xR = simulate_RDS(num_steps, disparity_map, appearance_rate, shift_amount, lifetime, out_path)
        self.data_left = xL
        self.data_right = xR

    def __getitem__(self, index):
        frame_left = self.data_left[index]
        frame_right = self.data_right[index]
        return frame_left, frame_right

    def __len__(self):
        return len(self.data_left)

    def show(self):
        for left_event_frame, right_event_frame in zip(self.data_left, self.data_right):
            anaglyph = fuse_to_analgyph(left_event_frame[0], right_event_frame[0])
            Lf = (left_event_frame[0] * 255).astype('uint8')
            Rf = (right_event_frame[0] * 255).astype('uint8')
            limit = np.zeros((260, 20), dtype='uint8')
            LRf = np.concatenate((Lf, limit, Rf), axis=1)
            cv2.imshow("Left/Right ON events", LRf)
            cv2.imshow("ON events (Red/Cyan anaglyph)", anaglyph)
            cv2.waitKey(int(1000 / 20))
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = "."
    dataset = NeuromorphicRDS(num_steps=200, disparity_map=None, appearance_rate=0.5, shift_amount=5, lifetime=5, out_path=".")
    index = 42
    frame_left = dataset[index][0][0]  # shape (2, 260, 346)
    frame_right = dataset[index][0][1]
    dataset.show()
    print('done')
