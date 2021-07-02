import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

from network.metrics import mask_dead_pixels


def get_img_from_fig(fig, dpi=180):
    """
    A function that returns an image as numpy array from a pyplot figure.

    :param fig:
    :param dpi:
    :return:
    """
    buf = io.BytesIO()

    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_learning(fig, chunk, out_depth_potentials, label, title):
    """
    On a pyplot figure, confront the outputs of the network with the corresponding groundtruths.

    :param fig:
    :param chunk:
    :param out_depth_potentials: a tensor of shape (batchsize, 1, 260, 346)
    :param label:  a tensor of shape (batchsize, 1, 260, 346)
    :return:
    """
    plt.title(title)
    plt.axis('off')

    # 1. Prepare spike histogram for the plot
    frame_ON = chunk[0, :, 0, :].sum(axis=0).cpu().numpy()
    frame_OFF = chunk[0, :, 1, :].sum(axis=0).cpu().numpy()

    frame = np.zeros((260, 346, 3), dtype='int16')

    ON_mask = (frame_ON > 0) & (frame_OFF == 0)
    OFF_mask = (frame_ON == 0) & (frame_OFF > 0)
    ON_OFF_mask = (frame_ON > 0) & (frame_OFF > 0)

    frame[ON_mask] = [255, 0, 0]
    frame[OFF_mask] = [0, 0, 255]
    frame[ON_OFF_mask] = [255, 25, 255]

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.title.set_text('Input spike histogram')
    plt.imshow(frame)
    plt.axis('off')

    # 2. Prepare network predictions for the plot
    out_depth_potentials, label = mask_dead_pixels(out_depth_potentials, label)

    potentials_copy = out_depth_potentials[-1]
    potentials_copy = potentials_copy.detach().cpu().numpy().squeeze()
    error = np.abs(potentials_copy - label[-1].detach().cpu().numpy().squeeze())

    ax1 = fig.add_subplot(1, 4, 2)
    ax1.title.set_text('Prediction')
    plt.imshow(potentials_copy)
    plt.axis('off')

    # 3. Prepare groundtruth map for the plot
    ax2 = fig.add_subplot(1, 4, 3)
    ax2.title.set_text('Groundtruth')
    plt.imshow(label[-1].detach().cpu().numpy().squeeze())
    plt.axis('off')

    # 4. Also plot the error map (error per pixel)
    ax3 = fig.add_subplot(1, 4, 4)
    ax3.title.set_text('Pixel-wise absolute error')
    plt.imshow(error)
    plt.axis('off')

    plt.draw()

    data = get_img_from_fig(fig, dpi=180)

    plt.pause(0.0001)
    plt.clf()

    return data
