import io
import time
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import functional

from mvsec import MVSEC
from network.models import SpikeFlowNetLike, SpikeFlowNetLike_exp, SpikeFlowNetLike_cext
from network.metrics import mask_dead_pixels, MeanDepthError, Total_Loss


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()

    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_learning(fig, chunk, out_depth_potentials, label):
    frame_ON = chunk[0][-1].sum(axis=0).cpu().numpy()
    frame_OFF = chunk[0][-1].sum(axis=0).cpu().numpy()

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

    potentials_copy = out_depth_potentials
    potentials_copy = potentials_copy.detach().cpu().numpy().squeeze()
    error = np.abs(potentials_copy - label.detach().cpu().numpy().squeeze())

    ax1 = fig.add_subplot(1, 4, 2)
    ax1.title.set_text('Prediction')
    plt.imshow(potentials_copy)
    plt.axis('off')

    ax2 = fig.add_subplot(1, 4, 3)
    ax2.title.set_text('Groundtruth')
    plt.imshow(label.detach().cpu().numpy().squeeze())
    plt.axis('off')

    ax3 = fig.add_subplot(1, 4, 4)
    ax3.title.set_text('Pixel-wise absolute error')
    plt.imshow(error)
    plt.axis('off')

    plt.draw()

    #data = get_img_from_fig(fig, dpi=180)

    plt.pause(0.0001)
    plt.clf()

    # cv2.imshow("cv2", data)
    # cv2.waitKey(int(1000 / (20 * 5)))

    """
    potentials_copy = out_depth_potentials
    potentials_copy = potentials_copy.detach().cpu().numpy().squeeze()
    potentials_copy = ((potentials_copy / potentials_copy.max()) * 255).astype('uint8')
    potentials_copy = cv2.applyColorMap(potentials_copy, cv2.COLORMAP_BONE)
    cv2.imshow("test", potentials_copy)
    cv2.waitKey(int(1000 / (20*5)))
    """


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

T = 10  # (!) don't choose T too big because of memory limitations (!)
nfpdm = 5
batch_size = 1
learning_rate = 0.0001
n_epochs = 10


dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                num_frames_per_depth_map=nfpdm,
                normalization=None,
                take_log=False,
                show_sequence=False)
train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False,
                                                pin_memory=True)


net = SpikeFlowNetLike_cext(tau=7, T=T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf')).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# TODO: add a learning rate scheduler. see cosine annealing
# TODO: add a Tensorboard logger

logfile = open("./checkpoints/training_logs.txt", "w+")
plt.ion()
fig = plt.figure()

currentFrame = 1
for epoch in range(n_epochs):
    start_time = time.time()
    running_loss = 0.0
    running_MDE = 0.0

    # reset the potential of all neurons before each sequence (i.e., epoch)
    functional.reset_net(net)

    # I think we have a case of TBPTT with k1 = k2 = 1 * num_frames_per_depth_map = 5 timesteps.
    # That is, the loss is computed and gradients backpropagated at each depth groundtruth timestamp. Depth groundtruth
    # timestamps are separated by num_frames_per_depth_map = 5 frames = 5 timesteps.
    net.train()
    for chunk, label in tqdm(train_data_loader):

        chunk = chunk[0].to(device=device, dtype=torch.float)  # only take left DVS data for monocular depth estimation
        label = label.to(device)

        out_depth_potentials = net(chunk)

        # mask all invalid pixels, corresponding to pixels of the groundtruth with values of 0 or 255
        # these pixels become 0 in both prediction and groundtruth (so that residuals become 0 too),
        # the others remain as they are
        # TODO ? We could flatten the tensors instead of setting invalid pixels to 0. But if we flatten,
        #  gradient matching loss not possible anymore !
        label = label.squeeze()
        out_depth_potentials = out_depth_potentials.squeeze()
        out_depth_potentials, label = mask_dead_pixels(out_depth_potentials, label)

        show_learning(fig, chunk, out_depth_potentials, label)

        # TODO: use custom loss (later)
        #loss = F.mse_loss(out_depth_potentials, label)
        #loss = Total_Loss(out_depth_potentials, label, alpha=0.)
        loss = MeanDepthError(out_depth_potentials, label)

        loss.backward(retain_graph=True)  # retain_graph can be False for k1 = k2 cases of the TBPTT
        optimizer.step()

        net.detach()
        optimizer.zero_grad()

        running_loss += loss.item() * chunk.size(0)

    epoch_loss = running_loss / (len(train_data_loader) / n_epochs)

    functional.reset_net(net)

    # evaluate performances after each epoch
    net.eval()
    with torch.no_grad():
        for chunk, label in tqdm(train_data_loader):
            chunk = chunk[0].to(device=device,
                                dtype=torch.float)  # only take left DVS data for monocular depth estimation
            label = label.to(device)

            out_depth_potentials = net(chunk)

            # mask all invalid pixels, corresponding to pixels of the groundtruth with values of 0 or 255
            # these pixels become 0 in both prediction and groundtruth (so that residuals become 0 too),
            # the others remain as they are
            label = label.squeeze()
            out_depth_potentials = out_depth_potentials.squeeze()
            out_depth_potentials, label = mask_dead_pixels(out_depth_potentials, label)

            net.detach()  # why do I need to use net.detach() even in eval mode to avoid CUDA out of memory errors ?

            running_MDE += MeanDepthError(out_depth_potentials, label)

        epoch_MDE = running_MDE / len(train_data_loader)

    end_time = time.time()
    epoch_summary = "Epoch: {}, Loss: {}, Mean Depth Error: {}, Time: {}\n".format(epoch, epoch_loss, epoch_MDE, end_time-start_time)
    print(epoch_summary)
    logfile.write(epoch_summary)

# save model
torch.save(net.state_dict(), "./checkpoints/spikeflownet_snn.pth")

# close log file
logfile.close()

