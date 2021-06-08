import io
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from spikingjelly.clock_driven import functional

from network.metrics import MeanDepthError, Total_Loss, mask_dead_pixels

from datasets.MVSEC import MVSEC, shuffled_MVSEC
from network.models import SpikeFlowNetLike

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


###########################
# VISUALIZATION FUNCTIONS #
###########################

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


plt.ion()
fig = plt.figure()

######################
# GENERAL PARAMETERS #
######################

nfpdm = 1  # (!) don't choose it too big because of memory limitations (!)
batchsize = 1
take_log = False
learning_rate = 0.001
weight_decay = 0.0
n_epochs = 10
show = True

########
# DATA #
########

shuffled_dataset = shuffled_MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/', scenario='indoor_flying', case='2',
                                  num_frames_per_depth_map=nfpdm, warmup_chunks=5, train_chunks=5,
                                  mirror_time=False, take_log=take_log, show_sequence=False)
train_indices = list(range(0, 50))
training_set = torch.utils.data.Subset(shuffled_dataset, train_indices)
train_data_loader = torch.utils.data.DataLoader(dataset=training_set,
                                                batch_size=1,
                                                shuffle=True,
                                                drop_last=True,
                                                pin_memory=True)

sequential_dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/', scenario='indoor_flying', case='1',
                           num_frames_per_depth_map=nfpdm,
                           mirror_time=False, take_log=take_log, show_sequence=False)
test_data_loader = torch.utils.data.DataLoader(dataset=sequential_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=False,
                                               pin_memory=True)

###########
# NETWORK #
###########

net = SpikeFlowNetLike(tau=5.,
                       v_threshold=1.0,
                       v_reset=0.0,
                       v_infinite_thresh=float('inf'),
                       final_activation=torch.abs,  # torch.nn.Identity(), #torch.nn.ReLU(),
                       use_plif=True
                       ).to(device)

################
# OPTIMIZATION #
################

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 15], gamma=0.1)
loss_module = Total_Loss

logfile = open("./results/checkpoints/training_logs.txt", "w+")

############
# TRAINING #
############


for epoch in range(n_epochs):

    running_train_loss = 0
    running_train_MDE = 0
    running_test_loss = 0
    running_test_MDE = 0

    net.train()
    start_time = time.time()
    for init_pots, warmup_chunks, train_chunks, label in tqdm(train_data_loader):
        # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_chunks*nfpdm, 2, 260, 346)
        warmup_chunks = warmup_chunks.view(batchsize, warmup_chunks.shape[1] * warmup_chunks.shape[2], 2, 260, 346).to(device,
                                                                                                               dtype=torch.float)
        train_chunks = train_chunks.view(batchsize, train_chunks.shape[1] * train_chunks.shape[2], 2, 260, 346).to(device,
                                                                                                           dtype=torch.float)
        init_pots = init_pots.to(device)
        label = label.to(device)

        # initialize output potentials
        functional.reset_net(net)
        net.set_init_depths_potentials(init_pots)

        # let intermediate neurons "warm up" and reach a steady state before "real" training
        with torch.no_grad():
            net(warmup_chunks)

        # forward pass a long sequence of chunks
        pred = net(train_chunks)

        # calculate loss and update weights with BPTT
        loss = loss_module(pred, label)
        loss.backward()
        optimizer.step()

        # save metrics
        running_train_loss += loss.item() * train_chunks.size(0)
        running_train_MDE += MeanDepthError(pred, label)

    # process saved metrics
    epoch_train_loss = running_train_loss / len(train_data_loader)
    epoch_train_MDE = running_train_MDE / len(train_data_loader)
    epoch_train_time = start_time - time.time()
    train_epoch_summary = "Epoch: {}, Training Loss: {}, Training Mean Depth Error: {}, Time: {}\n".format(epoch,
                                                                                                           epoch_train_loss,
                                                                                                           epoch_train_MDE,
                                                                                                           epoch_train_time)

    # initialize network for evaluation
    functional.reset_net(net)
    _, _, start_pots = next(iter(test_data_loader))
    start_pots = start_pots.to(device)
    net.set_init_depths_potentials(start_pots)

    net.eval()
    start_time = time.time()
    with torch.no_grad():
        for i, (chunk_left, chunk_right, label) in enumerate(tqdm(test_data_loader)):

            chunk = chunk_left.to(device, dtype=torch.float)
            label = label.to(device)

            pred = net(chunk)
            net.detach()

            running_test_loss += loss_module(pred, label)
            running_test_MDE += MeanDepthError(pred, label)

            if show:
                title = 'EPOCH {} - {}'.format(epoch, 'TEST')
                viz = show_learning(fig, chunk, pred, label, title)
                viz = Image.fromarray(viz)
                viz.save("./results/visualization/pngs/epoch{}_chunk{}_1.png".format(epoch, len(train_data_loader) + i))

    epoch_test_loss = running_test_loss / len(test_data_loader)
    epoch_test_MDE = running_test_MDE / len(test_data_loader)
    epoch_test_time = start_time - time.time()
    test_epoch_summary = "Epoch: {}, Test Loss: {}, Test Mean Depth Error: {}, Time: {}\n".format(epoch,
                                                                                                  epoch_test_loss,
                                                                                                  epoch_test_MDE,
                                                                                                  epoch_test_time)
    print(train_epoch_summary + test_epoch_summary)
    logfile.write(train_epoch_summary + test_epoch_summary)

    # save model if better results
    if epoch_test_MDE < net.get_max_accuracy():
        print("Best performances so far: saving model...\n")
        torch.save(net.state_dict(), "./results/checkpoints/spikeflownet_snn.pth")
        net.update_max_accuracy(epoch_test_MDE)

    net.increment_epoch()

    scheduler.step()
