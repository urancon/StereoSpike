import io
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from spikingjelly.clock_driven import functional

from network.metrics import mask_dead_pixels, MeanDepthError, Total_Loss, GradientMatching_Loss, MultiScale_GradientMatching_Loss


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


def show_learning(fig, chunk, out_depth_potentials, label):
    """
    On a pyplot figure, confront the outputs of the network with the corresponding groundtruths.

    :param fig:
    :param chunk:
    :param out_depth_potentials:
    :param label:
    :return:
    """
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

    #data = get_img_from_fig(fig, dpi=180)

    plt.pause(0.0001)
    plt.clf()

    # cv2.imshow("cv2", data)
    # cv2.waitKey(int(1000 / (20 * 5)))


class ApproximatedTBPTT:
    """
    Approximated version of the Truncated Propagation Through Time (TBPTT) with k2 > k1, or more precisely, k2 = n*k1

    EXAMPLE (from Tim):
    -------------------
        Take the MVSEC example with dt = 10 ms and a label every 50 ms, so the real k1 is 5.
        Say you want k2 =  4k1 = 20 (200ms)

        First train on the whole sequence with k1 = k2 = 20.
        This is easy to do, but the problem is that only one every four label is used (4th, 8th, 12th…)

        Then delete the first 50ms of the sequence, and redo the training with k1 = k2 = 20.
        So different labels will be used (5th, 9th, 13th…)

        Then again delete the first 50ms of the remaining sequence (so 100ms deleted in total), and redo the training with k1 = k2 = 20.
        The labels (6th, 10th, 14th, …) will be used.

        Then again delete the first 50ms of the remaining sequence (so 150ms deleted in total), and redo the training with k1 = k2 = 20.
        The labels (7th, 11th, 15th, …) will be used.

        (In practice, I think this four trainings could be done in one batch)

        So in the end, all the labels are used, and the backward does unroll 20 time steps.

    PARTICULAR CASES:
    -----------------
        To train like we did initially, that is, TBPTT with k1 = k2 = 1 chunk = 5 timesteps, use N=1 and nfpdm=5.
        To backpropagate on a longer temporal context, you could increase N. For instance, the preceding example section
         corresponds to the case N=1 and nfpdm=5.

    TODO: add a tensorboard logger
    """

    def __init__(self, one_chunk_module, loss_module, optimizer, lr_scheduler, n_epochs, device):
        self.one_chunk_module = one_chunk_module
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.n_epochs = n_epochs
        self.device = device

    def train(self, train_data_loader, nfpdm=5, N=1, learn_on_log=False, show=False):

        print("\n################")
        print("#   TRAINING   #")
        print("################\n")

        logfile = open("./checkpoints/training_logs.txt", "w+")

        print("\nTraining with approximated TBPTT procedure: {} frames between labels, k1=1, k2=N*K1, N={}\n".format(nfpdm, N))
        logfile.write("Training with approximated TBPTT procedure: {} frames between labels, k1=1, k2=N*K1, N={}\n\n".format(nfpdm, N))

        if show:
            plt.ion()
            fig = plt.figure()

        min_MDE = 100.0  # arbitrarily large value to account for worst performances at initialization

        for epoch in range(self.n_epochs):
            start_time = time.time()
            running_loss = 0.0
            running_MDE = 0.0

            # reset the potential of all neurons before each sequence (i.e., epoch)
            functional.reset_net(self.one_chunk_module)

            # set the potentials of output IF neurons to an ideal state (i.e. the first gt of the sequence) for a good
            # start. The idea is to not use all the network's expression capacity just to catch up with the ground truth
            # at the beginning of the sequence, but rather learn the smaller depth changes occurring during the
            # sequence.
            # i.e., learn the "steady" state, not the "transient".
            _, start_pots = next(iter(train_data_loader))
            start_pots = start_pots.repeat((N, 1, 1)).to(self.device)
            self.one_chunk_module.predict_depth[-1].v = start_pots

            chunk_sequence = []
            label_sequence = []
            chunk_batch = []
            label_batch = []

            self.one_chunk_module.train()
            for i, (chunk, label) in enumerate(tqdm(train_data_loader)):

                chunk = chunk[0].to(device=self.device, dtype=torch.float)  # only take left DVS data for monocular depth estimation
                label = label.to(self.device)

                chunk_sequence.append(chunk)
                label_sequence.append(label)

                if (i+1) % (2*N-1) == 0:

                    # the batch dimension is equal to N, so a batch is composed of N shifted chunk sequences
                    # e.g. N=3, chunk_sequence=[0, 1, 2, 3, 4] (len: 5=2N-1)
                    # --> chunk_batch=[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
                    # --> label_batch=[2, 3, 4]
                    for j in range(N):
                        chunk_batch.append(torch.cat(chunk_sequence[j:N+j], dim=1))  # concatenate chunks along frame (time) dimension  [1, nfpdm, 2, H, W] --> [1, N*nfpdm, 2, H, W]
                        label_batch.append(label_sequence[N-1+j])  # get the label corresponding to the chunk sequence --> [1, H, W]
                    chunk_batch = torch.cat(chunk_batch, dim=0)  # concatenate along batch dimension --> [N, N*nfpdm, 2, H, W]
                    label_batch = torch.cat(label_batch, dim=0)  # concatenate along batch dimension --> [N, H, W]

                    # the batches are ready now, train on them
                    out_depth_potentials = self.one_chunk_module(chunk_batch)

                    # if required, confront the goutputs of the network to 
                    if show:
                        show_learning(fig, chunk_batch, out_depth_potentials, label_batch)

                    loss = self.loss_module(out_depth_potentials, label_batch)
                    loss.backward(retain_graph=True)  # retain_graph can be False for k1 = k2 cases of the TBPTT

                    self.optimizer.step()

                    self.one_chunk_module.detach()
                    self.optimizer.zero_grad()

                    running_loss += loss.item() * chunk_batch.size(0)

                    chunk_sequence = []
                    label_sequence = []
                    chunk_batch = []
                    label_batch = []

            epoch_loss = running_loss / len(train_data_loader)

            functional.reset_net(self.one_chunk_module)

            _, start_pots = next(iter(train_data_loader))
            start_pots = start_pots.repeat((N, 1, 1)).to(self.device)
            self.one_chunk_module.predict_depth[-1].v = start_pots

            # for evaluation, processes the sequence chunk by chunk
            # Note: in eval mode, the multiscale network only outputs the full resolution map
            self.one_chunk_module.eval()
            with torch.no_grad():
                for chunk, label in tqdm(train_data_loader):

                    chunk = chunk[0].to(device=self.device,
                                        dtype=torch.float)
                    label = label.to(self.device)

                    out_depth_potentials = self.one_chunk_module(chunk)

                    if show:
                        show_learning(fig, chunk, out_depth_potentials, label)

                    self.one_chunk_module.detach()

                    running_MDE += MeanDepthError(out_depth_potentials, label, learn_on_log)

                epoch_MDE = running_MDE / len(train_data_loader)

            end_time = time.time()
            epoch_summary = "Epoch: {}, Loss: {}, Mean Depth Error: {}, Time: {}\n".format(epoch, epoch_loss,
                                                                                               epoch_MDE,
                                                                                               end_time - start_time)
            print(epoch_summary)
            logfile.write(epoch_summary)

            self.lr_scheduler.step()

            # save model if better results
            if epoch_MDE < min_MDE:
                print("Best performances so far: saving model...\n")
                torch.save(self.one_chunk_module.state_dict(), "./checkpoints/spikeflownet_snn.pth")
                min_MDE = epoch_MDE

        # close log file
        logfile.close()

