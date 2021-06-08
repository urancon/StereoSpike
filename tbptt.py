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
from torch.utils.data.sampler import Sampler, BatchSampler

from spikingjelly.clock_driven import functional

from network.metrics import mask_dead_pixels, MeanDepthError, Total_Loss, GradientMatching_Loss, \
    MultiScale_GradientMatching_Loss


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


def make_vid_from_pngs(png_folder, res_tuple, fps, outfile):

    import re

    def atoi(text):
        return int(text) if text.isdigit() else text
        elsetext

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outfile, fourcc, fps, res_tuple)

    i = 0
    sorted_filenames = os.listdir(png_folder)
    sorted_filenames.sort(key=natural_keys)  # sort png filenames in numerical order
    for file in sorted_filenames:
        i += 1
        frame = cv2.imread(png_folder + file)
        out.write(frame)
        cv2.waitKey(int(1000 / fps))

    out.release()
    print("created video file " + outfile)
    print()


class TBPTTsampler(Sampler):
    """
    This sampler simplifies the preparation process of chunks.
    It depends on the N = k2//k1 parameter of TBPTT.

    Example indices returned:
        + if N=2: 011223344556...
        + if N=3: 012123234345456...
        + etc.

    Using a BatchSampler(sampler, batch_size=N) allows to return the following batches:
        + if N=2: 01 12 23 34 45 ...
        + if N=3: 012 123 234 345 ...
        + etc.

    This way, we can train with both a multibatch approach (batch dimension is filled with shifted versions of the same
    sequence) or a monobatch approach.
    Inthe following, '|' symbol denotes loss calculation, gradient backpropagation, weight update, and detaching

        + multibatch approach: calculate loss and update weights once every other N batches
            --> if N=2: 01 12 | 23 34 | 45 56 | 67 78 | ...
            --> if N=3: 012 123 234 | 345 456 567 | ...

        + monobatch approach: calculate loss and update weights every batch
            --> if N=2: 01 | 12 | 23 | 34 | 45 | ...
            --> if N=3 012 | 123 | 234 | 345 | 456 | ...

    """
    def __init__(self, dataset, N):
        last_indices = list(range(N-1))  # last (N-1) indices are reused. We initialize them here.
        indices = []

        for i in range(N-1, len(dataset)):
            last_indices.append(i)
            indices = indices + last_indices
            del last_indices[0]

        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


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

    def train(self, train_data_loader, test_data_loader, nfpdm=5, N=1, learn_on_log=False, show=False):

        print("\n################")
        print("#   TRAINING   #")
        print("################\n")

        logfile = open("./results/checkpoints/training_logs.txt", "w+")

        print("\nTraining with approximated TBPTT procedure: {} frames between labels, k1=1, k2=N*K1, N={}\n".format(
            nfpdm, N))
        logfile.write(
            "Training with approximated TBPTT procedure: {} frames between labels, k1=1, k2=N*K1, N={}\n\n".format(
                nfpdm, N))

        if show:
            plt.ion()
            fig = plt.figure()
            png_dir = "./results/visualization/pngs/"

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
            # TODO: maybe add some mathematical morphology dilation to make holes in the gt disappear (relevant in multiscale nets)
            _, _, start_pots = next(iter(train_data_loader))
            start_pots = start_pots.to(self.device)
            self.one_chunk_module.set_init_depths_potentials(start_pots)

            self.one_chunk_module.train()
            for i, (chunk_batch_left, chunk_batch_right, label_batch) in enumerate(tqdm(train_data_loader)):

                chunk_batch = chunk_batch_left.to(device=self.device, dtype=torch.float)
                label_batch = label_batch.to(self.device)
                
                out_depth_potentials = self.one_chunk_module(chunk_batch)  # output shape: (N, 1, 260, 346)

                if (i+1) % N:

                    if show:
                        title = 'EPOCH {} - {}'.format(epoch, 'TRAINING')
                        viz = show_learning(fig, chunk_batch, out_depth_potentials, label_batch, title)
                        viz = Image.fromarray(viz)
                        viz.save(png_dir + "epoch{}_chunk{}_0.png".format(epoch, i))

                    loss = self.loss_module(out_depth_potentials, label_batch)
                    running_loss += loss.item() * chunk_batch.size(0)
                    loss.backward()
                    self.optimizer.step()
                    self.one_chunk_module.detach()
                    self.optimizer.zero_grad()

            epoch_loss = running_loss / len(train_data_loader)

            functional.reset_net(self.one_chunk_module)

            _, _, start_pots = next(iter(test_data_loader))
            start_pots = start_pots.to(self.device)
            self.one_chunk_module.set_init_depths_potentials(start_pots)

            # for evaluation, processes the sequence chunk by chunk
            # Note: in eval mode, the multiscale network only outputs the full resolution map
            self.one_chunk_module.eval()
            with torch.no_grad():
                for i, (chunk_left, chunk_right, label) in enumerate(tqdm(test_data_loader)):

                    chunk = chunk_left.to(device=self.device, dtype=torch.float)
                    label = label.to(self.device)

                    out_depth_potentials = self.one_chunk_module(chunk)

                    if show:
                        title = 'EPOCH {} - {}'.format(epoch, 'TEST')
                        viz = show_learning(fig, chunk, out_depth_potentials, label, title)
                        viz = Image.fromarray(viz)
                        viz.save(png_dir + "epoch{}_chunk{}_1.png".format(epoch, len(train_data_loader)+i))

                    self.one_chunk_module.detach()

                    running_MDE += MeanDepthError(out_depth_potentials, label, learn_on_log)

                epoch_MDE = running_MDE / len(test_data_loader)

            end_time = time.time()
            epoch_summary = "Epoch: {}, Loss: {}, Mean Depth Error: {}, Time: {}\n".format(epoch, epoch_loss,
                                                                                           epoch_MDE,
                                                                                           end_time - start_time)
            print(epoch_summary)
            logfile.write(epoch_summary)

            self.lr_scheduler.step()

            # save model if better results
            if epoch_MDE < self.one_chunk_module.get_max_accuracy():
                print("Best performances so far: saving model...\n")
                torch.save(self.one_chunk_module.state_dict(), "./results/checkpoints/spikeflownet_snn.pth")
                self.one_chunk_module.update_max_accuracy(epoch_MDE)

            self.one_chunk_module.increment_epoch()

        # close log file
        logfile.close()


if __name__ == "__main__":
    from datasets.MVSEC import MVSEC, MVSECmirror
    from datasets.DENSE import DENSE
    from network.models import SpikeFlowNetLike, SpikeFlowNetLike_cext, SpikeFlowNetLike_multiscale_v2, \
        Attention_SpikeFlowNetLike, Attention_SpikeFlowNetLike_cat, ConvLSTM_SpikeFlowNetLike, \
        FusionFlowNetLike, FusionFlowNetLike_cat, hybrid_FusionFlowNetLike_cat

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # make_vid_from_pngs("./results/checkpoints/visualization/pngs/", (2885, 864), 20, './results/checkpoints/visualization/videos/learning_viz.mp4')

    nfpdm = 1  # (!) don't choose it too big because of memory limitations (!)
    N = 2
    take_log = False
    learning_rate = 0.1
    weight_decay = 0.01
    n_epochs = 10

    dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                    num_frames_per_depth_map=nfpdm,
                    mirror_time=False,
                    take_log=take_log,
                    show_sequence=False)

    train_indices = list(range(0, 250))
    test_indices = list(range(0, 250))

    training_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    train_sampler = TBPTTsampler(training_set, N)
    train_data_loader = torch.utils.data.DataLoader(dataset=training_set,
                                                    batch_sampler=BatchSampler(train_sampler, batch_size=N, drop_last=False),
                                                    pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   pin_memory=True)

    net = SpikeFlowNetLike(tau=5.,
                           v_threshold=1.0,
                           v_reset=0.0,
                           v_infinite_thresh=float('inf'),
                           final_activation=torch.abs, #torch.nn.Identity(), #torch.nn.ReLU(),
                           use_plif=True
                           ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 15], gamma=0.1)

    loss_module = Total_Loss

    runner = ApproximatedTBPTT(net, loss_module, optimizer, scheduler, n_epochs, device)
    runner.train(train_data_loader, test_data_loader, nfpdm=nfpdm, N=N, learn_on_log=take_log, show=True)

    print("done")
