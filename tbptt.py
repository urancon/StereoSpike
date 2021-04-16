import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from spikingjelly.clock_driven import functional

from network.metrics import mask_dead_pixels, MeanDepthError, Total_Loss


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


    TODO: use custom loss
    TODO: add a learning rate scheduler
    TODO: add a tensorboard logger
    """

    def __init__(self, one_chunk_module, loss_module, optimizer, n_epochs, device):
        self.one_chunk_module = one_chunk_module
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device

    def train(self, train_data_loader, nfpdm=5, N=1):

        logfile = open("./checkpoints/training_logs.txt", "w+")

        for epoch in range(self.n_epochs):
            start_time = time.time()
            running_loss = 0.0
            running_MDE = 0.0

            # reset the potential of all neurons before each sequence (i.e., epoch)
            functional.reset_net(self.one_chunk_module)

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

                    # mask all invalid pixels, corresponding to pixels of the groundtruth with values of 0 or 255
                    # these pixels become 0 in both prediction and groundtruth (so that residuals become 0 too),
                    # the others remain as they are
                    out_depth_potentials, label_batch = mask_dead_pixels(out_depth_potentials, label_batch)

                    loss = self.loss_module(out_depth_potentials, label)
                    #loss = F.mse_loss(out_depth_potentials, label)
                    #loss = Total_Loss(out_depth_potentials, label, alpha=0.)
                    #loss = MeanDepthError(out_depth_potentials, label_batch)

                    loss.backward(retain_graph=True)  # retain_graph can be False for k1 = k2 cases of the TBPTT
                    optimizer.step()

                    self.one_chunk_module.detach()
                    optimizer.zero_grad()

                    running_loss += loss.item() * chunk_batch.size(0)

                    chunk_sequence = []
                    label_sequence = []
                    chunk_batch = []
                    label_batch = []

            epoch_loss = running_loss / (len(train_data_loader) / self.n_epochs)

            functional.reset_net(self.one_chunk_module)

            chunk_sequence = []
            label_sequence = []
            chunk_batch = []
            label_batch = []

            self.one_chunk_module.eval()
            with torch.no_grad():
                for i, (chunk, label) in enumerate(tqdm(train_data_loader)):

                    chunk = chunk[0].to(device=self.device,
                                        dtype=torch.float)  # only take left DVS data for monocular depth estimation
                    label = label.to(self.device)

                    chunk_sequence.append(chunk)
                    label_sequence.append(label)

                    if (i + 1) % (2 * N - 1) == 0:

                        # the batch dimension is equal to N, so a batch is composed of N shifted chunk sequences
                        # e.g. N=3, chunk_sequence=[0, 1, 2, 3, 4] (len: 5=2N-1)
                        # --> chunk_batch=[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
                        # --> label_batch=[2, 3, 4]
                        for j in range(N):
                            chunk_batch.append(torch.cat(chunk_sequence[j:N + j], dim=1))  # concatenate chunks along frame (time) dimension  [1, nfpdm, 2, H, W] --> [1, N*nfpdm, 2, H, W]
                            label_batch.append(label_sequence[N-1+j])  # get the label corresponding to the chunk sequence --> [1, H, W]
                        chunk_batch = torch.cat(chunk_batch, dim=0)  # concatenate along batch dimension --> [N, N*nfpdm, 2, H, W]
                        label_batch = torch.cat(label_batch, dim=0)  # concatenate along batch dimension --> [N, H, W]

                        # the batches are ready now, train on them
                        out_depth_potentials = self.one_chunk_module(chunk_batch)

                        # mask all invalid pixels, corresponding to pixels of the groundtruth with values of 0 or 255
                        # these pixels become 0 in both prediction and groundtruth (so that residuals become 0 too),
                        # the others remain as they are
                        out_depth_potentials, label_batch = mask_dead_pixels(out_depth_potentials, label_batch)

                        self.one_chunk_module.detach()

                        running_MDE += MeanDepthError(out_depth_potentials, label)

                        chunk_sequence = []
                        label_sequence = []
                        chunk_batch = []
                        label_batch = []

                    epoch_MDE = running_MDE / len(train_data_loader)

            end_time = time.time()
            epoch_summary = "Epoch: {}, Loss: {}, Mean Depth Error: {}, Time: {}\n".format(epoch, epoch_loss,
                                                                                               epoch_MDE,
                                                                                               end_time - start_time)
            print(epoch_summary)
            logfile.write(epoch_summary)

        # save model
        torch.save(self.one_chunk_module.state_dict(), "./checkpoints/spikeflownet_snn.pth")

        # close log file
        logfile.close()


if __name__ == "__main__":
    from mvsec import MVSEC
    from network.models import SpikeFlowNetLike

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    nfpdm = 5  # (!) don't choose it too big because of memory limitations (!)
    N = 1
    learning_rate = 0.0001
    n_epochs = 10

    dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                    num_frames_per_depth_map=nfpdm,
                    normalization=None,
                    take_log=False,
                    show_sequence=False)
    train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    drop_last=False,
                                                    pin_memory=True)

    net = SpikeFlowNetLike(tau=10.,
                           v_threshold=1.0,
                           v_reset=0.0,
                           v_infinite_thresh=float('inf'),
                           final_activation=torch.abs,
                           use_plif=True
                           ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    loss_module = MeanDepthError
    #loss_module = F.mse_loss

    runner = ApproximatedTBPTT(net, loss_module, optimizer, n_epochs, device)
    runner.train(train_data_loader, nfpdm=nfpdm, N=N)

    print("done")


