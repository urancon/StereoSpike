import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as F

from spikingjelly.clock_driven import functional

from network.metrics import MeanDepthError, Total_Loss, mask_dead_pixels

from datasets.MVSEC import MVSEC, shuffled_MVSEC, binocular_shuffled_MVSEC
from datasets.data_augmentation import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomTimeMirror, RandomEventDrop
from network.ANN_models import Analog_ConvLSTM_SpikeFlowNetLike, \
    concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    biased_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    attention_biased_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    binocular_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike
from viz import show_learning

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


###########################
# VISUALIZATION FUNCTIONS #
###########################

plt.ion()
fig = plt.figure()


######################
# GENERAL PARAMETERS #
######################

nfpdm = 2  # (!) don't choose it too big because of memory limitations (!)
batchsize = 1
take_log = False
learning_rate = 0.001
weight_decay = 0.0
n_epochs = 10
show = True


########
# DATA #
########

# random transformations for data augmentation
tsfm = transforms.Compose([
    ToTensor(),
    #RandomHorizontalFlip(p=0.5),
    #RandomVerticalFlip(p=0.5),
    #RandomTimeMirror(p=0.5),
    RandomEventDrop(p=1, min_drop_rate=0.1, max_drop_rate=0.4)
])

train_shuffled_dataset = shuffled_MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                                        scenario='indoor_flying', case='2',
                                        num_frames_per_depth_map=nfpdm, warmup_chunks=1, train_chunks=1,
                                        transform=tsfm, normalize=True, mirror_time=False, take_log=take_log,
                                        show_sequence=False)
train_data_loader = torch.utils.data.DataLoader(dataset=train_shuffled_dataset,
                                                batch_size=batchsize,
                                                shuffle=True,
                                                drop_last=True,
                                                pin_memory=True)

test_shuffled_dataset = shuffled_MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                                       scenario='indoor_flying', case='1',
                                       num_frames_per_depth_map=nfpdm, warmup_chunks=1, train_chunks=1,
                                       normalize=True, mirror_time=False, take_log=take_log,
                                       show_sequence=False)
test_data_loader = torch.utils.data.DataLoader(dataset=test_shuffled_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=False,
                                               pin_memory=True)


###########
# NETWORK #
###########

net = multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike().to(device)


################
# OPTIMIZATION #
################

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
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
    for init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, label in tqdm(train_data_loader):

        # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_chunks*nfpdm, 2, 260, 346)
        warmup_chunks = warmup_chunks_left.view(batchsize, warmup_chunks_left.shape[1] * warmup_chunks_left.shape[2], 2, 260, 346).to(device, dtype=torch.float)
        train_chunks = train_chunks_left.view(batchsize, train_chunks_left.shape[1] * train_chunks_left.shape[2], 2, 260, 346).to(device, dtype=torch.float)
        init_pots = init_pots.to(device)
        label = label.to(device)

        # initialize output potentials
        functional.reset_net(net)
        net.reset_convLSTM_states()

        # let intermediate neurons "warm up" and reach a steady state before "real" training
        with torch.no_grad():
            net(warmup_chunks)

        # forward pass a long sequence of chunks
        pred = net(train_chunks)

        # confront prediction and groundtruth
        if show:
            show_learning(fig, train_chunks, pred[0], label, 'train')

        # calculate loss and update weights with BPTT
        loss = loss_module(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # save metrics
        running_train_loss += loss.item() * train_chunks.size(0)
        running_train_MDE += MeanDepthError(pred[0], label)

    # process saved metrics
    epoch_train_loss = running_train_loss / len(train_data_loader)
    epoch_train_MDE = running_train_MDE / len(train_data_loader)
    epoch_train_time = start_time - time.time()
    train_epoch_summary = "Epoch: {}, Training Loss: {}, Training Mean Depth Error: {}, Time: {}\n".format(epoch,
                                                                                                           epoch_train_loss,
                                                                                                           epoch_train_MDE,
                                                                                                           epoch_train_time)

    net.eval()
    with torch.no_grad():
        start_time = time.time()
        for init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label in tqdm(test_data_loader):

            warmup_chunks = warmup_chunks_left.view(1, warmup_chunks_left.shape[1] * warmup_chunks_left.shape[2], 2, 260, 346).to(device, dtype=torch.float)
            test_chunks = test_chunks_left.view(1, test_chunks_left.shape[1] * test_chunks_left.shape[2], 2, 260, 346).to(device, dtype=torch.float)
            init_pots = init_pots.to(device)
            label = label.to(device)

            functional.reset_net(net)
            net.reset_convLSTM_states()

            net(warmup_chunks)

            pred = net(test_chunks)[0]  # only take the full scale prediction in evaluation

            if show:
                show_learning(fig, test_chunks, pred, label, 'eval')

            loss = loss_module(pred, label)

            net.detach()

            running_test_loss += loss.item() / test_chunks.size(0)
            running_test_MDE += MeanDepthError(pred, label)

    epoch_test_loss = running_test_loss / len(test_data_loader)
    epoch_test_MDE = running_test_MDE / len(test_data_loader)
    epoch_test_time = time.time() - start_time
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

print("training finished !")
print("saving final model...")
torch.save(net.state_dict(), "./results/checkpoints/spikeflownet_snn_final.pth")
