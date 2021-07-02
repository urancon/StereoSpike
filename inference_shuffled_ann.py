import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as F

from spikingjelly.clock_driven import functional

from network.metrics import MeanDepthError, Total_Loss, mask_dead_pixels

from datasets.MVSEC import MVSEC, shuffled_MVSEC, binocular_shuffled_MVSEC
from datasets.data_augmentation import ToTensor, RandomHorizontalFlip, RandomTimeMirror, RandomEventDrop
from network.ANN_models import Analog_ConvLSTM_SpikeFlowNetLike, \
    concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    biased_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    attention_biased_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    binocular_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike, \
    biased_binocular_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike
from viz import show_learning

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


###########################
# VISUALIZATION FUNCTIONS #
###########################

plt.ion()
fig = plt.figure()
png_dir = "./results/visualization/pngs/"


######################
# GENERAL PARAMETERS #
######################

nfpdm = 2  # (!) don't choose it too big because of memory limitations (!)
batchsize = 1
take_log = False
learning_rate = 0.001
weight_decay = 0.0
n_epochs = 20
show = True


########
# DATA #
########

# random transformations for data augmentation
tsfm = []

test_shuffled_dataset = binocular_shuffled_MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                                       scenario='indoor_flying', case='1',
                                       num_frames_per_depth_map=nfpdm, warmup_chunks=5, train_chunks=5,
                                       mirror_time=False, take_log=take_log, show_sequence=False)
test_data_loader = torch.utils.data.DataLoader(dataset=test_shuffled_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=False,
                                               pin_memory=True)


###########
# NETWORK #
###########

net = binocular_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike().to(device)
net.load_state_dict(torch.load('./results/checkpoints/shuffled_ann_binocular/spikeflownet_snn.pth'))

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

    start_time = time.time()
    running_test_loss = 0
    running_test_MDE = 0

    i=0

    net.eval()
    with torch.no_grad():
        start_time = time.time()
        for init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label in tqdm(test_data_loader):

            warmup_chunks_left = warmup_chunks_left.view(1, warmup_chunks_left.shape[1] * warmup_chunks_left.shape[2], 2, 260, 346).to(device, dtype=torch.float)
            #warmup_chunks_right = warmup_chunks_right.view(1, warmup_chunks_right.shape[1] * warmup_chunks_right.shape[2], 2, 260, 346).to(device, dtype=torch.float)
            warmup_chunks_right = torch.zeros_like(warmup_chunks_left)
            test_chunks_left = test_chunks_left.view(1, test_chunks_left.shape[1] * test_chunks_left.shape[2], 2, 260, 346).to(device, dtype=torch.float)
            #test_chunks_right = test_chunks_right.view(1, test_chunks_right.shape[1] * test_chunks_right.shape[2], 2, 260, 346).to(device, dtype=torch.float)
            test_chunks_right = torch.zeros_like(test_chunks_left)
            init_pots = init_pots.to(device)
            label = label.to(device)

            functional.reset_net(net)
            net.reset_convLSTM_states()

            net(warmup_chunks_left, warmup_chunks_right)

            pred = net(test_chunks_left, test_chunks_right)[0]  # only take full scale prediction in evaluation

            if show:
                title = 'EPOCH {} - {}'.format(epoch, 'TEST')
                viz = show_learning(fig, test_chunks_right, pred, label, 'eval')
                viz = Image.fromarray(viz)
                viz.save(png_dir + "epoch{}_chunk{}_1.png".format(epoch, i))

            loss = loss_module(pred, label)

            net.detach()

            running_test_loss += loss.item() / test_chunks_left.size(0)
            running_test_MDE += MeanDepthError(pred, label)

            i += 1

    epoch_test_loss = running_test_loss / len(test_data_loader)
    epoch_test_MDE = running_test_MDE / len(test_data_loader)
    epoch_test_time = time.time() - start_time
    test_epoch_summary = "Epoch: {}, Test Loss: {}, Test Mean Depth Error: {}, Time: {}\n".format(epoch,
                                                                                                  epoch_test_loss,
                                                                                                  epoch_test_MDE,
                                                                                                  epoch_test_time)
    print(test_epoch_summary)
    logfile.write(test_epoch_summary)

    # save model if better results
    if epoch_test_MDE < net.get_max_accuracy():
        print("Best performances so far: saving model...\n")
        torch.save(net.state_dict(), "./results/checkpoints/spikeflownet_snn.pth")
        net.update_max_accuracy(epoch_test_MDE)

    net.increment_epoch()

    scheduler.step()
