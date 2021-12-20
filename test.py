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
from spikingjelly.clock_driven import surrogate

from network.metrics import MeanDepthError, log_to_lin_depths, disparity_to_depth
from network.loss import Total_Loss

from datasets.MVSEC import load_MVSEC
from datasets.data_augmentation import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomTimeMirror, \
    RandomEventDrop

from network.SNN_models import StereoSpike, fromZero_feedforward_multiscale_tempo_Matt_SpikeFlowNetLike
from network.ANN_models import StereoSpike_equivalentANN

from network.metrics import MeanDepthError, log_to_lin_depths, disparity_to_depth
from network.loss import Total_Loss

from viz import show_learning

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


######################
# GENERAL PARAMETERS #
######################

nfpdm = 1  # (!) don't choose it too big because of memory limitations (!)
N_warmup = 1
N_inference = 1
learned_metric = 'LIN'
show = True


###########################
# VISUALIZATION FUNCTIONS #
###########################

plt.ion()
fig = plt.figure()


########
# DATA #
########

# random transformations for data augmentation
tsfm = transforms.Compose([
    ToTensor(),
    # RandomHorizontalFlip(p=0.5),
    # RandomVerticalFlip(p=0.5),
    # RandomTimeMirror(p=0.5),
    # RandomEventDrop(p=0.5, min_drop_rate=0., max_drop_rate=0.4)
])

test_set = load_MVSEC('./datasets/MVSEC/data/', scenario='indoor_flying', split='1',
                      num_frames_per_depth_map=1, warmup_chunks=1, train_chunks=1,
                      transform=tsfm, normalize=False, learn_on='LIN',
                      load_test_only=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=False,
                                               pin_memory=True)


###########
# NETWORK #
###########

net = StereoSpike(surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.).to(device)
net = StereoSpike_equivalentANN(activation_function=nn.Sigmoid()).to(device)
# net = fromZero_feedforward_multiscale_tempo_Matt_SpikeFlowNetLike(tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True, multiply_factor=10.).to(device)

net.load_state_dict(torch.load('./results/checkpoints/stereospike.pth'))


################
# OPTIMIZATION #
################

loss_module = Total_Loss(alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=False)


##############
# EVALUATION #
##############

logfile = open("./results/checkpoints/test_results.txt", "w+")

net.eval()
with torch.no_grad():

    start_time = time.time()
    running_test_loss = 0
    running_test_MDE = 0
    i = 0

    for init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label in tqdm(
            test_data_loader):

        # Pass tensors on the GPU / CPU
        init_pots = init_pots.to(device)
        warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
        warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
        test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
        test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
        label = label.to(device)

        # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_chunks*nfpdm, 2, 260, 346)
        warmup_chunks_left = warmup_chunks_left.view(1, N_warmup * nfpdm, 2, 260, 346)
        warmup_chunks_right = warmup_chunks_right.view(1, N_warmup * nfpdm, 2, 260, 346)
        test_chunks_left = test_chunks_left.view(1, N_inference * nfpdm, 2, 260, 346)
        test_chunks_right = test_chunks_right.view(1, N_inference * nfpdm, 2, 260, 346)

        # concatenate subsequent frames channel-wise: (B, num_frames, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
        # where num_frames = num_chunks * nfpdm
        # Used to give some sort of temporal information to the stateless model via the input
        # /!\ number of filters in the first convolution should be changes accordingly /!\
        warmup_chunks_left = warmup_chunks_left.view(1, 1, N_warmup * nfpdm * 2, 260, 346)
        warmup_chunks_right = warmup_chunks_right.view(1, 1, N_warmup * nfpdm * 2, 260, 346)
        test_chunks_left = test_chunks_left.view(1, 1, N_inference * nfpdm * 2, 260, 346)
        test_chunks_right = test_chunks_right.view(1, 1, N_inference * nfpdm * 2, 260, 346)

        # concatenate left and right inputs channel-wise
        # (for binocular model, ignore for monocular model)
        warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
        test_chunks = torch.cat((test_chunks_left, test_chunks_right), dim=2)

        # initialize all neuron potentials
        functional.reset_net(net)

        # let intermediate neurons "warm up" and reach a steady state before "real" training
        # Useful for stateful models, but not used, as StereoSpike is stateless
        '''
        with torch.no_grad():
            net(warmup_chunks_left, warmup_chunks_right)
        '''

        # forward pass a long sequence of chunks
        pred, spks = net(test_chunks)  # for monocular models: pred, spks = net(test_chunks_left)

        # confront prediction and groundtruth
        if show:
            show_learning(fig, test_chunks_left, pred[0], label, 'train')

        # calculate loss
        loss = loss_module(pred, label, spks)

        # only convert prediction back to linear (metric) depth, for Mean Depth Error (MDE) calculation
        # only consider full scale prediction for evaluation
        if learned_metric == 'LIN':
            lin_pred = pred[0]
        elif learned_metric == 'LOG':
            lin_pred = log_to_lin_depths(pred[0])
        elif learned_metric == 'DISP':
            lin_pred = disparity_to_depth(pred[0])

        # calculate MDE
        MDE = MeanDepthError(lin_pred, label)

        # save metrics
        running_test_loss += loss.item() / test_chunks_left.size(0)
        running_test_MDE += MDE

epoch_test_loss = running_test_loss / len(test_data_loader)
epoch_test_MDE = running_test_MDE / len(test_data_loader)
epoch_test_time = time.time() - start_time

# show results
test_epoch_summary = "Test Loss: {}, Test Mean Depth Error (m): {}, Time: {}\n".format(
    epoch_test_loss,
    epoch_test_MDE,
    epoch_test_time)
print(test_epoch_summary)
logfile.write(test_epoch_summary)
logfile.close()
