import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate

from datasets.MVSEC import load_MVSEC
from datasets.data_augmentation import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomTimeMirror, \
    RandomEventDrop

from network.SNN_models import StereoSpike
from network.ANN_models import SteroSpike_equivalentANN

from viz import show_learning

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


######################
# GENERAL PARAMETERS #
######################

nfpdm = 1  # (!) don't choose it too big because of memory limitations (!)
N_warmup = 1
N_inference = 1


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
# net = SteroSpike_equivalentANN(activation_function=nn.Sigmoid()).to(device)

net.load_state_dict(torch.load('./results/checkpoints/stereospike.pth'))

##############
# EVALUATION #
##############

logfile = open("./results/checkpoints/firing_rates_on_test.txt", "w+")

firing_rates_dict = {
    'out_bottom': 0.,
    'out_conv1': 0.,
    'out_conv2': 0.,
    'out_conv3': 0.,
    'out_conv4': 0.,
    'out_rconv': 0.,
    'out_combined': 0.,
    'out_deconv4': 0.,
    'out_add4': 0.,
    'out_deconv3': 0.,
    'out_add3': 0.,
    'out_deconv2': 0.,
    'out_add2': 0.,
    'out_deconv1': 0.,
    'out_add1': 0.,
}

net.eval()
with torch.no_grad():
    for init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label in tqdm(test_data_loader):

        # Pass tensors on the GPU / CPU
        init_pots = init_pots.to(device)
        warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
        warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
        test_chunks_left = test_chunks_right.to(device, dtype=torch.float)
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

        # calculate firing rates and update dictionary
        out_dict = net.calculate_firing_rates(test_chunks)  # out_dict = net.calculate_firing_rates(test_chunks_left)

        for key in out_dict:
            firing_rates_dict[key] += out_dict[key]

        net.detach()

# average firing rates across all inferences
for key in firing_rates_dict:
    firing_rates_dict[key] /= len(test_data_loader)

# output mean firing rates
print(firing_rates_dict)
logfile.write(firing_rates_dict)
logfile.close()
