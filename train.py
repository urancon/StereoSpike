from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import functional

from mvsec import MVSEC
from network.models import SpikeFlowNetLike
from network.metrics import mask_dead_pixels, MeanDepthError, Total_Loss

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


T = 10  # (!) don't choose T too big because of memory limitations (!)
batch_size = 1
learning_rate = 0.0001
n_epochs = 5

dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                num_frames_per_depth_map=5,
                normalization='max',
                take_log=True,
                show_sequence=True)
train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False,
                                                pin_memory=True)

net = SpikeFlowNetLike(tau=10, T=T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf')).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

currentFrame = 1
for epoch in range(n_epochs):
    print("epoch ", epoch)
    running_loss = 0.0

    functional.reset_net(net)

    # I think we have a case of TBPTT with k1 = k2 = 1 * num_frames_per_depth_map = 5 timesteps.
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

        # TODO: use custom loss (later)
        loss = F.mse_loss(out_depth_potentials, label)
        #loss = Total_Loss(out_depth_potentials, label)
        loss.backward()  # retain_graph can be False for k1 = k2 cases of the TBPTT
        optimizer.step()

        net.detach()
        optimizer.zero_grad()

        running_loss += loss.item() * chunk.size(0)

    epoch_loss = running_loss / (len(train_data_loader)/n_epochs)
    print("Loss:", epoch_loss)


# save model
torch.save(net.state_dict(), "./checkpoints/spikeflownet_snn.pth")

