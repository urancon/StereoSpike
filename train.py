from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import functional
from mvsec import MVSEC
from network import MonoSpikeFlowNetLike
from network.models import MonoSpikeFlowNetLike_v2

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


T = 10  # (!) don't choose T too big because of memory limitations (!)
batch_size = 1
learning_rate = 0.01
n_epochs = 5


dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/', num_frames_per_depth_map=5, normalization='max')
train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False,
                                                pin_memory=True)

net = MonoSpikeFlowNetLike_v2(tau=10, T=T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=1000).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
torch.autograd.set_detect_anomaly(True)

currentFrame = 1
for epoch in range(n_epochs):
    print("epoch ", epoch)
    running_loss = 0.0

    net.train()  # TODO: First train with TBPTT with k1 = k2 = 5 timesteps. Now it is some sort of k1 = k2 = 1
    for frame, label in tqdm(train_data_loader):

        optimizer.zero_grad()

        frame = frame[0].to(device=device, dtype=torch.float)  # only take left DVS data for monocular depth estimation
        label = label.to(device)

        out_depth_potentials = net(frame)

        loss = F.mse_loss(out_depth_potentials.squeeze(), label.squeeze())  # TODO: use custom loss
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item() * frame.size(0)

        functional.reset_net(net)

        """ TBPTT with k1 = k2 = 5 timesteps
        if currentFrame % 5 == 0:
            loss = torch.mean((out_depth_potentials.squeeze()-label.squeeze())**2)
            # loss = F.mse_loss(out_depth_potentials.squeeze(), label.squeeze())  # TODO: use custom loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        """

        currentFrame += 1

    epoch_loss = running_loss / len(train_data_loader)
    print("Loss:", epoch_loss)
