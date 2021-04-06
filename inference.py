from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import functional
from spikingjelly import visualizing

from mvsec import MVSEC
from network import SpikeFlowNetLike, FusionFlowNetLike
from network import MeanDepthError, Total_Loss


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

outfile = './checkpoints/prediction_vs_groundtruth.mp4'
nfpdm = 5
LIDAR_FPS = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(outfile, fourcc, LIDAR_FPS, (2*346, 260), isColor=False)
predicted_frames_list = []
label_list = []

T = 10  # (!) don't choose T too big because of memory limitations (!)
batch_size = 1

dataset = MVSEC('/home/ulysse/Desktop/PFE CerCo/datasets/MVSEC/',
                num_frames_per_depth_map=nfpdm,
                normalization='max',
                take_log=True,
                show_sequence=False)
train_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False,
                                                pin_memory=True)
#dataset.show()

for chunk, label in train_data_loader:
    chunk = chunk[0].squeeze()
    for frame in chunk:
        f = frame[0]
        plt.imshow(f)
        plt.show()


net = SpikeFlowNetLike(tau=10, T=T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf')).to(device)
net.load_state_dict(torch.load("./checkpoints/spikeflownet_snn.pth"))
net.eval()

torch.no_grad()

# inference on all frames
for chunk, label in tqdm(train_data_loader):
    chunk = chunk[0].to(device=device, dtype=torch.float)  # only take left DVS data for monocular depth estimation
    label = ((label / label.max()) * 255).detach().cpu().numpy().astype('uint8')
    label_list.append(np.squeeze(label))
    prediction = net(chunk)
    predicted_frames_list.append(torch.squeeze(prediction).cpu().detach().numpy())
    net.detach()

# constitution of a comparison video file
for f, j in zip(predicted_frames_list, range(len(label_list))):
    f = ((f / f.max()) * 255).astype('uint8')  # normalization of pixel values
    f = np.concatenate((f, label_list[j]), axis=1)  # concatenate to show left and right images side by side
    #f = cv2.applyColorMap(f, cv2.COLORMAP_JET)
    cv2.imshow("prediction_vs_groundtruth", f)
    out.write(f)
    cv2.waitKey(int(1000 / LIDAR_FPS))

out.release()
print("saved video sequence to 'prediction_vs_groundtruth.mp4'...\n")

