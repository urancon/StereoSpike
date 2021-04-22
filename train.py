import random
import numpy as np
import torch

from mvsec import MVSEC
from tbptt import ApproximatedTBPTT
from network.models import SpikeFlowNetLike, SpikeFlowNetLike_cext, SpikeFlowNetLike_multiscale
from network.metrics import mask_dead_pixels, MeanDepthError, Total_Loss, GradientMatching_Loss, \
    MultiScale_GradientMatching_Loss


def set_random_seed(seed):
    # Python
    random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if int(torch.__version__.split('.')[1]) < 8:
        torch.set_deterministic(True)  # for pytorch < 1.8
    else:
        torch.use_deterministic_algorithms(True)

    # NumPy
    np.random.seed(seed)


set_random_seed(2021)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

nfpdm = 1  # (!) don't choose it too big because of memory limitations (!)
N = 1
learning_rate = 0.001
n_epochs = 15

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

net = SpikeFlowNetLike_cext(tau=10.,
                            v_threshold=1.0,
                            v_reset=0.0,
                            v_infinite_thresh=float('inf'),
                            final_activation=torch.abs,
                            use_plif=True
                            ).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 15], gamma=0.1)

loss_module = Total_Loss
# loss_module = MeanDepthError
# loss_module = F.mse_loss
# loss_module = GradientMatching_Loss
# loss_module = MultiScale_GradientMatching_Loss

runner = ApproximatedTBPTT(net, loss_module, optimizer, scheduler, n_epochs, device)
runner.train(train_data_loader, nfpdm=nfpdm, N=N, show=True)

print("done")
