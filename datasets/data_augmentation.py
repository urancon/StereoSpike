###################################################################
# This script defines transformations for data augmentation
# They are geometric (horizontal flip) or temporal (time mirror)
# They are used with shuffled datasets like 'shuffled_MVSEC' class
###################################################################

import torch
from torchvision import transforms
import torchvision.transforms.functional as F


class ToTensor:
    def __call__(self, data):
        init_pots, warmup_chunks, train_chunks, groundtruth = data
        init_pots = torch.from_numpy(init_pots)
        warmup_chunks = torch.from_numpy(warmup_chunks)
        train_chunks = torch.from_numpy(train_chunks)
        groundtruth = torch.from_numpy(groundtruth)
        data = (init_pots, warmup_chunks, train_chunks, groundtruth)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        if torch.rand(1) < self.p:
            init_pots, warmup_chunks, train_chunks, groundtruth = data
            init_pots = F.hflip(init_pots)
            warmup_chunks = F.hflip(warmup_chunks)
            train_chunks = F.hflip(train_chunks)
            groundtruth = F.hflip(groundtruth)
            data = (init_pots, warmup_chunks, train_chunks, groundtruth)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomTimeMirror(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        if torch.rand(1) < self.p:
            pre_init_pots, pre_warmup_chunks, pre_train_chunks, pre_groundtruth = data

            init_pots = pre_groundtruth

            N_warmup = pre_warmup_chunks.size(0)

            pre_chunks = torch.cat((pre_warmup_chunks, pre_train_chunks), dim=0)
            # reverse the order of chunks, the order of frames within chunks, and the polarity of events within frames
            pre_chunks = torch.flip(pre_chunks, dims=[0, 1, 2])

            warmup_chunks = pre_chunks[:N_warmup]
            train_chunks = pre_chunks[N_warmup:]

            groundtruth = pre_init_pots

            data = (init_pots, warmup_chunks, train_chunks, groundtruth)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


data_augmentation = transforms.Compose([
    ToTensor(),
    RandomHorizontalFlip(p=0.5),
    RandomTimeMirror(p=0.5),
])
