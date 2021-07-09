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
        init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth = data

        init_pots = torch.from_numpy(init_pots)
        warmup_chunks_left = torch.from_numpy(warmup_chunks_left)
        train_chunks_left = torch.from_numpy(train_chunks_left)
        groundtruth = torch.from_numpy(groundtruth)

        if type(warmup_chunks_right) != int and type(train_chunks_right) != int:
            warmup_chunks_right = torch.from_numpy(warmup_chunks_right)
            train_chunks_right = torch.from_numpy(train_chunks_right)

        data = (init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        if torch.rand(1) < self.p:
            init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth = data

            init_pots = F.hflip(init_pots)
            warmup_chunks_left = F.hflip(warmup_chunks_left)
            train_chunks_left = F.hflip(train_chunks_left)
            groundtruth = F.hflip(groundtruth)

            if warmup_chunks_right.shape == warmup_chunks_left.shape and train_chunks_right.shape == train_chunks_left.shape:
                warmup_chunks_right = F.hflip(warmup_chunks_right)
                train_chunks_right = F.hflip(train_chunks_right)

            data = (init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        if torch.rand(1) < self.p:
            init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth = data

            init_pots = F.vflip(init_pots)
            warmup_chunks_left = F.vflip(warmup_chunks_left)
            train_chunks_left = F.vflip(train_chunks_left)
            groundtruth = F.vflip(groundtruth)

            if warmup_chunks_right.shape == warmup_chunks_left.shape and train_chunks_right.shape == train_chunks_left.shape:
                warmup_chunks_right = F.vflip(warmup_chunks_right)
                train_chunks_right = F.vflip(train_chunks_right)

            data = (init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomTimeMirror(torch.nn.Module):
    """
    Data augmentation by reversing the chronological order of the data.
    The order of chunks and spike frames is flipped, as well as the polarity of events.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        if torch.rand(1) < self.p:
            pre_init_pots, pre_warmup_chunks_left, pre_warmup_chunks_right, pre_train_chunks_left, pre_train_chunks_right, pre_groundtruth = data

            init_pots = pre_groundtruth

            N_warmup = pre_warmup_chunks_left.size(0)

            pre_chunks_left = torch.cat((pre_warmup_chunks_left, pre_train_chunks_left), dim=0)  # concatenate warmup and train chunks in chronological order
            pre_chunks = torch.flip(pre_chunks_left, dims=[0, 1, 2])  # reverse the order of chunks, the order of frames within chunks, and the polarity of events within frames
            warmup_chunks_left = pre_chunks[:N_warmup]
            train_chunks_left = pre_chunks[N_warmup:]

            if pre_warmup_chunks_right.shape == pre_warmup_chunks_left.shape and pre_train_chunks_right.shape == pre_train_chunks_left.shape:
                pre_chunks_right = torch.cat((pre_warmup_chunks_right, pre_train_chunks_right), dim=0)
                pre_chunks_right = torch.flip(pre_chunks_right, dims=[0, 1, 2])
                warmup_chunks_right = pre_chunks_right[:N_warmup]
                train_chunks_right = pre_chunks_right[N_warmup:]
            else:
                warmup_chunks_right = 0
                train_chunks_right = 0

            groundtruth = pre_init_pots

            data = (init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomEventDrop(torch.nn.Module):
    """"
    Inpired by the paper 'EventDrop: data augmentation for event-based learning':
    https://arxiv.org/abs/2106.05836v1
    """

    def __init__(self, p=0.5, min_drop_rate=0., max_drop_rate=0.4):
        super().__init__()
        self.p = p
        self.min_drop_rate = min_drop_rate
        self.max_drop_rate = max_drop_rate

    def forward(self, data):
        if torch.rand(1) < self.p:

            # probability of an input event to be dropped: random variable uniformly distributed on [0, 0.6] by default
            q = (self.min_drop_rate - self.max_drop_rate) * torch.rand(1) + self.max_drop_rate

            init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth = data

            mask_warmup_left = torch.rand_like(warmup_chunks_left)
            mask_train_left = torch.rand_like(train_chunks_left)
            warmup_chunks_left = warmup_chunks_left * (mask_warmup_left > q)
            train_chunks_left = train_chunks_left * (mask_train_left > q)

            if warmup_chunks_right.shape == warmup_chunks_left.shape and train_chunks_right.shape == train_chunks_left.shape:
                mask_warmup_right = torch.rand_like(warmup_chunks_right)
                mask_train_right = torch.rand_like(train_chunks_right)
                warmup_chunks_right = warmup_chunks_left * (mask_warmup_right > q)
                train_chunks_right = train_chunks_left * (mask_train_right > q)

            data = (init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, groundtruth)

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, min_drop_rate={}, max_drop_rate={})'.format(self.p, self.min_drop_rate, self.max_drop_rate)


data_augmentation = transforms.Compose([
    ToTensor(),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.1),
    RandomTimeMirror(p=0.5),
    RandomEventDrop(p=0.6)
])

