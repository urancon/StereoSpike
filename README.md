# StereoSpike: Depth Learning with a Spiking Neural Network
*Depth estimation with Spiking Neural Networks (SNN) on Dynamic Vision Sensor (DVS) data in a binocular setup.*

Started March 2021 - under construction

last update Nov. 21st, 2021

![demo-gif](./sources/demo.gif)


## Arxiv Paper

This repository is associated with the Arxiv paper *[StereoSpike: Depth Learning with a Spiking Neural Network](https://arxiv.org/abs/2109.13751)*.
It essentially provides training and evaluation codes we used to develop our model. 


## Overview

**Spiking neural networks (SNN)** are getting more and more interest in the AI community, as they use temporal information, to the
contrary of their "regular" ANN cousins. They could even outperform the latter in the coming years and become a new paradigm for complex computation.
Indeed, it is already broadly recognized that their hardware implementation on neuromorphic chips is far less energy consuming
than classical GPUs. In this context, a growing number of innovators and constructors propose compatible hardware, especially
in the field of vision. **Dynamic Vision Sensors (DVS)** capture the dynamics of a scene and moving objects with inspiration from 
the retina. Instead of taking shots at regular intervals, it rather produces spikes in extremely high temporal resolution
(in the order of the microsecond), whenever a detected luminance change in a given location of the field exceeds a threshold.
They are now being investigated for countless applications and at different levels, such as object detection, embedded systems
(**edge AI**), or videosurveillance.

This project aims at ***developing a SNN that estimates the depth of each pixel in a scene, given
two separate spike trains obtained in a bio-inspired manner with DVSs.***

In addition to providing a new solution for a practical engineering task, this work could lead to modest but still appreciable 
insights on the functioning of the **Visual Nervous System**.


## Dataset and Framework

We are working on [MVSEC dataset](https://daniilidis-group.github.io/mvsec/), which consists in a series of synchronised
measurements on a moving rig. In particular, two **DAVIS m346b** cameras produce two streams a spike events, while a 
**Velodyne Puck LITE** Lidar captures depth maps corresponding to each of them.

[Spikingjelly](https://github.com/fangwei123456/spikingjelly) is a recent Pytorch-based framework for developing SNNs. 
It features surrogate gradient-learning rule, a friendly syntax for those already familiar with torch, and a very 
powerful CUDA acceleration (up to ~300 faster than naive Pytorch).


## Installation

First, make sure to create a new virtual python environment and activate it. With [conda](https://www.anaconda.com/), 
you can do for instance:
```shell
conda create -n stereoSNN python=3.6
conda activate stereoSNN
```

Then install dependencies to run experiments with ```pip3 install -r requirements.txt```.

Finally, manually download *indoor_flying* sequences from [MVSEC website](https://daniilidis-group.github.io/mvsec/) 
**under hdf5 format**. Extract and order them so that they follow this architecture:

```
datasets/
    ├── MVSEC/
    │      ├── data/
    │      │    ├── indoor_flying/
    │      │    │   ├── indoor_flying1_data.hdf5
    │      │    │   ├── indoor_flying1_gt.hdf5
    │      │    │   ├── ...
    │      │    │   └── indoor_flying_calib/
    │      │    │       ├── camchain-imucam-indoor_flying.yaml
    │      │    │       ├── indoor_flying_left_x_map.txt
    │      │    │       └── ...
    │      │    └── ...
    │      │
    │      ├── __init__.py
    │      ├── mvsec_dataset.py 
    │      └── utils.py
    └── ...  
```

You are now ready to use the code on this repo !


## Training and evaluation

There are 3 main scripts that you can use to replicate our results:
* ```train.py``` - train a model, logs and models are saved in *results/checkpoints/* folder
* ```test.py``` - evaluate a model on its test set, get its Mean Depth Error (MDE)
* ```calculate_firing rates.py``` - model performs inference on test set, densities of all intermediate tensors are averaged
across all inferences

We let the default hyper-parameters that we used during our experiments, but feel free to edit them and adapt them to your
needs !


## Pre-trained models

You will be able to find a few pre-trained models at the following address:

The following table summarizes their features and performances:

| Model name        | Type           | Modality    | Eval Mean Depth Error (cm) |
| ----------------- |:--------------:|:-----------:|:--------------------------:|
|  *to be added*                 | SNN            | Binocular   |                            |
|  *to be added*                 | SNN            | Monocular   |                            |
|  *to be added*                 | ANN            | Binocular   |                            |


## Weights and Biases logs

You can check out training curves on this [wandb report](https://wandb.ai/urancon/stereospike_temporal_study/reports/StereoSpike-yet-another-study--VmlldzoyMzUyODQ0?accessToken=w3neipg7h5i2lln6g5aliwsv08el4nf3zxxwpe4gmxx7rq5f032prsnz15zygvkl).


## Citation

We hope that you find our code and article useful for your work. If that is the case, please cite us !

```text
@misc{rançon2021stereospike,
      title={StereoSpike: Depth Learning with a Spiking Neural Network}, 
      author={Ulysse Rançon and Javier Cuadrado-Anibarro and Benoit R. Cottereau and Timothée Masquelier},
      year={2021},
      eprint={2109.13751},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
