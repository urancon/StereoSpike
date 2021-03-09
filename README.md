# Depth-SNN
*Depth estimation with Spiking Neural Networks (SNN) on Dynamic Vision Sensor (DVS) data.*

In progress - March-August 2021.


## Overview

**Spiking neural networks** (SNN) are getting more and more interest in the AI community, as they use temporal information, to the
contrary of their "regular" ANN cousins. They could even outperform the latter in the coming years and become a new paradigm for complex computation.
Indeed, it is already broadly recognized that their hardware implementation on neuromorphic chips is far less energy consuming
than classical GPUs. In this context, a growing number of innovators and constructors propose compatible hardware, especially
in the field of vision. **Dynamic Vision Sensors** capture the dynamics of a scene and moving objects with inspiration from 
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



