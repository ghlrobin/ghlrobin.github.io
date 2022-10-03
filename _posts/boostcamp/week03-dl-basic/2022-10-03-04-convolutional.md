---
title: "[boostcamp AI Tech][DL Basic] Lecture 4: Convolutional Neural Networks"
date: 2022-10-03 12:00:00 + 0900
categories: [boostcamp AI Tech, Week 3 - DL Basic]
tags: [boostcamp, dl basic, level 1, week 3] # TAG names should always be lowercase
math: true
---

- [Convolution Neural Networks](#convolution-neural-networks)
- [Stride](#stride)
- [Padding](#padding)
- [1x1 Convolution](#1x1-convolution)

# Convolution Neural Networks

- Depending on the filter, 2D convolution will have different effects (Blur, Emoss, Outline, etc)

![](/assets/img/boostcamp/2022-10-03-21-26-27.png)

- If you want the output to have n channels, you also need to have n filters
- To calcualte the number of parameters in the output layer after a convolution layer (given padding(1) and stride(1), 3 x 3 kernel):
  - $\text{\# of parameters} = \text{kernel size} \times \text{\# of input channel} \times \text{\# of output channel}$

![](/assets/img/boostcamp/2022-10-03-21-37-21.png)

- CNN consists of convolution layer, pooling layer, and fully connected layer
- Convolution and pooling layers: feature extraction
  - pooling layers are used to reduce the dimensions of the feature maps. Thus it reduces the number of parameters to learn and the amount of computation performed in the network.
- Fully connectedlayer: decision making (e.g. classification)
  - However we tend to not use FC layers anymore as it increases the number of parameters too much so learning becomes difficult and the generalization performance decreases

# Stride

- Stride measures by how many pixels the filter will move
- So it measures how densely you will use the filter on the image

# Padding

- Padding refers to the amount of pixels added to an image to its fringe when it is being processed by the kerenl of a CNN
- Zero padding and and stride = 1 results in the output having the same dimension as the input

# 1x1 Convolution

![](/assets/img/boostcamp/2022-10-03-21-59-30.png)

- Pixel by pixel operation
- It's used for dimension reduction (channel-wise)
- It's used to reduce the number of parameters while increasing the depth e.g. bottleneck architecture.

-------------------------------


