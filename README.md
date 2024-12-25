# AlexNet - Image Classification Model

This repository contains an implementation of the AlexNet model for image classification tasks. The project utilizes the CIFAR-10 dataset, consisting of 10 classes of small images, resized to fit the input size required by AlexNet.

---
## Overview
AlexNet is a convolutional neural network (CNN) originally designed for large-scale image classification tasks. It uses multiple convolutional layers, pooling layers, and fully connected layers to extract and classify image features. In this repository:
- AlexNet has been adapted to handle the CIFAR-10 dataset.
- Includes data preprocessing, training, and evaluation pipelines.
- Features learning rate scheduling and early stopping for optimal training.

---

## Dataset
This project uses the **CIFAR-10 dataset**, which contains:
- 15,000 training images and 3,000 test images (subset of CIFAR-10).
- 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

## Results
Using 15,000 training samples and 3,000 test samples, the model achieved the following results:

- **Precision**: **70.26%**
- **Recall**: **69.95%**
- **F1-Score**: **69.93%**
---
## About AlexNet
AlexNet is a convolutional neural network (CNN) originally introduced in 2012 by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. 

For more details, you can read the official paper: ["ImageNet Classification with Deep Convolutional Neural Networks"](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).


