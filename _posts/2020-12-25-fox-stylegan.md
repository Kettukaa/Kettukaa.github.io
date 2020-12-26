---
layout: post
title: Utilizing StyleGAN2 and StyleGAN2-ADA to Generate Photorealistic Foxes
categories: [foxes, ML]
---

Generating high fidelity images through the power of machine learning has become increasingly trivial and accessible to the average person. NVLab's [StyleGAN2]() (SG2) and [StyleGAN2-ADA]() (SG2-ADA) generative GAN models can be easily used to generate a wide range of images if provided a large enough well-created dataset. 

![](../images/foxes/SG2ADA/samples.png)

## Preparation
Data preparation mainly consisted of gathering data, automatically locating fox heads, and cropping the heads into squares. To locate and crop the heads automatically, I used a YoloV4 model (which I talk about [here](https://kettukaa.github.io/fox-detection/)) to find the heads, and a script to crop the images. 

## StyleGAN2 vs. StyleGAN2-ADA
One problem with SG2 that SG2-ADA attempts to solve is enabling training on smaller datasets. NVLab's solution to this problem is, what they call, "Adaptive Discriminator Augmentation" or ADA. ADA in addition to the new Augmentation Pipeline allows me to generate higher quality foxes than what was possible in SG2 given the same dataset. The reason I include both SG2 and SG2-ADA is because I had started this project before SG2-ADA existed. So I felt the need to document both models. More information on SG2 vs. SG2-ADA can be found in their respective papers:
 - https://arxiv.org/abs/1912.04958
 - https://arxiv.org/abs/2006.06676

## Training Methods
For both SG2 and SG2-ADA, I used an existing github repository to train the models.

### TPU and GPU
The primary difference between how I trained the two models was that SG2 had a repo for utilizing a [Cloud TPU](https://cloud.google.com/tpu) (Tensor Processing Unit), a very powerful computer made specifically with neural networks in mind. Utilizing a TPU allows for fast training and model inference, the only issue is acquiring a TPU. Luckily, [TensorFlow](https://www.tensorflow.org/), the company behind the popular neural network framework of the same name, has a service available called TensorFlow Research Cloud ([TFRC](https://www.tensorflow.org/tfrc)) which allows researchers to apply and use one of Google's TPU V3-8's for no cost, aside from general Google Cloud Platform ([GCP](https://cloud.google.com/)) fees. Luckily, Google offers any new GCP account with $300 in cloud credits, which quickly offset any costs for GCP for a few months.   

Unfortunately, as of writing, SG2-ADA has not seen a TPU release just yet, so I had to train the SG2-ADA model using a GPU. Luckily, Google's generous hands help me out again by providing a powerful GPU through [Google Colab](https://colab.research.google.com/)

### Transfer Learning
In both methods, the basic process of training the model is pointing the corresponding training script towards a dataset, and, to reduce the training time, a "pretrained model" where my new model can train from. The process of starting training with an existing similar model is called "Transfer Learning." For the SG2-ADA run, I opted to transfer from Nvidia's [AFHQ Wild pretrained model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/)

## Results
The results are astounding. The quality of the images generated is so good, that very often you couldn't tell a fake generated fox from reals. 

Below are nine completely SG2-ADA generated images, these foxes do not exist:
![](../images/foxes/SG2ADA/samples-grid1.png)

## StyleGAN features