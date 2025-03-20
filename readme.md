<div align="center">
    <h1 align="center">Generative Model Zoo</h1>
    <img src="image.png" />
    <img src="https://img.shields.io/badge/Pytorch-2.4.1-red.svg" alt="Example Badge">
    <img src="https://img.shields.io/badge/Python-3.11.9-blue.svg" alt="Example Badge">
    <img src="https://img.shields.io/badge/Hydra-1.3.2-purple.svg" alt="Example Badge">
    <img src="https://img.shields.io/badge/Diffusers-0.32.2-yellow.svg" alt="Example Badge">
    <img src="https://img.shields.io/badge/Accelerate-1.0.0-yellow.svg" alt="Example Badge">
    <img src="https://img.shields.io/badge/Datasets-3.0.1-yellow.svg" alt="Example Badge">
</div>

This repository provides unified framework to train common generative models, including VAE, GAN and Diffusion model. 

### Usage

* You can simply train & sampling for MNIST diffusion model with argparse
```shell
# Train MNIST diffusion model
python3 train.py

# Sampling for MNIST diffusion model
python3 eval.py
```

* For advance, we use hydra to extend other various models, including AE, GAN and Diffusion model.
```shell
# Train MNIST diffusion model
python3 hydra_wrapper.py --task train optimizer=diffusion loss=diffusion model=size_32_g

# Train CIFAR-10 GAN
python3 hydra_wrapper.py --task train optimizer=style-gan loss=lsgan     model=size_32_g_d ++noise_scheduler=null

# Train an auto-encoder for MNIST
python3 hydra_wrapper.py --task train optimizer=diffusion loss=ae        model=size_32_g_b ++noise_scheduler=null

# Sampling for MNIST diffusion model via hydra
python3 hydra_wrapper.py --task eval grid=1
```