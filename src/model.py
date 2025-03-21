"""This script defines the models which are used additionally in training. """
import torch
import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple, Union

from diffusers import UNet2DModel
from diffusers.models.downsampling import Downsample2D
from diffusers.models.unets.unet_2d import UNet2DOutput


# ======================== Down-sample Discriminator based on diffusers ========================
# We utilize diffusers layers to define down-sample discriminator
# We have several modifications:
#   1. Because official ResBlock2D internally use GroupNorm as 
#      normalization layer which is less adopt in modern 
#      discriminator architecture, we change as spectral norm
#   2. We Follow BigGAN setting in 
#      https://github.com/ajbrock/BigGAN-PyTorch/blob/master/BigGAN.py#L255-L281
#      to use four ResBlock as default. You can change via
#      block_out_channels as diffusers UNet2DModel usage
# ==============================================================================================
class SpectralNormResBlock2D(nn.Module):
    r"""
    A Resnet block with spectral norm.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        non_linearity: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.time_emb_proj = spectral_norm(nn.Linear(temb_channels, out_channels), eps=1e-8)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), eps=1e-8)
        self.downsample = Downsample2D(channels=in_channels)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), eps=1e-8)
        self.nonlinearity = non_linearity
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        input_tensor = self.downsample(input_tensor)
        hidden_states = input_tensor

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]

        hidden_states = hidden_states + temb

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states)

        return output_tensor


class TimestepResidualDiscriminator(UNet2DModel):
    r"""
    A 2D model that takes a sample and a timestep and returns a patch output.

    This model inherits from [`UNet2DModel`]. Check the superclass documentation for it's generic methods.

    Parameters:
        in_channels (`int`, *optional*, defaults to 1): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 1): Number of channels in the output.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
    """

    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 1, 
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        act_fn: str = nn.SELU(),
        time_embedding_dim: Optional[int] = None,
        *args, **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=["DownBlock2D"] * len(block_out_channels),
            up_block_types=["UpBlock2D"] * len(block_out_channels),
            time_embedding_dim=time_embedding_dim,
            *args, **kwargs
        )

        self.conv_in  = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)

        # Clear all blocks in default
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # And define our down-sample blocks
        output_channel = block_out_channels[0]
        for i, channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            down_block = SpectralNormResBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                non_linearity=act_fn,
            )
            self.down_blocks.append(down_block)

        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, 1, 1)

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int],):
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        for downsample_block in self.down_blocks:
            sample = downsample_block(sample, emb)

        return UNet2DOutput(self.conv_out(sample))

# ======================================== 2D DCGAN Model ======================================
# To align input shape of diffusers, we modify G to handle 2D noise input. 
# Ref: https://github.com/gordicaleksa/pytorch-GANs/blob/master/models/definitions/dcgan.py
# ==============================================================================================
class DCGANGenerator(nn.Module):
    def __init__(self, sample_size: int = 32, in_channels: int = 1, out_channels: int = 1, ngf: int = 64):
        super().__init__()

        self.num_blocks = int(np.log2(sample_size) - 2)
        num_channels_per_layer = [out_channels] + [ngf * 2 ** l for l in range(self.num_blocks)]
        num_channels_per_layer = num_channels_per_layer[::-1]

        self.init_volume_shape = (num_channels_per_layer[0], 4, 4)

        # Both with and without bias gave similar results
        self.linear = nn.Linear(in_channels * sample_size ** 2, np.prod(self.init_volume_shape))

        net = []
        for block_idx in range(self.num_blocks):
            net += self.build_block(
                num_channels_per_layer[block_idx], 
                num_channels_per_layer[block_idx+1],
                normalize=block_idx != self.num_blocks-1,
                activation=nn.Tanh() if block_idx == self.num_blocks-1 else None,
            )
        self.net = nn.Sequential(*net)

    def build_block(self, in_channels, out_channels, normalize=True, activation=None):
        # Bias set to True gives unnatural color casts
        layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        # There were debates to whether BatchNorm should go before or after the activation function, in my experiments it
        # did not matter. Goodfellow also had a talk where he mentioned that it should not matter.
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.SiLU() if activation is None else activation)
        return layers
    
    def forward(self, latent_vector_batch, timesteps):
        """Forward process. timesteps is useless. Only valid in diffusion model"""
        latent_vector_batch = latent_vector_batch.view(latent_vector_batch.shape[0], -1)
        latent_vector_batch_projected = self.linear(latent_vector_batch)
        latent_vector_batch_projected_reshaped = latent_vector_batch_projected.view(latent_vector_batch_projected.shape[0], *self.init_volume_shape)
        return UNet2DOutput(self.net(latent_vector_batch_projected_reshaped))
        

class DCGANDiscriminator(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        num_channels_per_layer = [in_channels, 128, 256, 512, out_channels]

        # Since the last volume has a shape = 1024x4x4, we can do 1 more block and since it has a 4x4 kernels it will
        # collapse the spatial dimension into 1x1 and putting channel number to 1 and padding to 0 we get a scalar value
        # that we can pass into Sigmoid - effectively simulating a fully connected layer.
        self.net = nn.Sequential(
            *self.build_block(num_channels_per_layer[0], num_channels_per_layer[1], normalize=False),
            *self.build_block(num_channels_per_layer[1], num_channels_per_layer[2]),
            *self.build_block(num_channels_per_layer[2], num_channels_per_layer[3]),
            *self.build_block(num_channels_per_layer[3], num_channels_per_layer[4], normalize=False, activation=None),
        )
    
    def build_block(self, in_channels, out_channels, normalize=True, activation=nn.LeakyReLU(0.2), padding=1):
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=padding, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(activation)
        return layers

    def forward(self, img_batch, timesteps):
        """Forward process. timesteps is useless. Only valid in diffusion model"""
        return UNet2DOutput(self.net(img_batch))

if __name__ == '__main__':
    # Test to load 32x32 DCGAN generator and discriminator
    G = DCGANGenerator(sample_size=32).cuda()
    D = DCGANDiscriminator().cuda()
    input = torch.randn(4, 1, 32, 32).cuda()
    x = G(input, 0).sample
    y = D(x, 0).sample
    print(D)
    print(x.shape, y.shape)

    # Test to load 32x32 modified diffuser discriminator
    D = TimestepResidualDiscriminator(in_channels=1, out_channels=1).cuda()
    input = torch.randn(4, 1, 32, 32).cuda()
    timesteps  = torch.randint(0, 1000, (4,)).cuda()
    y = D(input, timesteps).sample
    print(D)
    print(y.shape)
