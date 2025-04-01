# Ref: https://github.com/MishaLaskin/vqvae
# Ref: https://github.com/lucidrains/vector-quantize-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.models.unets.unet_2d import UNet2DOutput
from vector_quantize_pytorch import VectorQuantize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(VectorQuantize):
    def forward(self, x, *args, **kwargs):
        x_perm = x.permute(0, 2, 3, 1).contiguous()             # B,C,H,W -> B,H,W,C
        x_flat = x_perm.view(x_perm.shape[0], -1, self.dim)     # B,H,W,C -> B, HW, C
        quantized, _, loss = super().forward(x_flat, *args, **kwargs)
        quantized = quantized.view(x_perm.shape)                # B, HW, C -> B,H,W,C
        quantized  = quantized.permute(0, 3, 1, 2).contiguous() # B,H,W,C -> B,C,H,W
        return quantized, loss
    
class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, e_dim=16, latent_size=8, in_dim=1, h_dim=64, n_res_layers=2, res_h_dim=64):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.e_dim = e_dim
        self.latent_size = latent_size
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

        # self.pre_quantization_conv = nn.Sequential(
        #     nn.Conv2d(h_dim*8*8, e_dim*8*8, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(e_dim*8*8),
        #     nn.ReLU(),
        #     nn.Conv2d(e_dim*8*8, e_dim*8*8, kernel_size=1, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(e_dim*8*8, e_dim*8*8, kernel_size=1, stride=1),
        # )

        self.pre_quantization_conv = nn.Conv2d(h_dim*8*8, e_dim*latent_size*latent_size, kernel_size=1, stride=1)


    def forward(self, x, t):
        output = self.conv_stack(x)
        output_flat = output.view(output.shape[0], -1, 1, 1)
        output_flat = self.pre_quantization_conv(output_flat)
        # output = output_flat.view([output_flat.shape[0], 64, 2, 2])
        output = output_flat.view([output_flat.shape[0], self.e_dim, self.latent_size, self.latent_size])
        return UNet2DOutput(output)
    
class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, e_dim=16, latent_size=8, in_dim=64, h_dim=64, n_res_layers=2, res_h_dim=64):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.post_quantization_conv = nn.Sequential(
            nn.Conv2d(e_dim*latent_size*latent_size, h_dim*4*4, kernel_size=1, stride=1),
            nn.BatchNorm2d(h_dim*4*4),
            nn.ReLU(),
            nn.Conv2d(h_dim*4*4, h_dim*8*8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(h_dim*8*8, h_dim*8*8, kernel_size=1, stride=1),
        )

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 1, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x, t):
        output_flat = x.view(x.shape[0], -1, 1, 1)
        try:
            output_flat = self.post_quantization_conv(output_flat)
        except:
            breakpoint()
        output = output_flat.view([output_flat.shape[0], 64, 8, 8])
        return UNet2DOutput(self.inverse_conv_stack(output))
    

if __name__ == '__main__':
    # Test to inference 32x32 VectorQuantizer
    Q = VectorQuantizer(dim=1, codebook_size=512,).cuda()
    input = torch.randn(4, 3, 32, 32).cuda()
    _, out_zq, _, _, _ = Q(input)
    breakpoint()
    print(out_zq.shape)