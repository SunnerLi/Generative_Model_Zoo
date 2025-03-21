"""This script defines different loss terms which will be used in training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# ======================== VAE related loss ========================
class VLBLoss(nn.Module):
    def forward(self, input: torch.Tensor):
        mu, logvar = input.chunk(2, dim=1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# ======================== GAN related loss ========================
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_preds, real_preds = None, crit_G: bool = False):
        assert crit_G or (not crit_G and real_preds is not None)

class VallinaGANLoss(GANLoss):
    def __init__(self):
        super().__init__()

    def forward(self, fake_preds, real_preds = None, crit_G: bool = False):
        super().forward(fake_preds, real_preds, crit_G)
        if crit_G:
            return F.binary_cross_entropy_with_logits(
                fake_preds, 
                torch.ones_like(fake_preds, device=fake_preds.device)
            )
        else:
            real_loss = F.binary_cross_entropy_with_logits(
                real_preds, 
                torch.ones_like(real_preds, device=fake_preds.device)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_preds, 
                torch.zeros_like(fake_preds, device=fake_preds.device)
            )
            return real_loss + fake_loss
        
class LSGANLoss(GANLoss):
    def __init__(self):
        super().__init__()

    def forward(self, fake_preds, real_preds = None, crit_G: bool = False):
        super().forward(fake_preds, real_preds, crit_G)
        if crit_G:
            return 0.5 * torch.mean((fake_preds - 1) ** 2)
        else:
            real_loss = torch.mean((real_preds - 1) ** 2)
            fake_loss = torch.mean(fake_preds ** 2)
            return 0.5 * (real_loss + fake_loss)
        
class WGANLoss(GANLoss):
    def __init__(self):
        super().__init__()

    def wgan_gp_gradient_penalty(self, D, real_samples, fake_samples, lambda_gp=10.0):
        device = fake_samples.device
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.size(), device=device)
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return lambda_gp * gradient_penalty

    def forward(self, fake_preds, real_preds = None, crit_G: bool = False):
        super().forward(fake_preds, real_preds, crit_G)
        if crit_G:
            return -torch.mean(fake_preds)
        else:
            return -torch.mean(real_preds) + torch.mean(fake_preds)

class HingeGANLoss(GANLoss):
    def __init__(self):
        super().__init__()

    def forward(self, fake_preds, real_preds = None, crit_G: bool = False):
        super().forward(fake_preds, real_preds, crit_G)
        if crit_G:
            return -torch.mean(fake_preds)
        else:
            real_loss = torch.mean(F.relu(1 - real_preds))
            fake_loss = torch.mean(F.relu(1 + fake_preds))
            return real_loss + fake_loss

class RelativisticGANLoss(GANLoss):
    def __init__(self):
        super().__init__()

    def forward(self, fake_preds, real_preds, crit_G: bool = False):
        if crit_G:
            real_loss = F.binary_cross_entropy_with_logits(
                real_preds - torch.mean(fake_preds), 
                torch.zeros_like(real_preds, device=real_preds.device)
                )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_preds - torch.mean(real_preds), 
                torch.ones_like(fake_preds, device=fake_preds.device)
                )
        else:
            real_loss = F.binary_cross_entropy_with_logits(
                real_preds - torch.mean(fake_preds), 
                torch.ones_like(real_preds, device=real_preds.device)
                )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_preds - torch.mean(real_preds), 
                torch.zeros_like(fake_preds, device=fake_preds.device)
                )
        return real_loss + fake_loss
