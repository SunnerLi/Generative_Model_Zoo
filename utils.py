import torch
import random
import numpy as np
import hydra
import os
from omegaconf import DictConfig
from torchvision.utils import make_grid

def set_seed(seed: int = 0):
    """Set manual seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ========== Instantiate & Initialization ==========
def instantiate(config: DictConfig | None):
    if config is None:
        return config
    if isinstance(config, DictConfig):
        return hydra.utils.instantiate(config) 
    else:
        raise NotImplementedError()
    
def initial_orthogonal(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)

# ==================== Sampling ====================
def q_sample(scheduler, model_output, timesteps, sample):
    """Perform posterior backward sampling in batch"""
    return torch.stack([
        scheduler.step(o, t, s).prev_sample
        for o, t, s in zip(model_output, timesteps, sample)
    ])

def q_rev_sample(scheduler, model_output, timesteps, sample):
    """Perform posterior forward sampling in batch"""

    def rev_step(
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ):
        """q(x_t|x_(t-1)) = N(sqrt(1-beta_t)*x_(t-1), beta_t*I)
        We use model_output to approximate noise
        """
        beta_t = scheduler.betas[timestep]
        return ((1 - beta_t) ** 0.5) * sample + beta_t * model_output

    return torch.stack([
        rev_step(o, t, s)
        for o, t, s in zip(model_output, timesteps, sample)
    ])

def reparam_trick(scheduler, model_output, timesteps, sample):
    mu, logvar = model_output.chunk(2, dim=1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def no_sample(scheduler, model_output, timesteps, sample):
    return model_output

# ====================== IO ======================
def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_figure(accelerator, name_image_pair: dict = {}, global_step: int = 0):
    for name, image in name_image_pair.items():
        if image is not None:
            accelerator.get_tracker("tensorboard").writer.add_image(
                name, 
                make_grid(image), 
                global_step=global_step
            )
