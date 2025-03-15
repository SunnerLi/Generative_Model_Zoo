import torch
import random
import numpy as np
import os

def q_sample(scheduler, model_output, timesteps, sample):
    """Perform posterior sampling in batch"""
    return torch.stack([
        scheduler.step(o, t, s).prev_sample
        for o, t, s in zip(model_output, timesteps, sample)
    ])

def set_seed(seed: int = 0):
    """Set manual seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)