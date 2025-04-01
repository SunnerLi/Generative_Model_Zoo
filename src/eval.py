import os

import fire
import rootutils
import torch
import omegaconf
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from hydra.utils import instantiate
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image

root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.data import build_data_loader
from src.noise_scheduler import build_noise_scheduler
from src.solver import odeint

def evaluate(
    G, 
    Q,
    B, 
    noise_scheduler, 
    noise: torch.Tensor, 
    sample_G: str = None, 
    method: str = "diffusion", 
    return_intermediates: bool = False
):
    """Core function of evaluation. """
    input = noise
    with torch.no_grad():
        
        # ========== Diffusion model & flow matching sampling logic ==========
        if noise_scheduler:
            num_train_timesteps = noise_scheduler.config.num_train_timesteps

            def ode_func(t, input):
                noise_pred = G(input, t * num_train_timesteps).sample
                if sample_G == "q_sample":
                    # Note: diffuser scheduler timestep is integer; while torchdiffeq timestep is float
                    if method == "diffusion":
                        t = (t * num_train_timesteps).long()
                    input = noise_scheduler.step(noise_pred, t, input).prev_sample
                elif sample_G is None or sample_G == "reparam":
                    input = noise_pred
                else:
                    raise NotImplementedError()
                return input
            
            output = odeint(ode_func, y0=input, t=noise_scheduler.timesteps/num_train_timesteps, method=method)

            if return_intermediates:
                return output
            else:
                return output[-1]
            
        # ========== GAN / VAE / VQVAE sampling logic ==========
        else:
            _t = torch.zeros(input.shape[0], device=input.device)
            if B:
                if Q:
                    input, _ = Q(input)
                output = B(input, _t).sample
            else:
                output = G(input, _t).sample
        return output

def eval(
    output_dir: str = "./output/sample",
    G = UNet2DModel(32, 1, 1),
    Q = None,
    B = None,
    sample_G = None,    # None | q_sample | reparam
    model_G_path: str = "./output/model/G_0001.pth",
    model_Q_path: str = None,
    model_B_path: str = None,
    dataset_path: str = "ylecun/mnist",
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]),
    batch_size: int = 8,
    noise_scheduler_cls = DDPMScheduler,
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 1000,
    method: str = "diffusion",   # diffusion | euler
    num_sample: int = 16,
    grid: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    # Define dataloader/noise scheduler
    loader = build_data_loader(dataset_path, preprocess, batch_size, split="test")
    noise_scheduler = build_noise_scheduler(noise_scheduler_cls, num_train_timesteps)

    # Load pre-trained model
    G = instantiate(G) if isinstance(G, omegaconf.DictConfig) else G
    G.load_state_dict(torch.load(model_G_path, weights_only=True))
    if Q:
        Q = instantiate(Q) if isinstance(Q, omegaconf.DictConfig) else Q
        Q.load_state_dict(torch.load(model_Q_path, weights_only=True))
    if B:
        B = instantiate(B) if isinstance(B, omegaconf.DictConfig) else B
        B.load_state_dict(torch.load(model_B_path, weights_only=True))

    # Define noise scheduler
    if noise_scheduler is not None:
        noise_scheduler = build_noise_scheduler(noise_scheduler_cls, num_train_timesteps)
        noise_scheduler.set_timesteps(num_inference_steps)

    # Set accelerator (GPU inference)
    accelerator = Accelerator()
    loader, G = accelerator.prepare(loader, G)
    Q = accelerator.prepare(Q) if Q else Q
    B = accelerator.prepare(B) if B else B
    device = accelerator.device

    # ========== Sampling ==========
    image_golden = next(iter(loader))['image']
    for sample_idx in tqdm(range(0, num_sample, batch_size)):
        noise_size = list(image_golden.shape)
        noise_size[1] = Q.dim if Q else noise_size[1]
        noise = torch.randn(noise_size, device=device)
        image = evaluate(G=G, Q=Q, B=B, noise_scheduler=noise_scheduler, noise=noise, sample_G=sample_G, method=method)

        # Save to disk
        if grid:
            image = make_grid(image, normalize=True)
            save_image(image, os.path.join(output_dir, f"{sample_idx//batch_size:06d}.png"))
        else:
            for i, img in enumerate(image):
                img = to_pil_image(img)
                img.save(os.path.join(output_dir, f"{sample_idx+i:06d}.png"))


if __name__ == '__main__':
    fire.Fire(eval)