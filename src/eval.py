import os

import fire
import rootutils
import torch
import omegaconf
from accelerate import Accelerator  # pip3 install tensorboard
from diffusers import DDPMScheduler, UNet2DModel
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import build_data_loader

def evaluate(G, B, noise_scheduler, noise: torch.Tensor):
    """Core evaluate function to perform inference. 
    We only use B to generate sample in VAE case (B is existed & no noise scheduler)
    otherwise, we use G to generate sample.
    """
    input = noise
    with torch.no_grad():
        if noise_scheduler:
            for t in noise_scheduler.timesteps:
                noise_pred = G(input, t).sample
                input = noise_scheduler.step(noise_pred, t, input).prev_sample
        else:
            if B:
                input = B(input, torch.zeros(input.shape[0], device=input.device)).sample
            else:
                input = G(input, torch.zeros(input.shape[0], device=input.device)).sample
    return input

def eval(
    output_dir: str = "./output/sample",
    G = UNet2DModel(32, 1, 1),
    B = None,
    model_G_path: str = "./output/model/G_0001.pth",
    model_B_path: str = None,
    dataset_path: str = "ylecun/mnist",
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]),
    batch_size: int = 8,
    noise_scheduler = DDPMScheduler(10),
    num_sample: int = 16,
    grid: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    # Define dataloader/optimizer/lr scheduler/noise scheduler
    loader = build_data_loader(dataset_path, preprocess, batch_size, split="test")

    # Load pre-trained model
    G = instantiate(G) if isinstance(G, omegaconf.DictConfig) else G
    G.load_state_dict(torch.load(model_G_path, weights_only=True))
    if B:
        B = instantiate(B) if isinstance(B, omegaconf.DictConfig) else B
        B.load_state_dict(torch.load(model_B_path, weights_only=True))

    if noise_scheduler is not None:
        noise_scheduler = instantiate(noise_scheduler) if isinstance(noise_scheduler, omegaconf.DictConfig) else noise_scheduler

    # Set accelerator (GPU inference)
    accelerator = Accelerator()
    loader, G = accelerator.prepare(loader, G)
    B = accelerator.prepare(B) if B else B
    device = accelerator.device

    # ========== Sampling ==========
    image_golden = next(iter(loader))['image']
    for sample_idx in tqdm(range(0, num_sample, batch_size)):
        noise = torch.randn_like(image_golden, device=device)
        image = evaluate(G=G, B=B, noise_scheduler=noise_scheduler, noise=noise)

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