from diffusers import UNet2DModel, DDPMScheduler
from hydra import compose, initialize
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import torch
import fire
import rootutils
import os
import omegaconf
from accelerate import Accelerator  # pip3 install tensorboard
from hydra.utils import instantiate

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data import build_data_loader

def evaluate(model, noise_scheduler, noise: torch.Tensor):
    input = noise
    with torch.no_grad():
        for t in noise_scheduler.timesteps:
            noise_pred = model(input, t).sample
            input = noise_scheduler.step(noise_pred, t, input).prev_sample
    return input

def main(
    output_dir: str,
    G = UNet2DModel(32, 1, 1),
    model_G_path: str = None,
    dataset_path: str = "ylecun/mnist",
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]),
    batch_size: int = 8,
    noise_scheduler_cls = DDPMScheduler,
    num_timesteps = 10,
    num_sample: int = 16,
    grid: bool = False,
    config_name: str = None,
):
    params = locals()

    sample_path = os.path.join(output_dir, 'sample')
    os.makedirs(sample_path, exist_ok=True)

    # Load config from YAML
    if config_name is not None:
        with initialize(version_base=None, config_path="./conf"):
            cfg = compose(config_name=config_name)

        kwargs = {k: v for k, v in cfg.items() if k in params}
        for subtype in ["paths", "model", "data", "trainer"]:
            if subtype in cfg:
                kwargs.update({k: v for k, v in cfg[subtype].items() if k in params})
        if kwargs:
            return main(config_name=None, output_dir=output_dir, **kwargs)

    # Define dataloader/optimizer/lr scheduler/noise scheduler
    loader = build_data_loader(dataset_path, preprocess, batch_size, split="test")
    noise_scheduler = noise_scheduler_cls(num_timesteps) if noise_scheduler_cls else None

    # Load pre-trained model
    G = instantiate(G) if isinstance(G, omegaconf.DictConfig) else G
    G.load_state_dict(torch.load(model_G_path, weights_only=True))

    # Set accelerator (GPU inference)
    accelerator = Accelerator()
    loader, G = accelerator.prepare(loader, G)
    device = accelerator.device

    # ========== Sampling ==========
    image_golden = next(iter(loader))['image']
    for sample_idx in tqdm(range(0, num_sample, batch_size)):
        noise = torch.randn_like(image_golden, device=device)
        image = evaluate(model=G, noise_scheduler=noise_scheduler, noise=noise)

        # Save to disk
        if grid:
            image = make_grid(image, normalize=True)
            save_image(image, os.path.join(sample_path, f"{sample_idx//batch_size:06d}.png"))
        else:
            for i, img in enumerate(image):
                img = to_pil_image(img)
                img.save(os.path.join(sample_path, f"{sample_idx+i:06d}.png"))


if __name__ == '__main__':
    fire.Fire(main)