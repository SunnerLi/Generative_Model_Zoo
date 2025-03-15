import copy
import importlib

import fire  # pip3 install fire
import hydra
import omegaconf
import rootutils  # pip3 install rootutils
import torch
from accelerate import Accelerator  # pip3 install tensorboard
from diffusers import DDPMScheduler, UNet2DModel
from hydra.utils import instantiate
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data import build_data_loader
from eval import evaluate
from loss import WGANLoss
from utils import q_sample, set_seed

def build_optimizer(optimizer_cls, params, lr, weight_decay):
    if isinstance(optimizer_cls, str):
        module_name, class_name = optimizer_cls.rsplit('.', 1)
        optimizer_cls = getattr(importlib.import_module(module_name), class_name)
    optim = optimizer_cls(params, lr=lr, weight_decay=weight_decay)
    return optim

def build_lr_scheduler(lr_scheduler_cls, optim, T_max):
    if isinstance(lr_scheduler_cls, str):
        module_name, class_name = lr_scheduler_cls.rsplit('.', 1)
        lr_scheduler_cls = getattr(importlib.import_module(module_name), class_name)
    lr_scheduler = lr_scheduler_cls(optim, T_max=T_max)
    return lr_scheduler

def train(
    # --- paths --- 
    output_dir: str = "./output",
    model_G_path: str = "./G.pth",
    model_D_path: str = "./D.pth",
    log_dir: str = "./logs",
    # --- data --- 
    dataset_path : str = "ylecun/mnist",
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]),
    batch_size: int = 4,
    # --- model ---
    G = UNet2DModel(32, 1, 1),
    D = None,
    # --- accelerator ---
    project_name: str = "MNIST",
    gradient_accumulation_steps: int = 4,
    # --- trainer ---
    optimizer_cls = torch.optim.AdamW,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    lr_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR,
    epochs: int = 1,
    seed: int = 0,
    # --- loss ---
    crit_diff = torch.nn.MSELoss(),
    crit_gan = None,
    lambda_gp: float = 10.0,
    # --- noise scheduler ---
    noise_scheduler = DDPMScheduler(1000),  # or None
):
    set_seed(seed)
    params = copy.deepcopy(locals())
        
    # Instantiate dataloader/criterion/noise scheduler
    loader = build_data_loader(dataset_path, preprocess, batch_size, split="train")
    crit_diff = instantiate(crit_diff) if isinstance(crit_diff, omegaconf.DictConfig) else crit_diff
    crit_gan  = instantiate(crit_gan) if isinstance(crit_gan, omegaconf.DictConfig) else crit_gan
    if noise_scheduler is not None:
        noise_scheduler = instantiate(noise_scheduler) if isinstance(noise_scheduler, omegaconf.DictConfig) else noise_scheduler
        num_train_timesteps = 2 * noise_scheduler.timesteps[0] - noise_scheduler.timesteps[1]

    # Instantiate Discriminator model/optimizer/lr scheduler
    if D is not None:
        D = instantiate(D) if isinstance(D, omegaconf.DictConfig) else D
        optim_d = build_optimizer(optimizer_cls, params=D.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler_d = build_lr_scheduler(lr_scheduler_cls, optim_d, T_max=len(loader) * epochs)
        D.train()

    # Instantiate Generator model/optimizer/lr scheduler
    G = instantiate(G) if isinstance(G, omegaconf.DictConfig) else G
    optim_g = build_optimizer(optimizer_cls, params=G.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler_g = build_lr_scheduler(lr_scheduler_cls, optim_g, T_max=len(loader) * epochs)
    G.train()

    # Set accelerator (GPU training/logger)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, 
        log_with=["tensorboard"], project_dir=log_dir
    )
    accelerator.init_trackers(
        project_name, 
        config={k: v for k, v in params.items() if isinstance(v, (int, float, str))},
    )    
    loader, crit_diff, crit_gan, G, optim_g = accelerator.prepare(loader, crit_diff, crit_gan, G, optim_g)
    if D is not None:
        D, optim_d = accelerator.prepare(D, optim_d)
    device = accelerator.device

    # ========== Training ==========
    global_step, eval_z = 0, None
    for epoch in range(epochs):
        progress_bar = tqdm(loader, total=len(loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for input in loader:
            image = input['image']
            eval_z = torch.randn_like(image, device=device) if eval_z is None else eval_z
            loss_zero = torch.zeros((1,), device=device)
            loss_g, loss_d = torch.clone(loss_zero), torch.clone(loss_zero)
            logs = {}

            # Sample timestamp. Set as zero if train w/o diffusion
            if noise_scheduler:
                timesteps = torch.randint(0, num_train_timesteps, (image.shape[0],), device=device)
            else:
                timesteps = torch.zeros(image.shape[0], device=device)

            # Prepare generator input/target. target is None for GAN since GAN does not perform regression
            noise = torch.randn_like(image)
            if noise_scheduler:
                input  = noise_scheduler.add_noise(image, noise, timesteps)
                target = noise
            else:
                input  = noise
                target = None

            # Perform forward & backward
            with accelerator.accumulate([G, D] if D is not None else G):

                # ----- Discriminator process -----
                noise_pred  = G(input, timesteps).sample.detach()
                if D and crit_gan:
                    # Forward
                    if noise_scheduler:
                        renoise_image = q_sample(noise_scheduler, noise_pred, timesteps, input)
                        input_fake = renoise_image
                        input_real = input
                    else:
                        input_fake = noise_pred
                        input_real = image
                    logit_fake = D(input_fake, timesteps).sample
                    logit_real = D(input_real, timesteps).sample

                    # Backward and update Discriminator
                    optim_d.zero_grad()
                    loss_d_gan = crit_gan(logit_fake, logit_real)
                    if lambda_gp and isinstance(crit_gan, WGANLoss):
                        loss_d_gp = crit_gan.wgan_gp_gradient_penalty(D, image, noise_pred)
                    else:
                        loss_d_gp = torch.zeros((1,), device=device)
                    loss_d = loss_d_gan + loss_d_gp
                    accelerator.backward(loss_d)
                    optim_d.step()
                    lr_scheduler_d.step()
                    
                    # Record loss/parameters
                    logs.update({"L_d_gan": loss_d_gan.item(), "L_d_gp": loss_d_gp.item(), "L_d": loss_d.item(), "lr_d": lr_scheduler_d.get_last_lr()[0]})

                # ----- Generator process -----
                noise_pred  = G(input, timesteps).sample
                if D and crit_gan:
                    # Forward
                    renoise_image = q_sample(noise_scheduler, noise_pred, timesteps, input)
                    logit_fake = D(renoise_image, timesteps).sample

                # Backward and update Generator
                optim_g.zero_grad()
                loss_g_gan  = crit_gan(logit_fake, crit_G=True) if crit_gan else loss_zero
                loss_g_diff = crit_diff(noise_pred, target) if target is not None else loss_zero
                loss_g = loss_g_gan + loss_g_diff
                accelerator.backward(loss_g)
                optim_g.step()
                lr_scheduler_g.step()
                
                # Record loss/parameters
                logs.update({"L_g_gan": loss_g_gan.item(), "L_g_diff": loss_g_diff.item(), "L_g": loss_g.item(), "lr_g": lr_scheduler_g.get_last_lr()[0], "step": global_step})

                # Update stdout & logger
                progress_bar.update(1)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        # Sampling with fixed z and store image/weight
        eval_image = evaluate(model=G, noise_scheduler=noise_scheduler, noise=eval_z)
        accelerator.get_tracker("tensorboard").writer.add_image("eval_image", make_grid(eval_image), global_step=epoch)
        torch.save(G.state_dict(), model_G_path)
        if D is not None:
            torch.save(D.state_dict(), model_D_path)

    # Finish training
    accelerator.end_training()
    progress_bar.close()
    print("Done!")

if __name__ == '__main__':
    fire.Fire(train)