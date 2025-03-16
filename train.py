import copy
import importlib
import os

import fire  # pip3 install fire
import hydra
import omegaconf
import rootutils  # pip3 install rootutils
import torch
from accelerate import Accelerator  # pip3 install tensorboard
from diffusers import DDPMScheduler, UNet2DModel
from hydra.utils import instantiate
from itertools import chain
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data import build_data_loader
from eval import evaluate
from loss import WGANLoss
from utils import q_sample, q_rev_sample, reparam_trick, no_sample, set_seed

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
    model_dir: str = "./output/model",
    log_dir: str = "./logs",
    # --- data --- 
    dataset_path : str = "ylecun/mnist",
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]),
    batch_size: int = 4,
    # --- model ---
    G = UNet2DModel(32, 1, 1),
    B = None,
    D = None,
    sample_G = None,    # None | q_sample | reparam
    sample_B = None,    # None | q_rev_sample
    # --- accelerator ---
    project_name: str = "MNIST",
    gradient_accumulation_steps: int = 4,
    # --- trainer ---
    optimizer_cls = torch.optim.AdamW,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    lr_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR,
    epochs: int = 1,
    epochs_save_weight: int = 50,
    seed: int = 0,
    # --- loss ---
    crit_rec = None,
    crit_gan = None,
    crit_diff = torch.nn.MSELoss(),
    crit_vlb = None,
    lambda_rec: float = 1.0,
    lambda_gan: float = 1.0,
    lambda_diff: float = 1.0,
    lambda_vlb: float = 1.0,
    lambda_gp: float = 10.0,
    # --- noise scheduler ---
    noise_scheduler = DDPMScheduler(1000),  # or None
):
    set_seed(seed)
    os.makedirs(model_dir, exist_ok=True)
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
    B = instantiate(B) if isinstance(B, omegaconf.DictConfig) else B
    optim_g = build_optimizer(optimizer_cls, params=G.parameters() if B is None else chain(G.parameters(), B.parameters()), lr=lr, weight_decay=weight_decay)
    lr_scheduler_g = build_lr_scheduler(lr_scheduler_cls, optim_g, T_max=len(loader) * epochs)
    G = G.train()
    B = B.train() if B is not None else B

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
    B = accelerator.prepare(B) if B is not None else B
    D, optim_d = accelerator.prepare(D, optim_d) if D is not None else D, None
    device = accelerator.device

    # ========== Training ==========
    global_step, eval_z = 0, None
    for epoch in range(epochs):
        progress_bar = tqdm(loader, total=len(loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for input in loader:
            x = input['image']
            eval_z = torch.randn_like(x, device=device) if eval_z is None else eval_z
            loss_zero = torch.zeros((1,), device=device)
            loss_g, loss_d = torch.clone(loss_zero), torch.clone(loss_zero)
            logs = {}

            # Sample timestamp. Set as zero if train w/o diffusion
            if noise_scheduler:
                timesteps = torch.randint(0, num_train_timesteps, (x.shape[0],), device=device)
            else:
                timesteps = torch.zeros(x.shape[0], device=device)

            # Prepare generator input/target. target is None for GAN since GAN does not perform regression
            noise = torch.randn_like(x)
            in_g  = noise_scheduler.add_noise(x, noise, timesteps) if noise_scheduler else noise
            sam_d = noise_scheduler.add_noise(x, noise, noise_scheduler.previous_timestep(timesteps)) if noise_scheduler else noise

            # Perform forward & backward
            with accelerator.accumulate([module for module in [G, D, B] if module is not None]):

                # ----- Discriminator process -----
                if D and crit_gan:
                    # Initialize loss dict
                    loss_dict = ["L_d_gan_prev", "L_d_gp_prev", "L_d_gan_curr", "L_d_gp_curr"]
                    loss_dict = {name: loss_zero for name in loss_dict}

                    # Forward encoder-decoder
                    out_g = G(in_g, timesteps).sample.detach()
                    in_b = {
                        "q_sample": q_sample, "reparam": reparam_trick,
                    }.get(sample_G, no_sample)(noise_scheduler, out_g, timesteps, x)
                    out_b = B(in_b, timesteps).sample.detach() if B else in_b
                    sam_b = out_b if sample_B is None else None
                    rev_b = q_rev_sample(noise_scheduler, out_g, timesteps, x) if sample_B == "q_rev_sample" else None

                    # Forward Discriminator
                    if sam_b is not None:
                        logit_fake = D(sam_b, noise_scheduler.previous_timestep).sample
                        logit_real = D(sam_d, noise_scheduler.previous_timestep).sample
                        loss_dict["L_d_gan_prev"] = lambda_gan * crit_gan(logit_fake, logit_real)
                        if lambda_gp and isinstance(crit_gan, WGANLoss):
                            loss_dict["L_d_gp_prev"] = crit_gan.wgan_gp_gradient_penalty(D, sam_d, sam_b, lambda_gp)
                    if rev_b is not None:
                        logit_fake = D(rev_b, timesteps).sample
                        logit_real = D(in_g, timesteps).sample
                        loss_dict["L_d_gan_curr"] = lambda_gan * crit_gan(logit_fake, logit_real)
                        if lambda_gp and isinstance(crit_gan, WGANLoss):
                            loss_dict["L_d_gp_curr"] = crit_gan.wgan_gp_gradient_penalty(D, in_g, rev_b, lambda_gp)
                    
                    # Backward and update Discriminator
                    optim_d.zero_grad()
                    loss_d = sum(l for l in loss_dict.values())
                    accelerator.backward(loss_d)
                    optim_d.step()
                    lr_scheduler_d.step()

                    # Record loss/parameters
                    logs.update({k: v.item() for k, v in loss_dict.items()})
                    logs.update({"L_d": loss_d.item(), "lr_d": lr_scheduler_d.get_last_lr()[0]})

                # ----- Generator process -----
                # Initialize loss dict
                loss_dict = ["L_g_gan_prev", "L_g_gan_curr", "L_diff", "L_rec", "L_g_vlb", "L_b_vlb"]
                loss_dict = {name: loss_zero for name in loss_dict}

                # Forward encoder-decoder
                out_g = G(in_g, timesteps).sample
                in_b = {
                    "q_sample": q_sample, "reparam": reparam_trick,
                }.get(sample_G, no_sample)(noise_scheduler, out_g, timesteps, x)
                out_b = B(in_b, timesteps).sample if B else in_b
                sam_b = out_b if sample_B is None else None
                rev_b = q_rev_sample(noise_scheduler, out_g, timesteps, x) if sample_B == "q_rev_sample" else None

                # Forward Discriminator
                if D and crit_gan:
                    if sam_b is not None:
                        logit_fake = D(sam_b, noise_scheduler.previous_timestep).sample
                        loss_dict["L_g_gan_prev"] = lambda_gan * crit_gan(logit_fake, crit_G=True)
                    if rev_b is not None:
                        logit_fake = D(rev_b, timesteps).sample
                        loss_dict["L_g_gan_curr"] = lambda_gan * crit_gan(logit_fake, crit_G=True)
                        
                # Backward and update Generator
                optim_g.zero_grad()
                loss_dict["L_diff"]  = lambda_diff * crit_diff(out_g, noise) if crit_diff and lambda_diff else loss_zero
                loss_dict["L_rec"]   = lambda_rec * crit_rec(rev_b, in_g) if crit_rec and lambda_rec else loss_zero
                loss_dict["L_g_vlb"] = lambda_vlb * crit_vlb(out_g) if crit_vlb and lambda_vlb else loss_zero
                loss_dict["L_b_vlb"] = lambda_vlb * crit_vlb(out_b) if crit_vlb and lambda_vlb else loss_zero
                loss_g = sum(l for l in loss_dict.values())
                accelerator.backward(loss_g)
                optim_g.step()
                lr_scheduler_g.step()
                
                # Record loss/parameters
                logs.update({k: v.item() for k, v in loss_dict.items()})
                logs.update({"L_g": loss_g.item(), "lr_g": lr_scheduler_g.get_last_lr()[0], "step": global_step})

                # Update stdout & logger
                progress_bar.update(1)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        # Sampling with fixed z and store image/weight
        eval_image = evaluate(G=G, B=B, noise_scheduler=noise_scheduler, noise=eval_z)
        accelerator.get_tracker("tensorboard").writer.add_image("eval_image", make_grid(eval_image), global_step=epoch)
        if epochs < epochs_save_weight or (epoch+1) % epochs_save_weight == 0:
            torch.save(G.state_dict(), os.path.join(model_dir, f"G_{epoch+1:04d}.pth"))
            if B is not None:
                torch.save(B.state_dict(), os.path.join(model_dir, f"B_{epoch+1:04d}.pth"))
            if D is not None:
                torch.save(D.state_dict(), os.path.join(model_dir, f"D_{epoch+1:04d}.pth"))

    # Finish training
    accelerator.end_training()
    progress_bar.close()
    print("Done!")

if __name__ == '__main__':
    fire.Fire(train)