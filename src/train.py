import importlib
import os

import fire
import rootutils
import torch
from accelerate import Accelerator
from diffusers import UNet2DModel
from itertools import chain
from torchvision import transforms
from torch.optim import Optimizer, AdamW, lr_scheduler
from tqdm import tqdm

root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.data import build_data_loader
from src.eval import evaluate
from src.loss import WGANLoss
from src.lr_scheduler import build_lr_scheduler
from src.noise_scheduler import build_noise_scheduler, DDPMScheduler
from src.utils import (
    q_sample, q_rev_sample, reparam_trick, no_sample, 
    set_seed, 
    save_model, save_figure,
    instantiate, initial_orthogonal,
)

def build_optimizer(optimizer_cls, params, lr, betas, weight_decay):
    if isinstance(optimizer_cls, str):
        module_name, class_name = optimizer_cls.rsplit('.', 1)
        optimizer_cls = getattr(importlib.import_module(module_name), class_name)
    if weight_decay is None:
        optim = optimizer_cls(params, lr=lr, betas=betas)
    else:
        optim = optimizer_cls(params, lr=lr, betas=betas, weight_decay=weight_decay)
    return optim

def train(
    # --- paths --- 
    model_dir: str = "./output/model",      # The folder path where the models are stored in.
    log_dir: str = "./logs",                # The folder path where log & tfboard are stored in.
    # --- data --- 
    dataset_path : str = "ylecun/mnist",    # Path or name of the dataset. Please refer to HF dataset.load_dataset().
    preprocess = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]),
    batch_size: int = 4,                    # Batch size in single forward/backward process.
    # --- model ---
    G = UNet2DModel(32, 1, 1),              # Generator model.
    B = None,                               # Decoder model.
    D = None,                               # Discriminator model.
    sample_G = None,                        # Sample method after Generator. None | q_sample | reparam
    sample_B = None,                        # Sample method after Decoder. None | q_rev_sample
    # --- accelerator ---
    project_name: str = "MNIST",            # Tfboard project name.
    gradient_accumulation_steps: int = 4,   # The number of steps that should pass before gradients are accumulated.
    # --- trainer ---
    optimizer_cls: Optimizer = AdamW,       # Type of optimizer.
    betas = (0.9, 0.999),                   # Coefficients used for computing running averages of gradient and its square.
    lr_g: float = 0.0002,                   # Learning rate of Generator
    lr_d: float = 0.0002,                   # Learning rate of Discriminator
    weight_decay: float = 0.01,             # Weight decay coefficient
    lr_scheduler_cls = lr_scheduler.CosineAnnealingLR,
    epochs: int = 1,                        # Total training epochs
    epochs_save_weight: int = 50,           # How often are the model parameters saved
    seed: int = 0,                          # Random seed
    # --- loss ---
    crit_rec = None,                        # Criterion of reconstruction loss
    crit_gan = None,                        # Criterion of adversarial loss. Used in GAN
    crit_diff = torch.nn.MSELoss(),         # Criterion of diffusion regression loss. Used in diffusion / flow matching
    crit_vlb = None,                        # Criterion of variation lower bound loss. Used in VAE
    lambda_rec: float = 0.0,                # Loss term weight of reconstruction loss
    lambda_gan: float = 0.0,                # Loss term weight of adversarial loss
    lambda_diff: float = 1.0,               # Loss term weight of diffusion regression loss.
    lambda_vlb_g: float = 0.0,              # Loss term weight of variation lower bound loss after generator.
    lambda_vlb_b: float = 0.0,              # Loss term weight of variation lower bound loss after decoder.
    lambda_gp: float = 0.0,                 # Loss term weight of gradient panelty. 
    # --- noise scheduler ---
    noise_scheduler_cls = DDPMScheduler,    # Noise scheduler object. Used in diffusion / floa matching. Set None in other cases.
    num_train_timesteps: int = 1000,        # The number of diffusion steps to train the model.
    method: str = "diffusion",              # Numerical method for sampling. diffusion | euler
):
    set_seed(seed)
    os.makedirs(model_dir, exist_ok=True)
    params = {k: v for k, v in locals().items() if isinstance(v, (int, float, str))}
        
    # Instantiate dataloader/model/criterions/noise scheduler
    loader = build_data_loader(dataset_path, preprocess, batch_size, split="train")
    criterions = [crit_rec, crit_gan, crit_diff, crit_vlb]
    crit_rec, crit_gan, crit_diff, crit_vlb = [instantiate(crit) for crit in criterions]
    G, B, D = [instantiate(model) for model in [G, B, D]]
    noise_scheduler = build_noise_scheduler(noise_scheduler_cls, num_train_timesteps)
    optim_g = optim_d = None

    # Initialize generator parameters. Use orthogonal since we use SiLU in all models
    G = G.apply(initial_orthogonal) if G else G
    B = B.apply(initial_orthogonal) if B else B
    D = D.apply(initial_orthogonal) if D else D

    # Instantiate Discriminator optimizer & lr scheduler
    if D is not None:
        optim_d = build_optimizer(optimizer_cls, D.parameters(), lr_d, betas, weight_decay)
        lr_scheduler_d = build_lr_scheduler(lr_scheduler_cls, optim_d, T_max=len(loader) * epochs)
        D.train()

    # Instantiate Generator optimizer & lr scheduler
    ae_params = G.parameters() if B is None else chain(G.parameters(), B.parameters())
    optim_g = build_optimizer(optimizer_cls, ae_params, lr_g, betas, weight_decay)
    lr_scheduler_g = build_lr_scheduler(lr_scheduler_cls, optim_g, T_max=len(loader) * epochs)
    G = G.train()
    B = B.train() if B is not None else B

    # Set accelerator (GPU training/logger) and prepare
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with=["tensorboard"], project_dir=log_dir)
    accelerator.init_trackers(project_name, config=params)    
    G, B, D = accelerator.prepare(G, B, D)
    loader, optim_g, optim_d = accelerator.prepare(loader, optim_g, optim_d)
    crit_rec, crit_gan, crit_diff, crit_vlb = accelerator.prepare(crit_rec, crit_gan, crit_diff, crit_vlb)
    device = accelerator.device

    # ========== Training ==========
    global_step, eval_z, _ = 0, None, 0
    for epoch in range(epochs):
        progress_bar = tqdm(loader, total=len(loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for batch_idx, input in enumerate(loader):
            x = input['image']
            eval_z = torch.randn_like(x, device=device) if eval_z is None else eval_z
            loss0 = torch.zeros((1,), device=device)
            loss_g, loss_d = loss0.clone(), loss0.clone()
            loss_dict, sample_dict = {}, {}

            # Sample timestamp. Set as zero if train w/o diffusion
            if noise_scheduler:
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (x.shape[0],), device=device)
                timesteps_prev = noise_scheduler.previous_timestep(timesteps)
            else:
                timesteps = torch.ones(x.shape[0], device=device)
                timesteps_prev = torch.ones(x.shape[0], device=device)

            # Prepare generator input/target. target is None for GAN since GAN does not perform regression
            noise = torch.randn_like(x)
            in_g, tar_g = noise_scheduler.add_noise(x, noise, timesteps) if noise_scheduler else (x, noise) if B is not None else (noise, None)
            sam_d, _    = noise_scheduler.add_noise(x, noise, timesteps_prev) if noise_scheduler else (x, _)
            sample_dict.update({"in_g": in_g, "sam_d": sam_d, "tar_g": tar_g})

            # Perform forward & backward
            with accelerator.accumulate([module for module in [G, D, B] if module is not None]):

                # ----- Discriminator process -----
                if D and crit_gan:
                    # Initialize loss dict
                    loss_dict_D = ["L_d_gan_prev", "L_d_gp_prev", "L_d_gan_curr", "L_d_gp_curr"]
                    loss_dict_D = {name: loss0 for name in loss_dict_D}

                    # Forward encoder-decoder
                    # Note: rev_b is used to compute t-1 adv loss; sam_b is used to compute t adv loss
                    out_g = G(in_g, timesteps).sample.detach()
                    in_b = {
                        "q_sample": q_sample, "reparam": reparam_trick
                    }.get(sample_G, no_sample)(noise_scheduler, out_g, timesteps, x)
                    out_b = B(in_b, timesteps).sample.detach() if B else in_b
                    sam_b = out_b if sample_B is None and B is None else None
                    rev_b = q_rev_sample(noise_scheduler, out_b, timesteps, x) if sample_B == "q_rev_sample" else None

                    # Forward Discriminator
                    if sam_b is not None:
                        logit_fake = D(sam_b, timesteps_prev).sample
                        logit_real = D(sam_d, timesteps_prev).sample
                        loss_dict_D["L_d_gan_prev"] = lambda_gan * crit_gan(logit_fake, logit_real)
                        if lambda_gp and isinstance(crit_gan, WGANLoss):
                            loss_dict_D["L_d_gp_prev"] = crit_gan.wgan_gp_gradient_penalty(D, sam_d, sam_b, lambda_gp)
                    if rev_b is not None:
                        logit_fake = D(rev_b, timesteps).sample
                        logit_real = D(in_g, timesteps).sample
                        loss_dict_D["L_d_gan_curr"] = lambda_gan * crit_gan(logit_fake, logit_real)
                        if lambda_gp and isinstance(crit_gan, WGANLoss):
                            loss_dict_D["L_d_gp_curr"] = crit_gan.wgan_gp_gradient_penalty(D, in_g, rev_b, lambda_gp)
                    
                    # Backward and update Discriminator
                    optim_d.zero_grad()
                    loss_d = sum(l for l in loss_dict_D.values())
                    accelerator.backward(loss_d)
                    grad_norm_d = accelerator.clip_grad_norm_(D.parameters(), max_norm=1.0)
                    optim_d.step()
                    lr_scheduler_d.step()

                    # Record loss/parameters
                    loss_dict.update({k: v.item() for k, v in loss_dict_D.items()})
                    loss_dict.update({"L_d": loss_d.item(), "lr_d": lr_scheduler_d.get_last_lr()[0]})
                    loss_dict.update({"grad_norm_d": grad_norm_d.item()})

                # ----- Generator process -----
                # Initialize loss dict
                loss_dict_G = ["L_g_gan_prev", "L_g_gan_curr", "L_diff", "L_rec", "L_g_vlb", "L_b_vlb"]
                loss_dict_G = {name: loss0 for name in loss_dict_G}

                # Forward encoder-decoder
                # Note: Besides to compute adv loss, rev_b is also used to compute reconstruction loss
                out_g = G(in_g, timesteps).sample
                in_b = {
                    "q_sample": q_sample, "reparam": reparam_trick
                }.get(sample_G, no_sample)(noise_scheduler, out_g, timesteps, x)
                out_b = B(in_b, timesteps).sample if B else in_b
                sam_b = out_b if sample_B is None and B is None else None
                rev_b = out_b if crit_rec and lambda_rec else None
                rev_b = q_rev_sample(noise_scheduler, out_b, timesteps, x) if sample_B == "q_rev_sample" else rev_b

                # Forward Discriminator
                if D and crit_gan:
                    if sam_b is not None:
                        logit_fake = D(sam_b, timesteps_prev).sample
                        logit_real = D(sam_d, timesteps_prev).sample
                        loss_dict_G["L_g_gan_prev"] = lambda_gan * crit_gan(logit_fake, logit_real, crit_G=True)
                    if rev_b is not None:
                        logit_fake = D(rev_b, timesteps).sample
                        logit_real = D(in_g, timesteps).sample
                        loss_dict_G["L_g_gan_curr"] = lambda_gan * crit_gan(logit_fake, logit_real, crit_G=True)
                        
                # Backward and update Generator
                optim_g.zero_grad()
                loss_dict_G["L_diff"]  = lambda_diff * crit_diff(out_g, tar_g) if crit_diff and lambda_diff else loss0
                loss_dict_G["L_rec"]   = lambda_rec * crit_rec(rev_b, in_g) if crit_rec and lambda_rec else loss0
                loss_dict_G["L_g_vlb"] = lambda_vlb_g * crit_vlb(out_g) if crit_vlb and lambda_vlb_g else loss0
                loss_dict_G["L_b_vlb"] = lambda_vlb_b * crit_vlb(out_b) if crit_vlb and lambda_vlb_b else loss0
                loss_g = sum(l for l in loss_dict_G.values())
                accelerator.backward(loss_g)
                grad_norm_g = accelerator.clip_grad_norm_(G.parameters(), max_norm=1.0)
                grad_norm_b = accelerator.clip_grad_norm_(B.parameters(), max_norm=1.0) if B else loss0
                optim_g.step()
                lr_scheduler_g.step()
                
                # Record loss/parameters
                loss_dict.update({k: v.item() for k, v in loss_dict_G.items()})
                loss_dict.update({"L_g": loss_g.item(), "lr_g": lr_scheduler_g.get_last_lr()[0], "step": global_step})
                loss_dict.update({"grad_norm_g": grad_norm_g.item(), "grad_norm_b": grad_norm_b.item()})
                sample_dict.update({"out_g": out_g, "in_b": in_b, "out_b": out_b, "sam_b": sam_b, "rev_b": rev_b})

                # Update stdout & logger
                progress_bar.update(1)
                progress_bar.set_postfix(**loss_dict)
                accelerator.log(loss_dict, step=global_step)
                global_step += 1

            # Visualize network input/output (for debug)
            if batch_idx == 0:
                save_figure(accelerator, sample_dict, epoch)

        # Sampling with fixed z and store image/weight
        eval_img = evaluate(G=G, B=B, noise_scheduler=noise_scheduler, noise=eval_z, sample_G=sample_G, method=method)
        save_figure(accelerator, {"fix_z_sample": eval_img}, epoch)
        if epochs < epochs_save_weight or (epoch+1) % epochs_save_weight == 0:
            accelerator.wait_for_everyone()
            save_model(model=G, path=os.path.join(model_dir, f"G_{epoch+1:04d}.pth"))
            save_model(model=B, path=os.path.join(model_dir, f"B_{epoch+1:04d}.pth"))
            save_model(model=D, path=os.path.join(model_dir, f"D_{epoch+1:04d}.pth"))

    # Finish training
    accelerator.wait_for_everyone()
    accelerator.end_training()
    progress_bar.close()
    print("Done!")

if __name__ == '__main__':
    fire.Fire(train)