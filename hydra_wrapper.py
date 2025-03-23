"""This script provides hydra wrapper to perform train or eval with hydra config mechanism. """
import argparse
import hydra
import os
import rootutils  # pip3 install rootutils
import sys
from omegaconf import DictConfig, OmegaConf

root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.eval import eval
from src.train import train

# Parse the task category first and remain rest argument for hydra config
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["train", "eval"], required=True)
args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

@hydra.main(version_base="1.3", config_path="./config", config_name="train.yaml")
def train_by_config(cfg: DictConfig):
    """Hydra wrapper to launch training by hydra config value"""
    return train(
        model_dir=os.path.join(cfg.paths.output_dir),
        log_dir=cfg.paths.log_dir, 
        dataset_path=cfg.data.dataset_path, 
        preprocess=cfg.data.preprocess, 
        batch_size=cfg.data.batch_size,
        G=cfg.model.G,
        B=cfg.model.get("B", None),
        D=cfg.model.get("D", None),
        sample_G=cfg.model.get("sample_G", None),
        sample_B=cfg.model.get("sample_B", None),
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        project_name=cfg.project_name,
        optimizer_cls=cfg.optimizer.optimizer_cls, 
        betas=cfg.optimizer.betas,
        lr_g=cfg.optimizer.lr_g, 
        lr_d=cfg.optimizer.get("lr_d", None),
        weight_decay=cfg.optimizer.get("weight_decay", None), 
        lr_scheduler_cls=cfg.trainer.lr_scheduler_cls, 
        epochs=cfg.trainer.epochs, 
        epochs_save_weight=cfg.trainer.epochs_save_weight,
        seed=cfg.trainer.seed, 
        crit_rec=cfg.loss.get("crit_rec", None), 
        crit_gan=cfg.loss.get("crit_gan", None), 
        crit_diff=cfg.loss.get("crit_diff", None), 
        crit_vlb=cfg.loss.get("crit_vlb", None), 
        lambda_rec=cfg.loss.get("lambda_rec", 0.0),
        lambda_gan=cfg.loss.get("lambda_gan", 0.0),
        lambda_diff=cfg.loss.get("lambda_diff", 0.0),
        lambda_vlb_g=cfg.loss.get("lambda_vlb_g", 0.0),
        lambda_vlb_b=cfg.loss.get("lambda_vlb_b", 0.0),
        lambda_gp=cfg.loss.get("lambda_gp", 0.0),
        noise_scheduler=cfg.get("noise_scheduler", None), 
        method=cfg.get("method", "diffusion"), 
    )

@hydra.main(version_base="1.3", config_path="./config", config_name="eval.yaml")
def eval_by_config(cfg: DictConfig):
    """Hydra wrapper to launch evaluation by hydra config value"""
    return eval(
        output_dir = cfg.paths.output_dir,
        G=cfg.model.G,
        B=cfg.model.get("B", None),
        sample_G=cfg.model.get("sample_G", None),
        model_G_path=cfg.paths.model_G_path,
        model_B_path=cfg.paths.get("model_B_path", None),
        dataset_path=cfg.data.dataset_path, 
        preprocess=cfg.data.preprocess, 
        batch_size=cfg.data.batch_size,
        noise_scheduler=cfg.get("noise_scheduler", None), 
        method=cfg.get("method", "diffusion"), 
        num_sample=cfg.get("num_sample", 16),
        grid=cfg.get("grid", False),
    )

if __name__ == '__main__':
    if args.task == "train":
        train_by_config()
    elif args.task == "eval":
        eval_by_config()
    else:
        print("Not support.")
