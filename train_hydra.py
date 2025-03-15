
import hydra
import os
import rootutils  # pip3 install rootutils
from omegaconf import DictConfig, OmegaConf

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from train import train

@hydra.main(version_base="1.3", config_path="./conf", config_name="train.yaml")
def train_by_config(cfg: DictConfig):
    return train(
        output_dir=cfg.paths.output_dir,
        model_G_path=os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "G.pth"), 
        model_D_path=os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "D.pth"), 
        log_dir=cfg.paths.log_dir, 
        dataset_path=cfg.data.dataset_path, 
        preprocess=cfg.data.preprocess, 
        batch_size=cfg.data.batch_size,
        G=cfg.model.G,
        D=cfg.model.get("D", None),
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        project_name=cfg.project_name,
        optimizer_cls=cfg.trainer.optimizer_cls, 
        lr=cfg.trainer.lr, 
        weight_decay=cfg.trainer.weight_decay, 
        lr_scheduler_cls=cfg.trainer.lr_scheduler_cls, 
        epochs=cfg.trainer.epochs, 
        seed=cfg.trainer.seed, 
        crit_diff=cfg.loss.crit_diff, 
        crit_gan=cfg.loss.get("crit_gan", None), 
        lambda_gp=cfg.loss.lambda_gp, 
        noise_scheduler=cfg.get("noise_scheduler", None), 
    )

if __name__ == '__main__':
    train_by_config()
