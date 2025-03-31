"""This script defines noise schedulers which will be used in diffusion model & flow matching
Currently we provide the schedulers. Please fill the class name into noise_scheduler YAML:

    # Diffusion model
    1. src.noise_scheduler.DDPMScheduler
    2. src.noise_scheduler.DDIMScheduler
    3. src.noise_scheduler.DPMSolverSinglestepScheduler
    4. src.noise_scheduler.DPMSolverMultistepScheduler

    # Flow matching
    5. src.noise_scheduler.AffineOTProbPathScheduler
"""

from flow_matching.path import CondOTProbPath
from typing import Tuple
import diffusers
import importlib
import torch


def build_noise_scheduler(
    noise_scheduler_cls, num_train_timesteps: int = 1000, *args, **kwargs
):
    if noise_scheduler_cls is not None:
        if isinstance(noise_scheduler_cls, str):
            module_name, class_name = noise_scheduler_cls.rsplit(".", 1)
            noise_scheduler_cls = getattr(
                importlib.import_module(module_name), class_name
            )
        noise_scheduler = noise_scheduler_cls(
            num_train_timesteps, *args, **kwargs
        )
    return noise_scheduler


# ====================== Diffusion scheduler ======================
def create_custom_scheduler_class(name, base_class):
    """Create custom scheduler to adapt torchdiffeq API"""

    class CustomScheduler(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_train_timesteps = 2 * self.timesteps[0] - self.timesteps[1]

        def add_noise(
            self, original_samples, noise, timesteps, *args, **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor]:
            r"""Generate sample x_t ~ q(x_t|x_0)

            Returns:
                x_t: Sample in time t.
                noise: The target which model need to learn.
            """
            x_t = super().add_noise(
                original_samples, noise, timesteps, *args, **kwargs
            )
            return x_t, noise

    CustomScheduler.__name__ = name
    return CustomScheduler


# Register custom scheduler by original name
for name in [
    "DDPMScheduler",
    "DDIMScheduler",
    "DPMSolverSinglestepScheduler",
    "DPMSolverMultistepScheduler",
]:
    base_class = getattr(diffusers, name)
    globals()[name] = create_custom_scheduler_class(name, base_class)


# ====================== Flow matching scheduler ======================
class AffineOTProbPathScheduler(CondOTProbPath):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.linspace(
            0.0, num_train_timesteps, num_train_timesteps, dtype=torch.float32
        )

    def add_noise(
        self, original_samples, noise, timesteps
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Generate sample x_t of PFODE in time t.

        Returns:
            x_t: Sample in time t.
            dx_t: The target vector which model need to learn.
        """
        path_sample = super().sample(
            noise, original_samples, timesteps / self.num_train_timesteps
        )
        return path_sample.x_t, path_sample.dx_t

    def set_timesteps(self, num_inference_steps: int, *args, **kwargs):
        # Dummy function. We do not use this API to set timesteps in flow matching
        pass

    def previous_timestep(self, timestep):
        return timestep - 1
