"""This script defines the function to create lr scheduler"""

import importlib


def build_lr_scheduler(lr_scheduler_cls, optim, T_max):
    if isinstance(lr_scheduler_cls, str):
        module_name, class_name = lr_scheduler_cls.rsplit(".", 1)
        lr_scheduler_cls = getattr(
            importlib.import_module(module_name), class_name
        )
    lr_scheduler = lr_scheduler_cls(optim, T_max=T_max)
    return lr_scheduler
