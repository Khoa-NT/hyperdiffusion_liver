"""
Utility functions for PyTorch operations.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2024-11-30
"""

import random
import os
import sys
import math
import copy
import multiprocessing
from functools import reduce, partial
from subprocess import call as subprocess_call
from collections import Counter
from pathlib import Path
import json

from omegaconf import DictConfig, OmegaConf, open_dict

import numpy as np

import torch
import torch.cuda as tcuda
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# -------------------------------------------------------------------- #
#                           Reproducibility
# -------------------------------------------------------------------- #
### Ref: https://pytorch.org/docs/stable/notes/randomness.html

class SeedAll:
    def __init__(self, seed, logger=None) -> None:
        """Set seed for reproducibility
        Args:
            seed (_type_, optional): Random seed. Defaults to None.
            logger (_type_, optional): Logger object. Defaults to None.
        """
        self.seed = seed
        
        self.logger = logger
        
        ### Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.numpy_generator = np.random.default_rng(seed=self.seed)

        ### Set PYTHONHASHSEED
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        ### This is on CPU
        self.torch_generator = torch.manual_seed(self.seed) ### Set this one is enough
        self.torch_gpu_generator = torch.Generator(device='cuda').manual_seed(self.seed)


        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if self.logger is None:
            print(f'Using random seed: {self.seed}')
        else:
            self.logger.info(f'Using random seed: {self.seed}')


def seed_worker(worker_id: int) -> None:
    """
    Set seed for workers in DataLoader
    Ref: https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    Args:
        worker_id (): Received worker_id. Ex: 0, 1, 2

    Returns: None

    For example:
    ------------
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # print(f"{worker_id=}")
    # print(f"{torch.initial_seed()=}")
    # print(f"{worker_seed=}")




# -------------------------------------------------------------------- #
#                       Learning Rate Scheduler
# -------------------------------------------------------------------- #

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
        ) -> LambdaLR:
    """
    This code is taken from diffusers.optimization.get_cosine_schedule_with_warmup
    Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/optimization.py#L154

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



# -------------------------------------------------------------------- #
#                           For Analysis
# -------------------------------------------------------------------- #
def calculate_stats(input_tensor: torch.Tensor | np.ndarray, return_dict: bool = True) -> dict | tuple:
    """
    Calculate the stats of the input tensor.
    Args:
        input_tensor (torch.Tensor): The input tensor.
        return_dict (bool): If True, return a dictionary. Otherwise, return a tuple.

    Returns:
        dict | tuple: The stats of the input tensor in dict. If return_dict is False, return a tuple in order of min, max, mean, std.
    """
    ### Convert the input tensor to a torch tensor if it is a numpy array
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor)
 
    ### Check if the input tensor is a torch tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError(f"input_tensor is not a numpy array or a torch tensor. input_tensor.dtype: {input_tensor.dtype}")

    
    ### Calculate the min and max
    min = input_tensor.min().item()
    max = input_tensor.max().item()

    ### Calculate the std and mean
    std, mean = torch.std_mean(input_tensor, keepdim=False)
    std = std.item()
    mean = mean.item()

    ### Return the result
    if return_dict:
        return {
            "min": min,
            "max": max,
            "mean": mean,
            "std": std,
        }
    
    else:
        return min, max, mean, std
