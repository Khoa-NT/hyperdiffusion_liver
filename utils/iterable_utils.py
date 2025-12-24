"""
This file contains some useful functions for iterables.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-02-16
"""
from typing import Union
from pathlib import Path
import json



# ================================== #
# List
# ================================== #

def num_to_groups(num: int, divisor: int) -> list[int]:
    """
    Convert a number to a list of numbers that sum to the original number.
    Taken and modified from lucidrains/denoising-diffusion-pytorch

    Args:
        num (int): The number to convert.
        divisor (int): The divisor.

    Returns:
        A list of numbers that sum to the original number.
    """
    groups, remainder = divmod(num, divisor)

    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
        
    return arr



# ================================== #
# Json
# ================================== #
def save_json(file_name: Union[str, Path], data_dict: dict, save_path: Path = None, sort_keys=False):
    # If save_path is given, concatenate the file_name and save_path
    if save_path is not None:
        if not isinstance(save_path, Path):
            save_path = Path(save_path)

        file_save_path = save_path / file_name

    else:
        file_save_path = file_name

    with open(file_save_path, 'w') as f:
        json.dump(data_dict, f, indent=4, sort_keys=sort_keys)
