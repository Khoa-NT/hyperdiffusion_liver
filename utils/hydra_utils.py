"""
This file contains some useful functions for hydra.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2024-11-30
"""

import collections
from pathlib import Path
from typing import Any, Text
from importlib import import_module

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

import pandas as pd

from .pathlib_utils import get_current_path, create_folder



def display_cfg(input_cfg, logger=None):
    """
    Print cfg as yaml
    Args:
        input_cfg (): input configuration
        logger (): If not None, use the given logger which is from logging
    Returns:
        None
    """
    if logger is None:
        print(OmegaConf.to_yaml(input_cfg, resolve=True))
    else:
        logger.info(f'{OmegaConf.to_yaml(input_cfg, resolve=True)}')



def preprocess_cfg(input_cfg: DictConfig, 
                   extra_folders: list[str] | None=None,
                   logger=None, 
                   verbose: bool=True) -> DictConfig:
    """
    Preprocess the configuration.
    Args:
        input_cfg (DictConfig):
        extra_folders (list[str]): List of folder names that we want to create in the experiment.
        logger ():
        verbose (bool):

    Returns:
        DictConfig:
    """
    ### ----------------------------- Update null configuration ----------------------------- ###
    ### The path where we operate the code from terminal.
    input_cfg.terminal_path = Path(hydra.utils.get_original_cwd())

    ### Relative path/the path of the folder where the code is running
    current_path = get_current_path()
    ### Have to convert to string because omegaconf doesn't support PosixPath
    ### Update 2025-02-12: We can use Path object now. Don't need to convert to string anymore.
    input_cfg.current_path = Path(current_path)

    ### The place for saving check point
    ### I think it's not necessary to create a folder for this for general usage
    # input_cfg.ckpt_path = str(current_path / 'ckpt')
    # create_folder('ckpt', logger=logger)

    ### ----------------------------- Create Folders ----------------------------- ###
    ### The place for saving output stuff
    if extra_folders is not None and len(extra_folders) > 0:
        for folder_name in extra_folders:
            folder_path = create_folder(folder_name, dir_path=current_path, logger=logger)

            ### Update the configuration at runtime
            ### Ref: 
            ### https://hydra.cc/docs/upgrades/0.11_to_1.0/strict_mode_flag_deprecated/#adding-fields-at-runtime
            ### https://omegaconf.readthedocs.io/en/latest/usage.html#struct-flag
            with open_dict(input_cfg):
                # input_cfg[f"{folder_name}_path"] = str(folder_path) ### It looks like we can use Path now. Don't need to convert to string anymore.
                input_cfg[f"{folder_name}_path"] = folder_path ### Path object


    ### ----------------------------- Print the configuration ----------------------------- ###
    if verbose:
        if logger is None:
            print("The Configuration of your experiment...")
        else:
            logger.info("The Configuration of your experiment...")

        display_cfg(input_cfg, logger=logger)

    return input_cfg


### ----------------------------- For Tensorboard ----------------------------- ###
### From https://stackoverflow.com/a/66789625
def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, collections.abc.Mapping):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                # print(f'\n{key=}')
                key = list(key)
                # print(f'{key=}')
                key.insert(0, k)  ### Insert the previous key to the in front of
                # print(f'{key=}')
                # print(f'{tuple(key)=}')
                res[tuple(key)] = val
    else:
        # res[()] = nested_dict
        res[''] = nested_dict

    return res


def cfg2md(input_cfg):
    ### Flatten the dict.
    ### All the nested keys will be concatenated together in a tuple.
    flat_dict = flatten_dict(input_cfg)

    ### Convert the flattened dict into DataFrame.
    ### Name the column as 'Setting'
    df = pd.DataFrame.from_dict(flat_dict, orient="index", columns=['Setting'])

    ### Modify the index.
    ### Decouple the concatenated keys.
    df.index = pd.MultiIndex.from_tuples(list(df.index))

    ### Remove the NaN value in the index
    ### From https://stackoverflow.com/a/63528284
    df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna(''))

    ### Reset the index to make column name equally.
    df = df.reset_index()

    ### Set the first column as index.
    df = df.set_index(0)

    ### If the input dict looks like this
    # cfg = {
    #     'data': 'sample_A',
    #     'input_size': [2, 2, 2],
    #     'plans': {
    #         'A': 'plan_A',
    #         'B': 'plan_B',
    #         'C': {
    #             'D': 'plan_D',
    #             'E': 'plan_E',
    #         }
    #     }
    # }
    ### The df now looks like this:
    ###                 1	2	Setting
    ###     0
    ###     data                sample_A
    ###     input_size			[2, 2, 2]
    ###     plans       A		plan_A
    ###     plans       B		plan_B
    ###     plans       C	D	plan_D
    ###     plans       C	E	plan_E

    ### Convert DataFrame to Markdown
    markdown_text = df.to_markdown()

    return markdown_text
