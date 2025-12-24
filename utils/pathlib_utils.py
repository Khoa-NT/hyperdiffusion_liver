"""
Utility functions for pathlib.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2024-11-30
"""

import shutil
import logging
import datetime

import pathlib
from pathlib import Path

from natsort import natsorted


def is_pathlib(input_path):
    return isinstance(input_path, (Path, pathlib.PosixPath, pathlib.WindowsPath))


def convert2Path(input_path):
    if is_pathlib(input_path):
        return input_path
    else:
        return Path(input_path)


def create_folder(folder_name:str, 
                  dir_path:Path|None=None, 
                  parents:bool=True, 
                  exist_ok:bool=True,
                  delete_if_exist:bool=True,
                  logger:logging.Logger|None=None, 
                  verbose:bool=True) -> Path:
    """
    Create a folder and return the path.
    """
    if dir_path is None:
        folder_path = Path(Path.cwd(), folder_name)
    else:
        dir_path = convert2Path(dir_path)
        folder_path = dir_path / folder_name

    if delete_if_exist:
        delete_dir(folder_path)

    folder_path.mkdir(parents=parents, exist_ok=exist_ok)

    if verbose:
        if logger is None:
            print(f'Create folder {folder_name} at {folder_path}')
        else:
            logger.info(f'Create folder {folder_name} at {folder_path}')

    return folder_path



def get_current_path(as_string=False):
    if as_string:
        return str(Path.cwd())
    else:
        return Path.cwd()



def delete_dir(dir_path:Path):
    """
    * pathlib.Path.unlink() removes a file or symbolic link.
    * pathlib.Path.rmdir() removes an empty directory.
    * shutil.rmtree() deletes a directory and all its contents.
    """
    dir_path = convert2Path(dir_path)
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
