"""
This file contains some useful functions for exporting figures.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2024-11-30
"""

import io
import numpy as np
from matplotlib import pyplot as plt


def fig_to_numpy(input_fig:plt.Figure, close_fig:bool=True) -> np.ndarray:
    """
    Convert matplotlib figure to numpy array

    Args:
        input_fig (plt.Figure): The figure
        close_fig (bool, optional): Close the figure after converting. 
                                    If not, it will plot on jupyter notebook.
                                    Defaults to True.

    Returns:
        np.ndarray: numpy array RGBA has shape (H, W, 4)

    """
    
    with io.BytesIO() as buff:
        input_fig.tight_layout() ### Add to remove the white space area
        input_fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)

    w, h = input_fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))

    ### Close the figure
    if close_fig:
        plt.close(input_fig)

    return im