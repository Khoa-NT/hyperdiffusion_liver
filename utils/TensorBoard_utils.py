"""
Lightweight TensorBoard helpers for MICCAI 2025 experiments.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-12-24
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Minimal TensorBoard logger used by training scripts."""

    def __init__(self, log_dir: str | None = None) -> None:
        """Create a writer.

        Args:
            log_dir (str | None): Output directory. Defaults to __None__ for ``./runs``.
        """

        self.writer = SummaryWriter(log_dir=log_dir)


    def scalar_summary(self, tag: str, value: float | int, step: int) -> None:
        """Log a scalar value."""

        self.writer.add_scalar(tag, value, step)
        self.writer.flush()


    def scalars_summary(self, tag: str, value_dict: dict[str, float | int], step: int, prefix: str = "") -> None:
        """Log multiple scalar values under a common tag."""

        for key, val in value_dict.items():
            self.writer.add_scalar(f"{tag}/{prefix}{key}", val, step)
        self.writer.flush()


    def image_summary(self, tag: str, images: np.ndarray | torch.Tensor, step: int, dataformats: str = "HWC") -> None:
        """Log a single image.

        Args:
            tag (str): TensorBoard image tag.
            images (np.ndarray | torch.Tensor): Image array.
            step (int): Step index.
            dataformats (str): TensorBoard dataformat string. Defaults to __"HWC"__.
        """

        if isinstance(images, torch.Tensor):
            self.writer.add_image(tag, images.type(torch.uint8), step, dataformats=dataformats)
        else:
            self.writer.add_image(tag, images.astype(np.uint8), step, dataformats=dataformats)
        self.writer.flush()


    def add_text(self, tag, text_string, step):
        """
        If the text_string is from yaml. Then yaml should use
        key: |
            markdown text

        If the text_string is from multi-line string. Then text_string should use
        ```
        markdown text
        ```

        """
        self.writer.add_text(tag=tag, text_string=text_string, global_step=step)
        self.writer.flush()
