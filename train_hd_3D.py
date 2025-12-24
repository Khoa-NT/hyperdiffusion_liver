"""
Train and test the diffusion component for MICCAI 2025 experiments.

Configuration is managed by Hydra via `configs/train_hd_3D.yaml`.
Set `HYDRA_FULL_ERROR=1` for verbose Hydra traces when debugging.
"""

from __future__ import annotations

### Using `egl` to make pyrender work
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import logging

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

from utils import pytorch_utils, hydra_utils, my_utils


@hydra.main(version_base=None, config_path="configs", config_name="train_hd_3D")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)

    logger.info("Pre-process configuration")
    cfg = hydra_utils.preprocess_cfg(cfg, extra_folders=cfg.extra_folders, logger=logger, verbose=True)

    ### Create random generator collection
    rng = pytorch_utils.SeedAll(cfg.seed, logger=logger)

    ### Create Trainer
    hd_trainer = hydra.utils.instantiate(cfg.diffusion_method[cfg.diffusion_method.selected].trainer, cfg, logger, rng)

    timer = my_utils.Timer(logger=logger)
    ### If running_mode is train, then train the model
    if cfg.running_mode == "train":
        logger.info("Training the model")
        hd_trainer.train()
        timer.soft_stop(prefix="[-] Training")

    ### Testing the model
    logger.info("Testing the model")
    hd_trainer.test()
    timer.soft_stop(prefix="[-] Testing")
    timer.stop(prefix="[-] Total")


if __name__ == "__main__":
    main()

