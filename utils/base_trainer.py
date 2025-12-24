"""
Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-12-24
"""

import logging
from pathlib import Path
from omegaconf import DictConfig

from utils import pytorch_utils, TensorBoard_utils, hydra_utils

class BaseTrainer:
    """
    This class is the base class for all the trainers.
    It contains the common functions for all the trainers.
    """
    def __init__(self, cfg: DictConfig, logger: logging.Logger, rng: pytorch_utils.SeedAll, current_path: Path|None=None):
        """
        Args:
            cfg (DictConfig): The config.
            logger (logging.Logger): The logger.
            rng (pytorch_utils.SeedAll): The random number generator.
            current_path (Path, optional): The current path to save the results in case of running trainer multiple times. 
                                           For example, if running multiple times then the directory of this trainer will be: 
                                           self.cfg.current_path / <current_path.stem>
                                           Defaults to None.
        """
        ### Shorten the config
        self.cfg = cfg

        self.logger = logger
        
        ### Create tensorboard writer
        self.tb_writer = TensorBoard_utils.Logger(log_dir=self.cfg.current_path if current_path is None else current_path)
        self.tb_writer.add_text(tag="config", text_string=hydra_utils.cfg2md(self.cfg), step=0)
        self.tb_writer.add_text(tag="_note_", text_string=self.cfg.note, step=0)
        
        self.rng: pytorch_utils.SeedAll = rng

        ### ------------------------------------------------------------------------------------------------ ###
        ###                         Example for Dataset, DataLoader, Optimizer, Scheduler                    ###
        ### ------------------------------------------------------------------------------------------------ ###
        # ### Training set
        # self.train_set = hydra.utils.instantiate(cfg.dataset[cfg.dataset.selected], split_path=self.cfg.split_path.train)
        # self.train_loader = hydra.utils.instantiate(cfg.data_loader.train, dataset=self.train_set)

        # ### Validation set
        # self.val_set = hydra.utils.instantiate(cfg.dataset[cfg.dataset.selected], split_path=self.cfg.split_path.val)
        # self.val_loader = hydra.utils.instantiate(cfg.data_loader.test, dataset=self.val_set)

        # ### Test set
        # self.test_set = hydra.utils.instantiate(cfg.dataset[cfg.dataset.selected], split_path=self.cfg.split_path.test)
        # self.test_loader = hydra.utils.instantiate(cfg.data_loader.test, dataset=self.test_set)

        # ### Create optimizer and scheduler
        # if self.cfg.running_mode == "train":
        #     ### Create optimizer
        #     self.optimizer = hydra.utils.instantiate(cfg.optimizer[cfg.optimizer.selected], params=self.model.parameters())

        #     ### Create scheduler
        #     if self.cfg.scheduler.selected == "get_cosine_schedule_with_warmup":
        #         from utils.pytorch_utils import get_cosine_schedule_with_warmup

        #         ### Get the number of training steps and warmup steps
        #         num_training_steps = len(self.train_loader) * self.cfg.epochs ### Total batches in all epochs
        #         num_warmup_steps = int(self.cfg.scheduler.get_cosine_schedule_with_warmup.percentage_warmup * num_training_steps) ### Percentage of the total training steps to warm up the learning rate

        #         self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, 
        #                                                          num_warmup_steps=num_warmup_steps, 
        #                                                          num_training_steps=num_training_steps)
        #         self.logger.info(f"get_cosine_schedule_with_warmup {self.cfg.scheduler.get_cosine_schedule_with_warmup.percentage_warmup}: num_warmup_steps: {num_warmup_steps}, num_training_steps: {num_training_steps}")
        #         self.scheduler_step_at_epoch = False ### Step during batches

    def train(self) -> None:
        """
        This function is the main training loop.
        """
        raise NotImplementedError("This function should be implemented by the subclass.")

    def train_step(self) -> float:
        """
        This function is the main training step.
        """
        raise NotImplementedError("This function should be implemented by the subclass.")

    def val_step(self) -> None:
        """
        This function is the main validation step.
        """
        raise NotImplementedError("This function should be implemented by the subclass.")

    def training_epoch_end(self) -> None:
        """
        This function is called at the end of each training epoch.
        """
        raise NotImplementedError("This function should be implemented by the subclass.")

    def test(self) -> None:
        """
        This function is the main test function.
        """
        raise NotImplementedError("This function should be implemented by the subclass.")

    def test_step(self) -> None:
        """
        This function is the main test step.
        """
        raise NotImplementedError("This function should be implemented by the subclass.")
    

    ### ------------------- Helper functions ------------------- ###
    def get_epoch_range(self) -> tuple[range, int, int]:
        """
        This function returns the epoch range.
        """
        if "start_epoch" in self.cfg and self.cfg.start_epoch is not None:
            start_epoch = self.cfg.start_epoch + 1
            end_epoch = self.cfg.start_epoch + self.cfg.epochs
            epoch_range = range(start_epoch, end_epoch + 1)
        else:
            start_epoch = 1
            end_epoch = self.cfg.epochs
            epoch_range = range(1, end_epoch + 1)

        self.logger.info(f"Start training from epoch {start_epoch} to {end_epoch}...")
        return epoch_range, start_epoch, end_epoch
    
    def get_ckpt_path(self) -> tuple[Path | None, Path | None]:
        """
        This function returns the checkpoint path of the model and the optimizer(optional).
        """
        if "load_ckpt_path" in self.cfg and self.cfg.load_ckpt_path is not None and self.cfg.load_ckpt_path != "":
            model_ckpt_path = Path(self.cfg.load_ckpt_path)
            if "load_optimizer_ckpt_path" in self.cfg and self.cfg.load_optimizer_ckpt_path is not None and self.cfg.load_optimizer_ckpt_path != "":
                optimizer_ckpt_path = Path(self.cfg.load_optimizer_ckpt_path)
            else:
                optimizer_ckpt_path = None
            return model_ckpt_path, optimizer_ckpt_path
        else:
            return None, None