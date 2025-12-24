"""
Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-12-24
"""
from pathlib import Path
import logging

import tqdm

import imageio.v3 as iio

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

import trimesh
from trimesh.sample import sample_surface # https://trimesh.org/trimesh.sample.html#trimesh.sample.sample_surface

import numpy as np

import torch
import torch.nn.functional as F

from einops import rearrange, reduce

### Lucidrains https://github.com/lucidrains/denoising-diffusion-pytorch
from denoising_diffusion_pytorch import GaussianDiffusion1D as LucidrainsGaussianDiffusion1D

from models.unet import UNet1DDiffusion, LucidrainsUNet1DDiffusion

from utils import pytorch_utils, TensorBoard_utils, weights_utils, my_utils, iterable_utils, hydra_utils, base_trainer
from utils.hpdf import hd_utils, evaluation_metrics_3d
from utils.hpdf.transformer import Transformer
from utils.hpdf.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType




class BaseTrainer(base_trainer.BaseTrainer):
    """
    This class is modified from the class HyperDiffusion in the hyperdiffusion.py file in the HyperDiffusion paper.
    Removed the pytorch-lightning part.
    """
    def __init__(self, cfg: DictConfig, logger: logging.Logger, rng: pytorch_utils.SeedAll):
        super().__init__(cfg, logger, rng)
        ### Get the GPU device
        self.gpu_device = torch.device("cuda")

        ### Create mesh creator for exporting mesh
        self.mesh_creator = hydra.utils.instantiate(cfg.MeshCreator.init, to_gpu_device=self.gpu_device)

        ### Create liver renderer
        self.liver_render = hd_utils.LiverRender(**cfg.LiverRender)

        ### Create place holder of the MLP model for exporting mesh
        ### len(parameter_sizes)=8, len(parameter_names)=8
        ### parameter_sizes=[3456, 128, 16384, 128, 16384, 128, 128, 1]
        ### parameter_names=['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias']
        ### n_weights=36737 in MLP model
        self.mlp_model = hydra.utils.instantiate(cfg.mlp_model[cfg.mlp_model.selected])
        self.mlp_model.to(self.gpu_device)

        ### Get the parameter sizes and names
        self.parameter_sizes, self.parameter_names = [], []
        for name, param in self.mlp_model.named_parameters():
            self.parameter_sizes.append(param.numel())
            self.parameter_names.append(name)
        self.n_weights = sum(self.parameter_sizes) ### The shape (n_weight) of the data to be used for training. E.g., 36737
        self.logger.info(f"parameter_sizes: {self.parameter_sizes}")
        self.logger.info(f"parameter_names: {self.parameter_names}")
        self.logger.info(f"n_weights: {self.n_weights}") 


        ### Create the model and diffusion process
        self.model = self.create_model()
        if self.cfg.load_ckpt_path is not None and self.cfg.load_ckpt_path != "":
            self.model.load_state_dict(torch.load(self.cfg.load_ckpt_path, weights_only=True))
            self.logger.info(f"Loaded model from ckpt at {self.cfg.load_ckpt_path}")
        else:
            self.logger.info("No ckpt to load")
        self.model.to(self.gpu_device)


        ### We don't need these in gradio demo mode
        if self.cfg.running_mode != "gradio":
            ### Training set
            self.train_set = hydra.utils.instantiate(cfg.dataset[cfg.dataset.selected], split_path=self.cfg.split_path.train)
            self.train_loader = hydra.utils.instantiate(cfg.data_loader.train, dataset=self.train_set, worker_init_fn=pytorch_utils.seed_worker, generator=self.rng.torch_generator)

            ### Validation set
            self.val_set = hydra.utils.instantiate(cfg.dataset[cfg.dataset.selected], split_path=self.cfg.split_path.val)
            self.val_loader = hydra.utils.instantiate(cfg.data_loader.test, dataset=self.val_set, worker_init_fn=pytorch_utils.seed_worker, generator=self.rng.torch_generator)

            ### Test set
            self.test_set = hydra.utils.instantiate(cfg.dataset[cfg.dataset.selected], split_path=self.cfg.split_path.test)
            self.test_loader = hydra.utils.instantiate(cfg.data_loader.test, dataset=self.test_set, worker_init_fn=pytorch_utils.seed_worker, generator=self.rng.torch_generator)

        ### Create optimizer and scheduler
        if self.cfg.running_mode == "train":
            ### Create optimizer
            self.optimizer = hydra.utils.instantiate(cfg.optimizer[cfg.optimizer.selected], params=self.model.parameters())

            ### Create scheduler
            if self.cfg.scheduler.selected == "get_cosine_schedule_with_warmup":
                from utils.pytorch_utils import get_cosine_schedule_with_warmup

                ### Get the number of training steps and warmup steps
                num_training_steps = len(self.train_loader) * self.cfg.epochs ### Total batches in all epochs
                num_warmup_steps = int(self.cfg.scheduler.get_cosine_schedule_with_warmup.percentage_warmup * num_training_steps) ### Percentage of the total training steps to warm up the learning rate

                self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, 
                                                                 num_warmup_steps=num_warmup_steps, 
                                                                 num_training_steps=num_training_steps)
                self.logger.info(f"get_cosine_schedule_with_warmup {self.cfg.scheduler.get_cosine_schedule_with_warmup.percentage_warmup}: num_warmup_steps: {num_warmup_steps}, num_training_steps: {num_training_steps}")
                self.scheduler_step_at_epoch = False ### Step during batches

            else:
                self.scheduler = hydra.utils.instantiate(cfg.scheduler[cfg.scheduler.selected], optimizer=self.optimizer)
                self.scheduler_step_at_epoch = True ### Step at the end of each epoch


    def train(self) -> None:
        ### ------------------------------------ Training loop ------------------------------------ ###
        ### Create the epoch range
        ### We don't use tqdm because we want to print during training
        # if self.cfg.start_epoch is not None:
        #     start_epoch = self.cfg.start_epoch + 1
        #     end_epoch = self.cfg.start_epoch + self.cfg.epochs
        #     epoch_range = range(start_epoch, end_epoch + 1)
        # else:
        #     start_epoch = 1
        #     end_epoch = self.cfg.epochs
        #     epoch_range = range(1, end_epoch + 1)
        # self.logger.info(f"Start training {self.cfg.select_data_name} from epoch {start_epoch} to {end_epoch}...")
        epoch_range, start_epoch, end_epoch = self.get_epoch_range()
        
        ### Initialize the minimum loss and the epoch of the minimum loss
        min_loss = np.inf
        min_loss_epoch = 0

        ### Training loop
        for epoch in epoch_range:
            self.logger.info(f"Epoch {epoch}/{end_epoch}")

            train_loss = self.train_step()
            self.logger.info(f"Loss: {train_loss:>8.6f}")

            self.training_epoch_end(epoch)
            self.val_step(epoch)

            ### Record learning rate and update the learning rate if needed
            self.tb_writer.scalar_summary("learning_rate", self.optimizer.param_groups[0]["lr"], epoch)
            if self.scheduler_step_at_epoch:
                self.scheduler.step()

            ### ---- Post-training stuff ---- ###
            self.tb_writer.scalar_summary("train_loss", train_loss, epoch)

            ### Save the model
            if train_loss < min_loss:
                min_loss = train_loss
                min_loss_epoch = epoch
                lower_loss_state_dict = self.model.state_dict()

            
        ### ------------------------------------ Post-training stuff ------------------------------------ ###
        ### Save the model with the lowest loss
        min_loss_path = self.cfg.ckpt_path / f"min_loss_model_at_{min_loss_epoch}.pt"
        torch.save(lower_loss_state_dict, min_loss_path)
        self.logger.info(f"Save the model with the lowest loss at epoch {min_loss_epoch} at {min_loss_path}")

        ### Save the final model
        torch.save(self.model.state_dict(), self.cfg.ckpt_path / "final_model.pt")
        self.logger.info(f"Save the final model at {self.cfg.ckpt_path / 'final_model.pt'}")

        ### Last Lr
        self.logger.info(f"Last learning rate: {self.optimizer.param_groups[0]['lr']} at epoch {epoch}")


    def train_step(self) -> float:
        self.logger.info("### Running train_step...")
        temp_loss = 0
        self.model.train()
        for input_weights in tqdm.tqdm(self.train_loader, desc="train_step", total=len(self.train_loader), ncols=120): 
            self.optimizer.zero_grad()
            input_weights = input_weights.to(self.gpu_device) ### [B, n_weight]

            ### ------------------- Diffusion ------------------- ###
            ### Get the random diffusion timestep [low, high) based on the batch size. Shape: [B]
            t = torch.randint(0, high=self.cfg.timesteps, size=(input_weights.shape[0],), device=self.gpu_device, dtype=torch.long, generator=self.rng.torch_gpu_generator)

            # Execute a diffusion forward pass
            ### Compute training losses for a single timestep.
            loss = self.training_losses(input_weights, t)

            ### Backward pass
            loss.backward()

            ### Clip gradient norm
            if self.cfg.clip_grad_norm.use:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm.max_norm, self.cfg.clip_grad_norm.norm_type)

            ### Update the model parameters
            self.optimizer.step()

            ### Update the learning rate if needed
            if not self.scheduler_step_at_epoch:
                self.scheduler.step()

            ### Update the total loss
            temp_loss += loss.item() * input_weights.shape[0]

        ### Return the total loss
        temp_loss /= len(self.train_loader.dataset)

        return temp_loss


    @torch.inference_mode()
    def training_epoch_end(self, epoch) -> None:
        ### Only run in an interval
        if epoch % self.cfg.training_epoch_end.interval == 0:
            self.logger.info(f"### Running training_epoch_end at epoch {epoch}...")
            self.model.eval()

            ### ------------------------------------ Sample weights ------------------------------------ ###
            ### Sample `num_samples` weights from the model.
            x_0s = self.sample_loop(self.model, (self.cfg.training_epoch_end.num_samples, self.n_weights)).detach().clone().cpu().float()

            ### ------------------------------------ Generate meshes ------------------------------------ ###
            meshes, occ_cubes, _ = self.generate_meshes(x_0s, export_path=self.cfg.export_mesh_path / f"epoch_{epoch}", 
                                                        export_flag=self.cfg.training_epoch_end.export_mesh, 
                                                        return_occ_cubes=True, return_sample_points=False)

            ### ------------------------------------ Logging ------------------------------------ ###
            ### Statistics of the weights
            x_0s_stats = pytorch_utils.calculate_stats(x_0s)
            # self.logger.info(f"x_0s.stats: {x_0s_stats['min']:.6f}, {x_0s_stats['max']:.6f}, {x_0s_stats['mean']:.6f}, {x_0s_stats['std']:.6f}")
            self.tb_writer.scalars_summary("x_0s_stats", x_0s_stats, epoch)

            ### Statistics of the occupancy cubes
            occ_cubes_stats = pytorch_utils.calculate_stats(np.stack(occ_cubes))
            # self.logger.info(f"occ_cubes.stats: {occ_cubes_stats['min']:.6f}, {occ_cubes_stats['max']:.6f}, {occ_cubes_stats['mean']:.6f}, {occ_cubes_stats['std']:.6f}")
            self.tb_writer.scalars_summary("occ_cubes_stats", occ_cubes_stats, epoch)
            
            ### Render the meshes. [mesh1, mesh2] --> [img1, img2]
            ### recreate_meshes=True: Recreate a plain mesh from the vertices and faces to avoid rendering issues. 
            ### For example, if the mesh already has vertex colors, the rendered image will be overlapped by the vertex colors + pyrender's material.
            rendered_imgs, _ = self.liver_render.render_meshes(meshes, recreate_meshes=True)

            ### Log the rendered images
            ### Stack the list of images into a batch of images. Shape: [N, H, W, C]
            ### Use list because batch_image_summary read list of batch images. [[NHWC], [NHWC], ...]
            self.tb_writer.batch_image_summary("generated_renders/view1_view2", [np.stack(rendered_imgs)], step=epoch)


    def generate_meshes(self, mlp_weights, 
                        export_path: str | Path | None = None, export_flag: bool = False,
                        return_occ_cubes: bool = False,
                        return_sample_points: bool = False,
                        stop_at_threshold: int | None = None,
                        ) -> tuple[list[trimesh.Trimesh], list[np.ndarray], list[np.ndarray]]:
        """
        Args:
            mlp_weights (torch.Tensor): The weights of the MLP model. Shape: [B, n_weight]
            export_path (str | Path): The path to export the meshes.
            export_flag (bool): Whether to export the meshes.
            return_sample_points (bool): Whether to return the sample points.
            stop_at_threshold (int | None): The number of samples to stop at. 
            filter_empty_mesh (bool): Whether to filter out the empty mesh.

        Returns:
            meshes (list[trimesh.Trimesh]): The list of meshes.
            occ_cubes (list[np.ndarray]): The list of occupancy cubes. Only returned if return_occ_cubes is True.
            sample_points (list[np.ndarray]): The list of sample points from the meshes. Only returned if return_sample_points is True.
        """
        ### ------------------------------------ Check export_path ------------------------------------ ###
        if export_path is not None:
            if isinstance(export_path, str):
                export_path = Path(export_path)
            
            if export_flag:
                ### Create the export path if it doesn't exist
                export_path.mkdir(parents=True, exist_ok=True)
        else:
            if export_flag:
                raise ValueError("export_path is None but export_flag is True")

        ### ------------------------------------ Generate meshes ------------------------------------ ###
        meshes = []
        occ_cubes = [] if return_occ_cubes else None
        sample_pcs = [] if return_sample_points else None

        for i, mlp_weight in enumerate(mlp_weights):
            ### Add the weights to the MLP model
            self.mlp_model = weights_utils.add_weights_to_mlp(self.mlp_model, mlp_weight)

            ### Whether to save the mesh. If export_path is None or export_flag is False, the mesh will not be saved.
            if export_path is not None and export_flag:
                mesh, occ_cube = self.mesh_creator.export_mesh(self.mlp_model, file_name=export_path / f"mesh_{i}.ply", export_flag=export_flag, **self.cfg.MeshCreator.export)
            else:
                mesh, occ_cube = self.mesh_creator.export_mesh(self.mlp_model, file_name=None, export_flag=export_flag, **self.cfg.MeshCreator.export)

            ### Check if the mesh is empty
            if mesh is not None:
                ### If the mesh is not empty, add it to the list
                meshes.append(mesh)
            else:
                ### If the mesh is empty, skip it
                self.logger.info(f"Empty mesh at index {i}. Skipping...")
                continue
            
            ### Append the occupancy cube if needed
            if return_occ_cubes:
                occ_cubes.append(occ_cube)
    
            ### Sample points from the mesh
            if return_sample_points:
                ### Sample surface points from the mesh. The number is the same as the number of sampled points in GT. Shape: [N, 3]
                sampled_pc, _ = sample_surface(mesh, self.cfg.calc_metrics.n_sample_points, seed=self.cfg.seed)
                sample_pcs.append(sampled_pc)


            ### If the stop_at_threshold is set, stop the loop
            if stop_at_threshold is not None and stop_at_threshold > 0:
                if len(meshes) >= stop_at_threshold:
                    break

        ### Return the meshes, occ_cubes, and sample_pcs
        return meshes, occ_cubes, sample_pcs
   


    @torch.inference_mode()
    def val_step(self, epoch: int) -> None:
        ### Only run in an interval
        if epoch % self.cfg.val_step.interval == 0:
            self.logger.info(f"### Running validation at epoch {epoch}...")
            self.model.eval()

            ### Calculate the metrics
            train_metrics = self.calc_metrics(self.train_loader, split_name="train", max_samples=self.cfg.val_step.max_samples)
            val_metrics = self.calc_metrics(self.val_loader, split_name="val", max_samples=self.cfg.val_step.max_samples)

            ### Log the metrics
            self.tb_writer.scalars_summary(f"train_metrics_{self.cfg.val_step.max_samples}", train_metrics, epoch)
            self.tb_writer.scalars_summary(f"val_metrics_{self.cfg.val_step.max_samples}", val_metrics, epoch)

            ### Print the metrics
            for (key, value), (key_val, value_val) in zip(train_metrics.items(), val_metrics.items()):
                self.logger.info(f"Train {key}: {value:.6f}, Val {key_val}: {value_val:.6f}")

    

    def calc_metrics(self, input_DataLoader: torch.utils.data.DataLoader, split_name: str="train", 
                     export_flag: bool = False, max_samples: int|None = None) -> dict:
        """
        Calculate the metrics of the generated meshes from the given dataloader.
        Args:
            input_DataLoader (torch.utils.data.DataLoader): The dataloader to calculate the metrics.
            split_name (str): The name of the split to calculate the metrics. {train, val, test}
            export_flag (bool): Whether to export the meshes.
            max_samples (int|None): The maximum number of samples to calculate the metrics. Lower value will speed up the calculation. If None, all samples will be used.

        Returns:
            metrics (dict): The metrics.
        """
        timer = my_utils.Timer(logger=self.logger)

        ### Running calc_metrics on all dataset will take too much time. Therefore, we calculate on a subset of the dataset by setting max_samples.
        if max_samples is not None:
            max_samples = min(max_samples, len(input_DataLoader.dataset))

        self.logger.info(f"### Running calc_metrics for {split_name} with {max_samples if max_samples is not None else 'all'} samples...")
        ### ------------------------------------ Get the surface points from the dataset ------------------------------------ ###
        timer.soft_reset()
        pcs_gt = input_DataLoader.dataset.get_surface_points(num_points=self.cfg.calc_metrics.n_sample_points, 
                                                             seed=self.cfg.seed, max_idx=max_samples) ### list [(N_points, 3), (N_points, 3), ...]
        pcs_gt = torch.from_numpy(np.stack(pcs_gt, axis=0)) ### [N, N_points, 3]
        timer.soft_stop(prefix=f"[-] Sampling points {pcs_gt.shape} from dataset")

        ### ------------------------------------ Sample weights ------------------------------------ ###
        ### In HyperDiffusion paper, they over sample the weights by a factor of 1.1x the number of GT samples then crop to the size of GT.
        ### Oversampling is to ensure the generated mesh is large enough that we have non-empty mesh equal to the size of GT.
        number_of_samples_to_generate = int(len(pcs_gt) * self.cfg.calc_metrics.sample_mult)
        val_test_sample_batch_size = min(number_of_samples_to_generate, self.cfg.val_test_sample_batch_size)

        ### Compute the number of full batches needed, including any remainder
        iter_num_batches = iterable_utils.num_to_groups(number_of_samples_to_generate, val_test_sample_batch_size)
        
        sample_x_0s = []
        for num_batches in tqdm.tqdm(iter_num_batches, desc="Sampling weights", total=len(iter_num_batches), ncols=120):
            ### Sample `num_samples` weights from the model.
            x_0s = self.sample_loop(self.model, (num_batches, self.n_weights)).detach().clone().cpu().float()
            sample_x_0s.append(x_0s)

        ### Concatenate the sample_x_0s 
        # sample_x_0s = torch.cat(sample_x_0s[:len(pcs_gt)], dim=0) ### [N, n_weight] equal to the length of pcs_gt
        sample_x_0s = torch.cat(sample_x_0s, dim=0) ### [N, n_weight]
        timer.soft_stop(prefix=f"[-] Sampling weights {sample_x_0s.shape}")

        ### Save the sample_x_0s as a pth file
        # torch.save(sample_x_0s, self.cfg.ckpt_path / "sample_x_0s__train.pt")

        ### ------------------------------------ Sample points from generated meshes ------------------------------------ ###
        timer.soft_reset()
        meshes, occ_cubes, sample_pcs = self.generate_meshes(sample_x_0s, export_path=self.cfg.export_mesh_path / f"{split_name}", 
                                                             export_flag=export_flag, 
                                                             return_occ_cubes=export_flag, return_sample_points=True,
                                                             stop_at_threshold=len(pcs_gt))
        assert len(sample_pcs) == len(pcs_gt), f"The number of sample points and ground truth points must be the same. But got {len(sample_pcs)}, {len(pcs_gt)}"
        sample_pcs = torch.from_numpy(np.stack(sample_pcs, axis=0)) ### [N, N_points, 3]
        timer.soft_stop(prefix=f"[-] Sampling points {sample_pcs.shape} from generated meshes")

        ### ------------------------------------ Calculate metrics ------------------------------------ ###
        timer.soft_reset()
        sample_pcs = sample_pcs.to(self.gpu_device).float()
        pcs_gt = pcs_gt.to(self.gpu_device).float()
        metrics = evaluation_metrics_3d.compute_all_metrics(sample_pcs, pcs_gt, self.cfg.calc_metrics.compute_all_metrics_batch_size, self.logger)
        fid = hd_utils.calculate_fid_3d(sample_pcs, pcs_gt, batch_size=self.cfg.calc_metrics.calculate_fid_3d_batch_size)
        metrics["fid"] = fid.item()

        timer.soft_stop(prefix="[-] compute_all_metrics & calculate_fid_3d")
        timer.stop(prefix=f"[-] Running calc_metrics for {split_name}")

        
        if export_flag:
            ### ------------------------------------ Export ------------------------------------ ###
            save_path = self.cfg.ckpt_path / f"{split_name}"
            save_path.mkdir(parents=True, exist_ok=True)

            timer.soft_reset()
            ### Export sampled weights, sampled points of generated meshes, and sampled GT points from GT meshes.
            torch.save(sample_x_0s, save_path / f"{split_name}_sample_x_0s.pt")
            torch.save(sample_pcs, save_path / f"{split_name}_sample_pcs.pt")
            torch.save(pcs_gt, save_path / f"{split_name}_pcs_gt.pt")
            timer.soft_stop(prefix=f"[-] Exporting {split_name} sampled weights, sampled points of generated meshes, and sampled GT points from GT meshes to {self.cfg.ckpt_path}")
            
            ### ------------------------------------ Logging ------------------------------------ ###
            timer.soft_reset()
            ### Statistics of the weights
            x_0s_stats = pytorch_utils.calculate_stats(sample_x_0s)
            self.logger.info(f"{split_name}_x_0s.stats: {x_0s_stats['min']:.6f}, {x_0s_stats['max']:.6f}, {x_0s_stats['mean']:.6f}, {x_0s_stats['std']:.6f}")
            self.tb_writer.scalars_summary(f"{split_name}_x_0s_stats", x_0s_stats, 0)

            ### Statistics of the occupancy cubes
            occ_cubes_stats = pytorch_utils.calculate_stats(np.stack(occ_cubes))
            self.logger.info(f"{split_name}_occ_cubes.stats: {occ_cubes_stats['min']:.6f}, {occ_cubes_stats['max']:.6f}, {occ_cubes_stats['mean']:.6f}, {occ_cubes_stats['std']:.6f}")
            self.tb_writer.scalars_summary(f"{split_name}_occ_cubes_stats", occ_cubes_stats, 0)
            timer.soft_stop(prefix=f"[-] Logging {split_name} metrics")

            ### ------------------------------------ Render the meshes ------------------------------------ ###
            save_path = self.cfg.generated_renders_path / f"{split_name}"
            save_path.mkdir(parents=True, exist_ok=True)

            timer.soft_reset()
            ### Render the meshes. [mesh1, mesh2] --> [img1, img2]
            rendered_imgs, _ = self.liver_render.render_meshes(meshes, recreate_meshes=True)
            timer.soft_stop(prefix=f"[-] Rendering {split_name} meshes")

            ### Log the rendered images
            ### Convert the list of images into a batch of images. Shape: [N, H, W, C]
            ### Use list because batch_image_summary read list of batch images. [[NCHW], [NCHW], ...]
            self.tb_writer.batch_image_summary(f"{split_name}_generated_renders/view1_view2", [np.stack(rendered_imgs)], step=0)
            for i, img in enumerate(rendered_imgs):
                iio.imwrite(save_path / f"img_{i}.png", img)
            timer.soft_stop(prefix=f"[-] Logging {split_name} rendered images")
            

        return metrics
    

    def test(self) -> None:
        ### Run test in two cases:
        ### 1. In train mode AND test_step.run_test is True
        ### 2. In any other mode (test, val, etc.)
        if (self.cfg.running_mode == "train" and self.cfg.test_step.run_test) or (self.cfg.running_mode != "train"):
            self.logger.info("### Running test...")
            self.test_step()
        else:
            self.logger.info("### Skipping test...")


    @torch.inference_mode()
    def test_step(self) -> None:
        self.model.eval()
        if self.cfg.test_step.max_samples is None:
            self.logger.info("Running test_step on all samples...")
        else:
            self.logger.info(f"Running test_step on {self.cfg.test_step.max_samples} samples...")

        ### Calculate the metrics for all splits
        train_metrics = self.calc_metrics(self.train_loader, split_name="train", export_flag=True, max_samples=self.cfg.test_step.max_samples)
        val_metrics = self.calc_metrics(self.val_loader, split_name="val", export_flag=True, max_samples=self.cfg.test_step.max_samples)
        test_metrics = self.calc_metrics(self.test_loader, split_name="test", export_flag=True, max_samples=self.cfg.test_step.max_samples)

        ### Log the metrics
        self.tb_writer.scalars_summary("train_metrics_FINAL", train_metrics, 0)
        self.tb_writer.scalars_summary("val_metrics_FINAL", val_metrics, 0)
        self.tb_writer.scalars_summary("test_metrics_FINAL", test_metrics, 0)

        ### Print and save the metrics
        self.logger.info("\n### Final train metrics:")
        for (key, value) in train_metrics.items():
            self.logger.info(f"Train {key}: {value:.6f}")
        iterable_utils.save_json(self.cfg.metrics_path / "train_metrics_FINAL.json", train_metrics)

        self.logger.info("\n### Final val metrics:")
        for (key, value) in val_metrics.items():
            self.logger.info(f"Val {key}: {value:.6f}")
        iterable_utils.save_json(self.cfg.metrics_path / "val_metrics_FINAL.json", val_metrics)

        self.logger.info("\n### Final test metrics:")
        for (key, value) in test_metrics.items():
            self.logger.info(f"Test {key}: {value:.6f}")
        iterable_utils.save_json(self.cfg.metrics_path / "test_metrics_FINAL.json", test_metrics)


    ### ----------------------------------- Subclass methods ----------------------------------- ###
    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def training_losses(self, input_weights: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def sample_loop(self, model: torch.nn.Module, sample_shape: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented by the subclass.")


class HyperDiffusionTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, logger: logging.Logger, rng: pytorch_utils.SeedAll):
        super().__init__(cfg, logger, rng)

    def create_model(self):
        self.transformer_cfg = self.cfg.diffusion_method.hpdf.transformer
        self.diff_cfg = self.cfg.diffusion_method.hpdf.diff_config

        model = Transformer(self.parameter_sizes, self.parameter_names, **self.transformer_cfg)
        
        ### Create diffusion process
        
        betas = np.linspace(1e-4, 2e-2, self.diff_cfg.timesteps) ### betas.shape = [500]
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[self.diff_cfg.model_mean_type], ### START_X
            model_var_type=ModelVarType[self.diff_cfg.model_var_type], ### FIXED_LARGE
            loss_type=LossType[self.diff_cfg.loss_type], ### MSE
            diff_pl_module=model, ### self.model: Transformer class. Will be used to register buffers but temporary comented.
        )

        return model
    
    def training_losses(self, input_weights: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ### loss_terms is a dictionary which contains the loss values in the key "loss".
        loss_terms = self.diff.training_losses(self.model, input_weights * self.diff_cfg.normalization_factor, t)
        
        ### Calculate the mean of the loss values.
        loss_mse = loss_terms["loss"].mean()

        return loss_mse


    def sample_loop(self, model: torch.nn.Module, sample_shape: tuple[int, ...]) -> torch.Tensor:
        ### Sample `num_samples` weights from the model.
        x_0s = self.diff.ddim_sample_loop(model, sample_shape)
        x_0s = x_0s / self.diff_cfg.normalization_factor

        return x_0s


### Taken and modified from https://github.com/lucidrains/denoising-diffusion-pytorch
class lucidrainsTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, logger: logging.Logger, rng: pytorch_utils.SeedAll):
        super().__init__(cfg, logger, rng)

    def create_model(self) -> torch.nn.Module:

        self.lucidrains_cfg = self.cfg.diffusion_method.lucidrains
        self.diff_cfg = self.lucidrains_cfg.diff_config
        self.accelerator_cfg = self.lucidrains_cfg.accelerator

        model = LucidrainsUNet1DDiffusion(self.parameter_sizes, **self.lucidrains_cfg.model)

        ### Create diffusion process
        self.diff = LucidrainsGaussianDiffusion1D(
            model=model,
            seq_length=model.length_size,
            timesteps=self.cfg.timesteps,
            auto_normalize=self.diff_cfg.auto_normalize,
            objective=self.diff_cfg.objective,
        ).to(self.gpu_device)

        return model
    

    def training_losses(self, input_weights: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ### Follow the forward() function in lucidrainsUNet1DDiffusion
        input_weights = self.diff.normalize(input_weights)

        loss = self.p_losses(input_weights, t)
        
        return loss


    def p_losses(self, x_start, t, noise = None):
        """
        Follow the p_losses() function in GaussianDiffusion1D
        """
        noise = torch.randn(x_start.shape, device=x_start.device, generator=self.rng.torch_gpu_generator)

        # noise sample
        x = self.diff.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
    
        x_self_cond = None
        if self.diff.self_condition and self.rng.numpy_generator.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.diff.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.diff.objective == 'pred_noise':
            target = noise
        elif self.diff.objective == 'pred_x0':
            target = x_start
        elif self.diff.objective == 'pred_v':
            v = self.diff.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * self.extract(self.diff.loss_weight, t, loss.shape)
        return loss.mean()

    @staticmethod
    def extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def sample_loop(self, model: torch.nn.Module, sample_shape: tuple[int, ...]) -> torch.Tensor:
        # sampled_weights = self.diff.sample(batch_size=sample_shape[0])

        sample_fn = self.p_sample_loop if not self.diff.is_ddim_sampling else self.ddim_sample

        return sample_fn(sample_shape)
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.diff.betas.device

        img = torch.randn(shape, device=device, generator=self.rng.torch_gpu_generator)

        x_start = None

        for t in tqdm.tqdm(reversed(range(0, self.diff.num_timesteps)), desc = 'sampling loop time step', total = self.diff.num_timesteps, ncols=120):
            self_cond = x_start if self.diff.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.diff.unnormalize(img)
        return img
    
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.diff.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        # noise = torch.randn_like(x, generator=self.rng.torch_gpu_generator) if t > 0 else 0. # no noise if t == 0
        noise = torch.randn(x.shape, device=x.device, generator=self.rng.torch_gpu_generator) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.diff.betas.device, self.diff.num_timesteps, self.diff.sampling_timesteps, self.diff.ddim_sampling_eta, self.diff.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device, generator=self.rng.torch_gpu_generator)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.diff.self_condition else None
            pred_noise, x_start, *_ = self.diff.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.diff.alphas_cumprod[time]
            alpha_next = self.diff.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # noise = torch.randn_like(img, generator=self.rng.torch_gpu_generator)
            noise = torch.randn(img.shape, device=img.device, generator=self.rng.torch_gpu_generator)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.diff.unnormalize(img)
        return img