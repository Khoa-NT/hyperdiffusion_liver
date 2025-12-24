"""
Train the INRs/MLP model for the MICCAI 2025 liver diffusion pipeline.

Check the `configs/train_mlp_3D.yaml` for configuration guidance.
+ If you want to train on a single file, set `data_path` to the path of the file.
+ If you want to train on a directory, set `data_path` to the path of the directory.
"""

from __future__ import annotations

from pathlib import Path
import logging
import trimesh
import tqdm

import numpy as np
import pandas as pd

import torch

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

from utils import pytorch_utils, hydra_utils, mesh_utils, TensorBoard_utils, export_figure


def infer(
    cfg: DictConfig,
    model: torch.nn.Module,
    mesh_creator: mesh_utils.MeshCreator,
    save_path: Path,
    epoch: int,
    logger: logging.Logger,
    tb_writer: TensorBoard_utils.Logger,
    gt_mesh: trimesh.Trimesh | None = None,
    tb_tag_metric_name: str = "metrics",
    plot_mesh_views_flag: bool = True,
    export_flag: bool = True,
    df: pd.DataFrame | None = None,
) -> None:
    """Infer the model and export the mesh.

    Args:
        cfg (DictConfig): Configuration dictionary
        model (torch.nn.Module): Model to infer
        mesh_creator (mesh_utils.MeshCreator): Mesh creator object
        save_path (Path): Path to save the mesh
        epoch (int): Current epoch number
        logger (logging.Logger): Logger object
        tb_writer (TensorBoard_utils.Logger): TensorBoard writer object
        gt_mesh (trimesh.Trimesh | None, optional): Ground truth mesh. Defaults to __None__
        tb_tag_metric_name (str, optional): Name of the tag for the metrics. Defaults to __"metrics"__
        plot_mesh_views_flag (bool, optional): Whether to plot mesh views. Defaults to __True__
        export_flag (bool, optional): Whether to export the mesh. Defaults to __True__
        df (pd.DataFrame | None, optional): Dataframe to save metrics. Defaults to __None__
    """

    logger.info(f"Running the inference at {epoch=}")

    ### Create mesh
    mesh_path = save_path / f"mesh_{epoch}.{mesh_creator.mesh_format}"
    mesh = mesh_creator.export_mesh(
        model,
        file_name=mesh_path,
        export_flag=export_flag,
        **cfg.MeshCreator.export,
    )
    if export_flag:
        logger.info(f"Saved the mesh at {mesh_path=}")

    ### Evaluate mesh if ground truth is provided
    if gt_mesh is not None:
        mesh_evaluator = hydra.utils.instantiate(cfg.MeshEvaluator.init)
        metrics, debug_metrics_dict = mesh_evaluator.eval_mesh(mesh, gt_mesh)

        ### Log metrics
        for metric_name, value in metrics.items():
            tb_writer.scalar_summary(f"{tb_tag_metric_name}/{metric_name}", value, epoch)
            logger.info(f"{metric_name}: {value:.4f}")

            ### Save the metrics to the dataframe (row index is the file name, column is the metrics)
            if df is not None:
                df.loc[df.index[-1], metric_name] = value

        ### Log all metrics in Text
        ### tb_writer.add_ConfigDict(tag="All evaluation metrics", cfg_dict=debug_metrics_dict, step=epoch)

    ### Plot mesh multiple views
    if plot_mesh_views_flag:
        logger.info("Plotting the mesh...")
        fig = mesh_utils.plot_mesh_views(mesh, title=f"Mesh at epoch {epoch}", show=False)
        img = export_figure.fig_to_numpy(fig)
        tb_writer.image_summary("mesh", img, epoch)
        logger.info("Done plotting the mesh...")


def train(
    cfg: DictConfig,
    data_path: Path,
    save_path: Path,
    rng: pytorch_utils.SeedAll,
    logger: logging.Logger,
    gpu_device: torch.device | None = None,
    gt_path: Path | None = None,
    df: pd.DataFrame | None = None,
) -> None:
    logger.info(f"Training on a file: {data_path=}")

    ### Create tensorboard writer
    tb_writer = TensorBoard_utils.Logger(log_dir=save_path)

    ### Create dataset and data loader
    dataset = hydra.utils.instantiate(
        cfg.dataset.PointCloud,
        PointCloud_path=data_path,
        rng=rng.numpy_generator,
        to_gpu_device=gpu_device,
    )
    data_loader = hydra.utils.instantiate(
        cfg.data_loader, dataset=dataset, worker_init_fn=pytorch_utils.seed_worker, generator=rng.torch_generator
    )

    ### Create ground truth mesh if provided
    if gt_path is not None:
        logger.info(f"Loading ground truth mesh from {gt_path=}")
        gt_mesh = trimesh.load(gt_path, process=False)
    else:
        logger.info("No ground truth mesh provided...")
        gt_mesh = None

    ### Create model
    model = hydra.utils.instantiate(cfg.model[cfg.model.selected]).to(gpu_device)

    ### Create mesh creator for exporting mesh
    mesh_creator = hydra.utils.instantiate(cfg.MeshCreator.init, to_gpu_device=gpu_device)

    ### Create optimizer and loss function
    optim = hydra.utils.instantiate(cfg.optimizer[cfg.optimizer.selected], params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn[cfg.loss_fn.selected])

    ### ------------------------------------ Training loop ------------------------------------ ###
    logger.info("Start training...")
    min_loss = np.inf
    min_loss_epoch = 0
    with tqdm.tqdm(total=cfg.epochs, ncols=100) as pbar:
        for epoch in range(1, cfg.epochs + 1):  ### Range [1, cfg.epochs]
            temp_loss = 0.0
            model.train()

            for coords, occs in data_loader:
                optim.zero_grad()

                preds = model(coords)  ### preds.shape == occs.shape
                loss = loss_fn(preds, occs)  ### loss is scalar

                ### Calculate the loss for the current batch by averaging the loss over the mini-batch size
                ### coords.shape = (batch_size, mini_batch_size, 3)
                temp_loss += loss.item() * coords.shape[0]

                ### Backward pass
                loss.backward()

                ### Update the model parameters
                optim.step()

            ### Update the loss for the current epoch
            temp_loss /= len(data_loader.dataset)

            ### Update the loss on the progress bar
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch:>4}/{cfg.epochs:>4} | Loss: {temp_loss:>8.4f}")

            ### Save the loss for the current epoch
            if epoch % 10 == 0:
                tb_writer.scalar_summary("loss", temp_loss, epoch)

            ### Save the model if the loss is the minimum
            if temp_loss < min_loss:
                min_loss = temp_loss
                min_loss_epoch = epoch
                best_ckpt_path = save_path / "best_model.pt"
                torch.save(model.state_dict(), best_ckpt_path)
                # logger.info(f"Updated best checkpoint at {best_ckpt_path=}")

            ### ------------------- Infer in an interval ------------------- ###
            if cfg.infer_flag and (epoch % cfg.infer_freq == 0 or epoch == cfg.epochs):
                ### Save the model for each interval
                if cfg.save_infer_ckpt_flag:
                    ckpt_path = save_path / f"model_{epoch}.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Saved the model at {ckpt_path=}")

                ### Infer the model
                infer(
                    cfg,
                    model,
                    mesh_creator,
                    save_path,
                    epoch,
                    logger,
                    tb_writer,
                    gt_mesh,
                    plot_mesh_views_flag=cfg.plot_mesh_views_flag,
                    export_flag=cfg.export_flag,
                )

    logger.info("Done training...")

    ### ------------------- Infer for the best model ------------------- ###
    logger.info(f"Saved best checkpoint at {best_ckpt_path=}")
    logger.info(f"Infer for the best model at epoch {min_loss_epoch} with {min_loss=:.6f}")

    ### Save the best model metrics to the dataframe
    if df is not None:
        df.loc[df.index[-1], "best_model_epoch"] = min_loss_epoch
        df.loc[df.index[-1], "best_model_loss"] = min_loss

    ### Load the best model
    best_ckpt_path = save_path / "best_model.pt"
    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))

    ### Infer the best model
    infer(
        cfg,
        model,
        mesh_creator,
        save_path,
        min_loss_epoch,
        logger,
        tb_writer,
        gt_mesh,
        tb_tag_metric_name="best_model",
        plot_mesh_views_flag=True,
        export_flag=True,
        df=df,
    )

    logger.info(f"Done inferring for the best model at {min_loss_epoch=}")


@hydra.main(version_base=None, config_path="configs", config_name="train_mlp_3D")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)

    logger.info("Pre-process configuration")
    cfg = hydra_utils.preprocess_cfg(cfg, extra_folders=cfg.extra_folders, logger=logger, verbose=True)

    ### Create random generator collection
    rng = pytorch_utils.SeedAll(cfg.seed, logger=logger)

    ### Create GPU device
    gpu_device = torch.device("cuda")

    ### Add for-loop here to train all the Livers
    data_path = Path(cfg.data_path)
    current_path = Path(cfg.current_path)
    logger.info(f"Data path: {data_path=}")

    ### Create dataframe to save the results (Metrics, Loss, etc.)
    ### The row index is the file name, and the column is the metrics
    df = pd.DataFrame()

    ### Train mlp on a multiple files in a directory
    if data_path.is_dir():
        for file_path in data_path.iterdir():
            if file_path.is_file() and file_path.suffix == ".npy":
                ### Create ground truth path. The GT is in stl format.
                gt_path = file_path.parents[2] / "mesh" / file_path.name.replace(".npy", ".stl")

                ### Create a new row in the dataframe for the current file
                df = pd.concat([df, pd.DataFrame(index=[file_path.stem])])

                ### Train one model
                train(cfg, file_path, current_path / file_path.stem, rng, logger, gpu_device, gt_path=gt_path, df=df)

    ### Train mlp on a single file
    elif data_path.is_file():
        ### Create ground truth path. The GT is in stl format.
        gt_path = data_path.parents[2] / "mesh" / data_path.name.replace(".npy", ".stl")

        ### Create a new row in the dataframe for the current file
        df = pd.concat([df, pd.DataFrame(index=[data_path.stem])])

        ### Train one model
        train(cfg, data_path, current_path / data_path.stem, rng, logger, gpu_device, gt_path=gt_path, df=df)

    ### Save the dataframe to a csv file
    results_path = current_path / "results.csv"
    df.to_csv(results_path)
    logger.info(f"Saved the results to {results_path=}")


if __name__ == "__main__":
    main()

