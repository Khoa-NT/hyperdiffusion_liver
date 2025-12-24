"""
Lightweight dataset loaders used in the MICCAI 2025 release.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-12-24
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from trimesh.sample import sample_surface

from .weights_utils import state_dict_to_weights


class PointCloud(Dataset):
    """Load a point cloud saved as ``.npy`` with occupancies."""

    def __init__(
        self,
        PointCloud_path: Path | str,
        mini_batch_size: int,
        rng: np.random.Generator,
        getItem_mode: str = "random",
        to_gpu_device: torch.device | None = None,
    ):
        """Initialize a point cloud dataset.

        Args:
            PointCloud_path (Path | str): Path to the ``.npy`` file.
            mini_batch_size (int): Number of points to sample per iteration.
            rng (np.random.Generator): Random generator.
            getItem_mode (str): ``random`` | ``sequential`` | ``all``.
            to_gpu_device (torch.device | None): Optional device to preload tensors.
        """

        super().__init__()

        ### Load point cloud: shape (N_points, 4)
        point_cloud = np.load(PointCloud_path)
        if to_gpu_device is not None:
            point_cloud = torch.from_numpy(point_cloud).to(to_gpu_device).float()

        ### Split coordinates and occupancy values
        self.coords = point_cloud[:, :3]  ### (N, 3)
        self.occupancies = point_cloud[:, 3]  ### (N,)

        self.n_points = point_cloud.shape[0]
        self.mini_batch_size = mini_batch_size
        self.rng = rng
        self.getItem_mode = getItem_mode
        self.to_gpu_device = to_gpu_device


    def __len__(self) -> int:
        ### Length is based on the mini-batch size
        return self.n_points // self.mini_batch_size


    def __getitem__(self, idx: int):
        ### Select indices
        if self.getItem_mode == "random":
            idx = self.rng.choice(self.n_points, size=self.mini_batch_size, replace=False)

        elif self.getItem_mode == "sequential":
            if idx < 0:
                start = self.n_points + (idx * self.mini_batch_size)
                end = self.n_points + ((idx + 1) * self.mini_batch_size)
                idx = slice(start, end)
            else:
                idx = slice(idx * self.mini_batch_size, (idx + 1) * self.mini_batch_size)

        elif self.getItem_mode == "all":
            idx = slice(0, self.n_points)

        else:
            raise ValueError(f"getItem_mode {self.getItem_mode} not implemented")

        coords = self.coords[idx]  ### (batch_size, 3)
        occs = self.occupancies[idx, None]  ### (batch_size, 1)

        if self.to_gpu_device is not None:
            return coords, occs

        return torch.from_numpy(coords).float(), torch.from_numpy(occs).float()



class WeightDataset(Dataset):
    """Load INR weights (best_model.pt) per object based on a split file."""

    def __init__(self, split_path: Path | str, data_path: str, mesh_path: str):
        """Initialize a weight dataset.

        Args:
            split_path (Path | str): Path to the split file.
            data_path (str): Folder containing MLP checkpoints.
            mesh_path (str): Folder containing meshes for optional inspection.
        """

        super().__init__()

        with open(split_path, "r") as f:
            self.obj_names = [line.strip() for line in f.readlines()]

        self.data_path = Path(data_path)
        self.mesh_path = Path(mesh_path)


    def load_mesh(self, obj_name: str) -> trimesh.Trimesh:
        mesh_path = self.mesh_path / f"{obj_name}.stl"
        assert mesh_path.exists(), f"Mesh path {mesh_path} does not exist"
        return trimesh.load(mesh_path)


    def get_surface_points(
        self,
        num_points: int,
        seed: int = 0,
        max_idx: int | None = None,
    ) -> list[np.ndarray]:
        """Sample surface points from meshes for quick inspection."""

        surface_points = []
        for obj_name in self.obj_names[:max_idx]:
            mesh = self.load_mesh(obj_name)
            points, _ = sample_surface(mesh, num_points, seed=seed)  ### (num_points, 3)
            surface_points.append(points)

        return surface_points


    def __len__(self) -> int:
        return len(self.obj_names)


    def __getitem__(self, idx: int) -> torch.Tensor:
        obj_name = self.obj_names[idx]
        weight_path = self.data_path / obj_name / "best_model.pt"
        if not weight_path.exists():
            weight_path = self.data_path / obj_name / "best_model.pth"
        assert weight_path.exists(), f"Weight path {weight_path} does not exist"

        state_dict = torch.load(weight_path, weights_only=True, map_location="cpu")
        return state_dict_to_weights(state_dict)  ### (n_weights,)
