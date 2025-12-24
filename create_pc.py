"""
Create normalized point clouds from 3D liver meshes for MICCAI 2025.

The exported meshes are centered and scaled to `[-max_norm * weight, max_norm * weight]`
following the HyperDiffusion settings.

Sampled point clouds are saved in `npy` format with occupancies:
    + 1: inside
    + 0: outside

Sampled point cloud categories:
    + 3D_Reconstruction: random cube sample points + nearby surface points
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import igl
import numpy as np
import pandas as pd
import tqdm
import trimesh
from trimesh.sample import sample_surface

from utils.pathlib_utils import create_folder



### the igl 2.6 doesn' have fast_winding_number_for_meshes, we have to use fast_winding_number instead
### https://github.com/libigl/libigl/issues/2477
### Checking if the fast_winding_number_for_meshes is available
if hasattr(igl, "fast_winding_number_for_meshes"):
    fast_winding_number_for_meshes = igl.fast_winding_number_for_meshes
else:
    fast_winding_number_for_meshes = igl.fast_winding_number


class PointCloudCreator:
    """Generate point clouds from 3D mesh files."""

    def __init__(
        self,
        obj_suffix: str = "stl",
        random_seed: int = 42,
        max_norm: float = 0.5,
        decay_factor: float = 0.95,
        thresh: float = 0.5,
        nomalize_point_cloud: bool = True,
    ) -> None:
        """Initialize the creator.

        Args:
            obj_suffix (str): Mesh file extension. Defaults to __"stl"__.
            random_seed (int): Random seed for reproducibility. Defaults to __42__.
            max_norm (float): Range is ``[-max_norm, max_norm]``. Defaults to __0.5__.
            decay_factor (float): Optional shrink factor. Defaults to __0.95__.
            thresh (float): Winding number threshold. Defaults to __0.5__.
            nomalize_point_cloud (bool): Normalize vertices to the range. Defaults to __True__.
        """

        ### Settings
        self.obj_suffix = obj_suffix
        self.random_seed = random_seed
        self.max_norm = max_norm
        self.decay_factor = decay_factor
        self.thresh = thresh
        self.nomalize_point_cloud = nomalize_point_cloud

        ### Random generator
        self.rng = np.random.default_rng(seed=self.random_seed)

    def get_mesh(
        self,
        obj_path: Path,
        nomalize: bool = True,
        max_norm: float = 1.0,
        weight: float | None = 0.95,
    ) -> tuple[trimesh.Trimesh, np.ndarray, float] | trimesh.Trimesh:
        """Load and optionally normalize a mesh.

        Args:
            obj_path (Path): Path to mesh file.
            nomalize (bool): Normalize vertices to ``[-max_norm, max_norm]``. Defaults to __True__.
            max_norm (float): Target range magnitude. Defaults to __1.0__.
            weight (float | None): Additional scale factor. Defaults to __0.95__.

        Returns:
            tuple[trimesh.Trimesh, np.ndarray, float] | trimesh.Trimesh: Normalized mesh with offset and scale.
        """

        ### Read the mesh file
        obj: trimesh.Trimesh = trimesh.load(obj_path)

        if nomalize:
            ### Normalize the vertices to range [-max_norm, max_norm]
            vertices = obj.vertices

            ### Translate to origin
            offset = np.mean(vertices, axis=0, keepdims=True)  ### (1, 3)
            vertices -= offset

            ### Normalize to [-1, 1] symmetrically
            max_abs = np.max(np.abs(vertices))  ### (1,)
            vertices /= max_abs

            ### Scale to [-max_norm, max_norm]
            vertices *= max_norm

            ### Scale to [-max_norm * weight, max_norm * weight]
            if weight is not None:
                vertices *= weight

            ### Update the vertices
            obj.vertices = vertices

            return obj, offset, max_abs

        return obj

    def save_point_cloud(
        self,
        OBJs_path: Path,
        save_dir_path: Path,
        n_sampled_points: int,
        list_obj_names: list[str] | None = None,
    ) -> None:
        """Create point clouds from meshes and write them to disk.

        Args:
            OBJs_path (Path): Directory containing mesh files.
            save_dir_path (Path): Directory to save generated assets.
            n_sampled_points (int): Points sampled per mesh for each category.
            list_obj_names (list[str] | None): Mesh basenames to process. Defaults to __None__ for all meshes.
        """

        ### ------------------------- Create folders ------------------------- ###
        mesh_save_path = create_folder("mesh", dir_path=save_dir_path)
        npy_save_path = create_folder("npy", dir_path=save_dir_path)
        npy3DRec_save_path = create_folder("3D_Reconstruction", dir_path=npy_save_path)

        ### ------------------------- Write log ------------------------- ###
        df_log = pd.DataFrame(
            columns=[
                "name",
                "total_vertices",
                "offset",
                "scale",
                "coor_max_x",
                "coor_max_y",
                "coor_max_z",
                "coor_min_x",
                "coor_min_y",
                "coor_min_z",
            ]
        )

        log_path = save_dir_path / "log.txt"
        log_path.unlink(missing_ok=True)
        with open(log_path, "w") as log_file:
            log_file.write(f"Date: {datetime.datetime.now() }\n")

            ### Print the arguments in this function
            for arg_name, arg_value in locals().items():
                if arg_name not in ["self", "log_file"] and not arg_name.startswith("__"):
                    log_file.write(f"{arg_name}: {arg_value}\n")

            log_file.write("\n")
            log_file.write(f"### {'-'*30} Process Objects {'-'*30} ###\n")
            log_file.write("\n")

            ### ------------------------- Get data ------------------------- ###
            n_points_uniform = n_sampled_points
            n_points_surface = n_sampled_points

            ### Get all obj names if not specified
            if list_obj_names is None:
                list_obj_names = [obj_path.stem for obj_path in OBJs_path.glob(f"*.{self.obj_suffix}")]

            ### ------------------------- Process each object ------------------------- ###
            start_time = time.time()
            for obj_name in tqdm.tqdm(list_obj_names, ncols=100):
                start_obj_time = time.time()

                ### Get obj path
                obj_path = OBJs_path / f"{obj_name}.{self.obj_suffix}"

                ### Get normalized mesh in range [-max_norm, max_norm]
                obj, object_offset, object_scale = self.get_mesh(
                    obj_path, self.nomalize_point_cloud, self.max_norm, self.decay_factor
                )

                ### Write log
                vertices = obj.vertices
                coor_max = np.max(vertices, axis=0)  ### shape (3,) for x, y, z
                coor_min = np.min(vertices, axis=0)  ### shape (3,) for x, y, z
                log_file.write(f"# {obj_path.name}:\n\t{vertices.shape}\n")
                log_file.write(f"\tObject offset: {object_offset}, object scale: {object_scale}\n")
                log_file.write(f"\tCoordinates (x,y,z) max: {coor_max}, coordinates (x,y,z) min: {coor_min}\n")
                df_log.loc[len(df_log)] = {
                    "name": obj_name,
                    "total_vertices": vertices.shape[0],
                    "offset": object_offset,
                    "scale": object_scale,
                    "coor_max_x": coor_max[0],
                    "coor_max_y": coor_max[1],
                    "coor_max_z": coor_max[2],
                    "coor_min_x": coor_min[0],
                    "coor_min_y": coor_min[1],
                    "coor_min_z": coor_min[2],
                }

                ### ------------------------- Sample points ------------------------- ###
                points_uniform = self.rng.uniform(-self.max_norm, self.max_norm, size=(n_points_uniform, 3))  ### (n_points_uniform, 3)
                points_surface, _ = sample_surface(obj, n_points_surface, seed=self.random_seed)  ### (n_points_surface, 3)
                nearby_points_surface = points_surface + 0.01 * self.rng.standard_normal((n_points_surface, 3))  ### (n_points_surface, 3)

                ### Merge points = (2) + (1)
                points = np.concatenate([nearby_points_surface, points_uniform], axis=0)  ### (2*n_sampled_points, 3)
                log_file.write(f"\tSampled points: {points.shape}\n")
                log_file.write(
                    f"\tSampled points (x,y,z) max: {np.max(points, axis=0)}, Sampled points (x,y,z) min: {np.min(points, axis=0)}\n"
                )

                ### Calculate winding number
                inside_surface_values = fast_winding_number_for_meshes(obj.vertices, obj.faces, points)  ### (2*n_sampled_points,)
                log_file.write(
                    f"\tInside surface values: {inside_surface_values.shape}, max: {inside_surface_values.max()}, min: {inside_surface_values.min()}\n"
                )

                ### Create labels
                occupancies_winding = np.piecewise(
                    inside_surface_values,
                    [inside_surface_values < self.thresh, inside_surface_values >= self.thresh],
                    [0, 1],
                )
                occupancies = occupancies_winding[..., None]  ### (2*n_sampled_points, 1)
                log_file.write(
                    f"\tOccupancies: {occupancies.shape}, max: {occupancies.max()}, min: {occupancies.min()}\n"
                )

                ### ------------------------- Save point clouds and mesh ------------------------- ###
                sampled_pointcloud_and_occupancies = np.concatenate((points, occupancies), axis=-1)  ### (2*n_sampled_points, 4)
                log_file.write(f"\tPoint cloud for 3D Reconstruction: {sampled_pointcloud_and_occupancies.shape}\n")
                np.save(npy3DRec_save_path / f"{obj_path.stem}.npy", sampled_pointcloud_and_occupancies)

                ### Save the normalized mesh to data folder
                obj.export(mesh_save_path / obj_path.name)

                ### ------------------------- Running time ------------------------- ###
                process_time = time.time() - start_obj_time
                log_file.write(f"\tProcess time: {process_time:.2f} seconds\n")
                log_file.write("\n")

            ### ------------------------- End ------------------------- ###
            end_time = time.time()
            running_time = datetime.timedelta(seconds=end_time - start_time)
            log_file.write(
                f"### {'-'*30} Total {len(list_obj_names)} objects; time: {running_time} {'-'*30} ###\n"
            )

        ### Save excel log
        df_log.to_csv(save_dir_path / "log.csv", index=False)

    def get_cube_points_with_occupancies(self, obj: trimesh.Trimesh, log_file, N: int = 256) -> np.ndarray:
        """Create cube samples with occupancies.

        Args:
            obj (trimesh.Trimesh): The mesh object.
            log_file (TextIOWrapper): File handle for logging.
            N (int): Points per dimension. Defaults to __256__.

        Returns:
            np.ndarray: Cube points with occupancies of shape ``(N**3, 4)``.
        """

        points_range = np.linspace(-self.max_norm, self.max_norm, N)
        points = np.meshgrid(points_range, points_range, points_range, indexing="ij")
        points = np.stack(points, axis=-1).reshape(-1, 3)  ### (N^3, 3)

        inside_surface_values = fast_winding_number_for_meshes(obj.vertices, obj.faces, points)  ### (N^3,)
        log_file.write(
            f"\tInside surface values of cube points: {inside_surface_values.shape}, max: {inside_surface_values.max()}, min: {inside_surface_values.min()}\n"
        )

        occupancies_winding = np.piecewise(
            inside_surface_values,
            [inside_surface_values < self.thresh, inside_surface_values >= self.thresh],
            [0, 1],
        )
        occupancies = occupancies_winding[..., None]  ### (N^3, 1)
        log_file.write(
            f"\tOccupancies of cube points: {occupancies.shape}, max: {occupancies.max()}, min: {occupancies.min()}\n"
        )

        cube_points_with_occupancies = np.concatenate((points, occupancies), axis=-1)  ### (N^3, 4)
        return cube_points_with_occupancies


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sampling point clouds.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(description="Sample point clouds from meshes")
    parser.add_argument("--objs_path", type=Path, required=True, help="Directory containing meshes (stl/obj)")
    parser.add_argument("--save_root", type=Path, required=True, help="Root directory to store outputs")
    parser.add_argument("--obj_suffix", type=str, required=True, help="Mesh file extension to read")
    parser.add_argument(
        "--names_file",
        type=Path,
        default=None,
        help="Optional txt file with mesh names (without extension) to process (one per line)",
    )
    parser.add_argument(
        "--n_sampled_points",
        type=int,
        default=20_000,
        help="Number of points per category per mesh",
    )
    return parser.parse_args()


def main() -> None:
    """Run sampling based on CLI inputs."""

    args = parse_args()

    list_obj_names = None
    if args.names_file is not None:
        with open(args.names_file, "r") as f:
            ### Get the names of the meshes without extension to make it's consistent with the obj_suffix
            list_obj_names = [name.strip().split(".")[0] for name in f.read().splitlines() if name.strip()]

    save_dir_name = args.names_file.stem if args.names_file is not None else "all"
    save_dir_path = create_folder(
        f"sampled_{args.n_sampled_points}_{args.obj_suffix}", dir_path=args.save_root / save_dir_name
    )

    point_cloud_creator = PointCloudCreator(obj_suffix=args.obj_suffix)
    point_cloud_creator.save_point_cloud(
        args.objs_path,
        save_dir_path,
        args.n_sampled_points,
        list_obj_names=list_obj_names,
    )

    print(f"Point cloud sampling finished for {save_dir_path=}, {args.n_sampled_points=:,}")


if __name__ == "__main__":
    main()

