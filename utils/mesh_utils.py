"""
This file contains some useful functions for mesh processing.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Disclaimer: This file is taken and modified from many sources.
"""

from typing import Callable

from pathlib import Path

import numpy as np

import skimage
import trimesh
from trimesh.sample import sample_surface # https://trimesh.org/trimesh.sample.html#trimesh.sample.sample_surface

import torch
from torch import nn


from .occupancy_dual_contouring import occupancy_dual_contouring
from .chamfer_distance import OccNet_CD, DeepSDF_CD
from .volumetric_iou import check_mesh_contains, compute_iou, get_cube_winding_number



### Some colors
# [255,  76,  76, 255]: Red in HyperDiffusion
# [128, 174, 128, 255]: Green in 3DSlicer app



### --------------------------------------------------------- Inference --------------------------------------------------------- ###

class MeshCreator:
    def __init__(self, 
                 N: int = 256, 
                 linspace_min: tuple|int = -1, 
                 linspace_max: tuple|int = 1, 
                 voxel_origin: tuple|list|int = (-1, -1, -1), 
                 batch_size: int = 1024, 
                 to_gpu_device: torch.device|None = None,
                 mesh_format: str = 'ply',
                 vertex_color: list[int]|tuple[int]|None = (128, 174, 128, 255),
                 method: str = 'skimage',
                 ):
        """
        Args:
            N (int): Number of points in each dimension. 256 is a good number. Defaults to 256
            linspace_min (tuple|int): Minimum values for torch.linspace to create sampled points in range
            linspace_max (tuple|int): Maximum values for torch.linspace to create sampled points in range
            voxel_origin (tuple|int): Origin of the voxel grid. If input is an integer, it will be converted to a tuple with same value for each axis. Defaults to (-1, -1, -1)
            batch_size (int): Batch size or number of points to feed to the network. Defaults to 1024
            to_gpu_device (torch.device|None): Whether to move the point cloud to GPU. If None, the point cloud remains on CPU (default is None for GPU poor)
            mesh_format (str): Format to export the mesh. Defaults to 'ply'
            vertex_color (list[int]|tuple[int]|None): Color of the mesh vertices. Defaults to (128, 174, 128, 255)
            method (str): Method to extract the mesh {skimage; ocd}. Defaults to 'skimage'
        """
        self.N = N
        self.linspace_min = to_tuple(linspace_min)
        self.linspace_max = to_tuple(linspace_max)


        ### Note from SIREN: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        ### voxel_origin is the lowest values of each axis.
        voxel_origin = to_tuple(voxel_origin)
        self.voxel_origin = np.array(voxel_origin, dtype=np.float32).reshape(1, 3)

        ### Calculate size of each voxel to span for skimage marching cubes.
        ### For example, if the range is [-1, 1] in each dimension, then the size of each voxel is 2.0 / (N - 1).
        self.voxel_size = ((self.linspace_max[0] - self.linspace_min[0]) / (self.N - 1), 
                           (self.linspace_max[1] - self.linspace_min[1]) / (self.N - 1), 
                           (self.linspace_max[2] - self.linspace_min[2]) / (self.N - 1))

        self.batch_size = batch_size
        self.to_gpu_device = to_gpu_device
        self.mesh_format = mesh_format
        self.vertex_color = vertex_color

        ### Method to extract the mesh
        self.method = method
        if self.method == 'ocd':
            ### Create OCD object
            self.ocd = occupancy_dual_contouring(self.to_gpu_device if self.to_gpu_device is not None else torch.device('cpu'))
        elif self.method == 'skimage':
            ### Create 3D coordinates
            self.coords = self.create_3d_coords()
        else:
            raise ValueError(f"Invalid method: {self.method}")


    def create_3d_coords(self) -> torch.Tensor:
        x_coords = torch.linspace(self.linspace_min[0], self.linspace_max[0], self.N)
        y_coords = torch.linspace(self.linspace_min[1], self.linspace_max[1], self.N)
        z_coords = torch.linspace(self.linspace_min[2], self.linspace_max[2], self.N)

        coords = torch.cartesian_prod(x_coords, y_coords, z_coords) ### shape: (N^3, 3)
        coords.requires_grad = False
        if self.to_gpu_device is not None:
            coords = coords.to(self.to_gpu_device)

        return coords
    

    @torch.inference_mode()
    def predict_occ_cube(self, 
                         model: nn.Module, 
                         batch_size: int|None = None, 
                         device: torch.device|None = None) -> torch.Tensor:
        """
        Args:
            model (nn.Module): MLP model
            batch_size (int|None): Batch size or number of points to feed to the network. If None, use batch size from class.
            device (torch.device|None): Device to move the batch-sized points to. If None, the points remain on CPU and make sure the model is on CPU too.
                If provided, the points will be moved to the specified device if they have not already been moved by to_gpu_device.
                Only provide this parameter if you have limited GPU memory and want to move just the batch-sized points to GPU during inference.
        """
        model.eval()
        
        ### Use batch size from class if not provided
        batch_size = self.batch_size if batch_size is None else batch_size

        ### If both to_gpu_device and device are provided, we don't need to move the points to GPU twice.
        if self.to_gpu_device is not None and device is not None:
            ### If both to_gpu_device and device are provided, we don't need to move the points to GPU twice.
            ### But make sure the model is on the same device as the points.
            device = None
            

        ### Initialize OCC storage
        occ_values = torch.zeros(self.coords.shape[0])

        idx = 0
        while idx < self.coords.shape[0]:
            ### Get a slice of the coordinates
            idx_slicing = slice(idx, min(idx + batch_size, self.coords.shape[0]))
            sample_subset = self.coords[idx_slicing, ...]
            if device is not None:
                sample_subset = sample_subset.to(device)

            ### Feed the points to the network
            pred_occ_values = model(sample_subset)

            ### Update occupancy values
            occ_values[idx_slicing] = pred_occ_values.squeeze().detach().cpu()

            idx += batch_size

        ### Reshape to 3D cube and convert to numpy array
        occ_cube = occ_values.reshape(self.N, self.N, self.N).numpy()

        return occ_cube
    

    def export_mesh(self,
                    model: nn.Module | Callable, 
                    file_name: str|Path|None = None, 
                    level: float = 0.0,
                    return_occupancy: bool = False,
                    imp_func_cplx: int = 512,
                    export_flag: bool = True,
                    ) -> trimesh.Trimesh | None | tuple[trimesh.Trimesh, np.ndarray]| tuple[None, np.ndarray]:
        """
        Args:
            model (nn.Module | Callable): MLP model or a function that takes in a tensor of coordinates and returns a tensor of occupancy values.
            file_name (str|Path|None): Path including the file name to export the mesh. If None, the mesh will not be exported.
            level (float): Contour value. Defaults to 0.0 because we usually use the output logits directly as occupancy values. If use sigmoid, set level to 0.5
            return_occupancy (bool): Whether to return the occupancy values. Defaults to False
            imp_func_cplx (int): Complexity of the implicit function for method == 'ocd'. Defaults to 512
            export_flag (bool): Whether to export the mesh. Defaults to True. If the mesh is empty, the mesh will not be exported and the function will return None.

        Returns:
            trimesh.Trimesh | None: Extracted mesh. If the mesh is empty, return None.
            occ_cube (np.ndarray): Occupancy values. Only returned if return_occupancy is True
        """
        if self.method == 'skimage':
            occ_cube = self.predict_occ_cube(model)
            verts, faces, normals, values = skimage_marching_cubes(occ_cube, level=level, voxel_size=self.voxel_size)

            ### Translate the mesh to the origin because skimage marching cubes returns the vertices in the range [0, 1]
            verts = verts + self.voxel_origin

        elif self.method == 'ocd':
            imp_func = lambda xyz_coords: model(xyz_coords.to(torch.float32)).view(-1)
            temp_results = self.ocd.extract_mesh(imp_func, 
                                                 min_coord = self.linspace_min,
                                                 max_coord = self.linspace_max,
                                                 num_grid = self.N,
                                                 isolevel = level, 
                                                 batch_size = self.batch_size,
                                                 imp_func_cplx = imp_func_cplx,
                                                 outside = False, 
                                                 return_occupancy = return_occupancy,
                                                 )
            ### Post-process the results
            if return_occupancy:
                verts, faces, occ_cube = temp_results
                occ_cube = occ_cube.detach().clone().cpu().numpy()
            else:
                verts, faces = temp_results
            
            ### Move to CPU and convert to numpy
            verts = verts.detach().clone().cpu().numpy()
            faces = faces.detach().clone().cpu().numpy()
            
            ### OCD does not return normals
            normals = None


        ### Create the mesh
        if len(verts) > 0:
            mesh = trimesh.Trimesh(
                vertices=verts, 
                faces=faces, 
                vertex_normals=normals,
                process=False,
                vertex_colors=self.vertex_color,
            )

            ### Fix the normals if needed
            mesh.fix_normals()
        
        else:
            mesh = None

        ### Save the mesh if needed
        if file_name is not None and export_flag and mesh is not None:
            ### Check suffix of file_name and export accordingly
            if isinstance(file_name, str):
                file_name = Path(file_name)

            if file_name.suffix == "":
                file_name = file_name.with_suffix(f".{self.mesh_format.lower()}")

            else:
                mesh_format = file_name.suffix.replace(".", "")
                assert mesh_format == self.mesh_format, f"Mesh format {mesh_format} does not match the format {self.mesh_format}"

            ### Export the mesh
            mesh.export(file_name, file_type=self.mesh_format)


        ### Return the mesh and occupancy values if needed
        if return_occupancy:
            return mesh, occ_cube
        else:
            return mesh




### --------------------------------------------------------- Surface Reconstruction --------------------------------------------------------- ###
def skimage_marching_cubes(data_volume: np.ndarray, 
                   level: float = 0.0, 
                   voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a triangular mesh from a 3D volume using marching cubes algorithm.
    
    Args:
        data_volume (np.ndarray): 3D data volume to extract mesh from. Should be a binary volume or signed distance field.
        level (float): Contour value of skimage.measure.marching_cubes(). Defaults to 0.0.
        voxel_size (tuple[float, float, float]): Physical size of each voxel in the volume. Defaults to (1.0, 1.0, 1.0).
    
    Returns:
        tuple: (vertices, faces, normals, values) of the extracted mesh
    """
    ### Marching cubes
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(data_volume, level=level, spacing=voxel_size)
    
    ### If marching cubes fails, try without a level
    except ValueError as e:
        verts, faces, normals, values = skimage.measure.marching_cubes(data_volume, level=None, spacing=voxel_size)
    
    ### If all else fails, return an empty mesh
    except Exception as e:
        print(e)
        verts, faces, normals, values = (
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.zeros(0),
        )
    
    return verts, faces, normals, values







### --------------------------------------------------------- Metrics --------------------------------------------------------- ###
class MeshEvaluator:
    def __init__(self, 
                 N_pointcloud: int = 100_000, 
                 N_cube: int = 128, 
                 min_max_range: tuple[float, float] | list[float, float] = (-0.5, 0.5), 
                 winding_number_threshold: float = 0.5, 
                 hash_resolution: int = 512,
                 verbose: bool = False,
                 random_seed: int = 42,
                 Fscore_thresholds: np.ndarray = np.linspace(1./1000, 1, 1000),
                 ):
        """
        Args:
            N_pointcloud (int): Number of points to sample from the predicted mesh.
            N_cube (int): Number of points to sample from the ground truth mesh.
            min_max_range (tuple[float, float] | list[float, float]): Range of the cube.
            winding_number_threshold (float): Threshold for the winding number.
            hash_resolution (int): Resolution of the hash grid.
            verbose (bool): Whether to print verbose output.
            random_seed (int): Random seed for the sampling.
            Fscore_thresholds (np.ndarray): Thresholds for the F-score.


        How to use:
        mesh_evaluator = MeshEvaluator(
            N_pointcloud=100_000,
            N_cube=128,
            min_max_range=[-0.5, 0.5],
            winding_number_threshold=0.5,
            hash_resolution=512,
            verbose=False,
            random_seed=42,
        )
        metrics, debug_metrics_dict = mesh_evaluator.eval_mesh(mesh, gt_mesh)
        """

        self.N_pointcloud = N_pointcloud
        self.N_cube = N_cube
        self.min_max_range = min_max_range
        self.winding_number_threshold = winding_number_threshold
        self.hash_resolution = hash_resolution
        self.verbose = verbose
        self.random_seed = random_seed
        self.Fscore_thresholds = Fscore_thresholds

    def eval_mesh(self, pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh):
        ### Sample surface points from the predicted mesh. And get the normals of the points
        pred_points, pred_face_index = sample_surface(pred_mesh, self.N_pointcloud, seed=self.random_seed)
        pred_normals = pred_mesh.face_normals[pred_face_index]

        ### Sample surface points from the ground truth mesh. And get the normals of the points
        gt_points, gt_face_index = sample_surface(gt_mesh, self.N_pointcloud, seed=self.random_seed)
        gt_normals = gt_mesh.face_normals[gt_face_index]


        ### -------------------------- Evaluation -------------------------- ###
        ### Evaluate the pointcloud
        eval_dict, debug_dict = self.eval_pointcloud(pred_points, gt_points, pred_normals, gt_normals)

        ### Calculate the VIoU and Dice score
        viou, dice = self.calc_viou(pred_mesh, gt_mesh, get_dice=True)

        ### Add VIoU and Dice score to the evaluation dictionary
        ### Higher is better
        eval_dict['VIoU'] = viou 
        eval_dict['Dice'] = dice

        return eval_dict, debug_dict
        
    
    def eval_pointcloud(self,
                        pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray, 
                        pred_normals: np.ndarray, gt_normals: np.ndarray,
                        ):
        
        ### Calculate the Chamfer distance between the predicted pointcloud and the ground truth pointcloud
        CD_dict, all_dict = self.calc_chamfer_distance(pred_pointcloud, gt_pointcloud, pred_normals, gt_normals)

        ### Combine the Chamfer distance dictionary
        eval_dict = {**CD_dict}

        return eval_dict, all_dict



    def calc_chamfer_distance(self, 
                              pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray, 
                              pred_normals: np.ndarray, gt_normals: np.ndarray,
                              ) -> tuple[dict, dict]:
        ### Calculate the Chamfer distance between the predicted pointcloud and the ground truth pointcloud
        occ_cd = OccNet_CD(pred_pointcloud, gt_pointcloud, pred_normals, gt_normals)
        deepSDF_cd = DeepSDF_CD(pred_pointcloud, gt_pointcloud)

        ### Chamfer distance dictionary
        CD_dict = {
            'OccNet chamfer-L1': occ_cd['chamfer-L1'], ### Lower is better
            'OccNet chamfer-L2': occ_cd['chamfer-L2'], ### Lower is better
            'OccNet normal consistency': occ_cd['normal consistency'],  ### Higher is better
            'OccNet f-scores': occ_cd['f-scores'][9], ### threshold = 1.0% Higher is better
            'DeepSDF chamfer': deepSDF_cd, ### Lower is better
        }

        ### Put all other information in one dictionary
        all_dict = {**occ_cd, 'DeepSDF chamfer':deepSDF_cd}

        return CD_dict, all_dict


    def calc_viou(self, pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, get_dice: bool = True) -> tuple[float, float]:
        cube_points, gt_occ = get_cube_winding_number(gt_mesh, N=self.N_cube, min_max_norm=self.min_max_range, 
                                                      winding_number_threshold=self.winding_number_threshold, verbose=self.verbose)
        pred_occ = check_mesh_contains(pred_mesh, cube_points, hash_resolution=self.hash_resolution)
        viou, dice = compute_iou(pred_occ, gt_occ, get_dice=get_dice)

        return viou, dice


### --------------------------------------------------------- Visualization --------------------------------------------------------- ###

def plot_mesh_views(mesh: trimesh.Trimesh, title: str = "Mesh from different views", show: bool = True):
    """Plot 6 views of a 3D mesh from different angles
    Args:
        mesh (trimesh.Trimesh): The input mesh to visualize
        title (str): Title of the plot
        show (bool): Whether to show the plot. Defaults to True
    """
    ### Set backend to Agg to turn off X11 or XQuartz
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
    fig.suptitle(title)
    axes = axes.flatten()

    ### Define 6 views from different angles (elevation, azimuth)
    views = [
        (0, 0),    ### Front view
        (0, 90),   ### Right view 
        (0, 180),  ### Back view
        (0, 270),  ### Left view
        (90, 0),   ### Top view
        (-90, 0)   ### Bottom view
    ]

    for i, (ax, (elev, azim)) in enumerate(zip(axes, views)):
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'View {i}')

    fig.tight_layout()

    if show:
        plt.show()

    return fig



### --------------------------------------------------------- Helper functions --------------------------------------------------------- ###
def to_tuple(value: int|tuple|list) -> tuple:
    if isinstance(value, (int, float)):
        return (value,)*3
    
    elif hasattr(value, '__len__') and not isinstance(value, str):
        assert len(value) == 3, f"Invalid input length: {len(value)}"
        return tuple(value)
    
    ### Sanity check
    assert isinstance(value, tuple) and len(value) == 3, f"Invalid input type: {type(value)} != tuple or length: {len(value)} != 3"
    return value

