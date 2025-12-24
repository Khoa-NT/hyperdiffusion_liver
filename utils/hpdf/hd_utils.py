"""
Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-12-24

Disclaimer: This file is taken and modified from many sources.
"""

from math import ceil
from pathlib import Path
import numpy as np
import pyrender
import torch
import trimesh

import tqdm

from .Pointnet_Pointnet2_pytorch.log.classification.pointnet2_ssg_wo_normals import \
    pointnet2_cls_ssg
from .torchmetrics_fid import FrechetInceptionDistance
from .torchmetrics_fid_3d import FrechetInceptionDistance3D


### --------------------------------------------------------- Code from HyperDiffusion --------------------------------------------------------- ###
# Using edited 2D-FID code of torch_metrics
fid = FrechetInceptionDistance(reset_real_features=True)

### Get the location of this file
Pointnet2_ckpt_path = Path(__file__).parent / "Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"

def calculate_fid_3d(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 10,
    path: str | Path = Pointnet2_ckpt_path,
):
    """Calculate 3D Fréchet Inception Distance (FID) between sample and reference point clouds using PointNet2 features.
    
    This function computes the FID score by extracting deep features from point clouds using a pre-trained 
    PointNet2 classification model, then calculating the Fréchet distance between the feature distributions 
    of sample and reference point clouds. The computation is performed in batches for memory efficiency.
    
    Args:
        sample_pcs (torch.Tensor): Generated/sample point clouds tensor of shape [N, num_points, 3]
        ref_pcs (torch.Tensor): Reference/real point clouds tensor of shape [N, num_points, 3]  
        batch_size (int): Number of point clouds to process in each batch. Defaults to 10
        path (str | Path): Path to the PointNet2 model checkpoint file. Defaults to Pointnet2_ckpt_path
        
    Returns:
        torch.Tensor: The computed 3D FID score (lower values indicate better similarity)
    """
    point_net = pointnet2_cls_ssg.get_model(40, normal_channel=False) ### (num_class, normal_channel)
    checkpoint = torch.load(path)
    point_net.load_state_dict(checkpoint["model_state_dict"])
    point_net.eval().to(sample_pcs.device)
    count = len(sample_pcs)
    for i in tqdm.tqdm(range(ceil(count / batch_size)), desc="Calculating FID", ncols=120):
        if i * batch_size >= count:
            break
        # print(
        #     ref_pcs[i * batch_size : (i + 1) * batch_size].shape,
        #     i * batch_size,
        #     (i + 1) * batch_size,
        # )

        ### Get the interm_repr from the PointNet2 model
        real_features = point_net(
            ref_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fake_features = point_net(
            sample_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fid.update(real_features, real=True, features=real_features)
        fid.update(fake_features, real=False, features=fake_features)

    ### Compute the FID score
    x = fid.compute()

    ### Reset the FID metric
    fid.reset()
    
    # print("x fid_value", x)
    return x




### ----------------------------------- Khoa: updated version of render_mesh ----------------------------------- ###
def render_meshes(meshes: list[trimesh.Trimesh], recreate_meshes: bool = False) -> list[np.ndarray]:
    out_imgs = []
    for mesh in meshes:
        ### Later: Add depth image if needed
        img, _ = render_mesh(mesh, recreate_mesh=recreate_meshes)
        out_imgs.append(img)

    return out_imgs

def render_mesh(obj: trimesh.Trimesh | np.ndarray, 
                recreate_mesh: bool = False,
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        obj (trimesh.Trimesh | np.ndarray): The mesh to render.
        recreate_mesh (bool): Whether to recreate a plain mesh from the vertices and faces. Defaults to False. Only used for input mesh rendering.

    Returns:
        tuple[np.ndarray, np.ndarray]: The rendered image and the depth image.
    """
    ### Check if obj is a trimesh.Trimesh or a numpy array
    if isinstance(obj, trimesh.Trimesh):
        # Handle mesh rendering
        if recreate_mesh:
            ### Recreate a plain mesh from the vertices and faces to avoid rendering issues. 
            ### For example, if the mesh already has vertex colors, the rendered image will be overlapped by the vertex colors + pyrender's material.
            obj = trimesh.Trimesh(vertices=obj.vertices, faces=obj.faces)

        mesh = pyrender.Mesh.from_trimesh(
            obj,
            material=pyrender.MetallicRoughnessMaterial(
                alphaMode="BLEND",
                baseColorFactor=[1, 0.3, 0.3, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8,
            ),
        )

    else:
        # Handle point cloud rendering, (converting it into a mesh instance)
        ### Ref: https://pyrender.readthedocs.io/en/latest/examples/models.html#point-spheres
        pts = obj
        sm = trimesh.creation.uv_sphere(radius=0.01)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0] ### assign red color to the sphere

        ### Create a transformation matrix for each point
        tfs = np.tile(np.eye(4), (len(pts), 1, 1))  ### shape (N_points, 4, 4)
        tfs[:, :3, 3] = pts
        mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    ### Create a scene
    scene = pyrender.Scene()
    scene.add(mesh)

    ### Create a camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    ### Create a light
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)

    ### Add the camera and light to the scene
    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    nl = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(nc)
    scene.add_node(nl)

    ### Create a renderer
    r = pyrender.OffscreenRenderer(800, 800)
    
    ### First camera pose
    camera_pose1 = look_at(elev=10, azim=70, roll=0, distance=1, target=np.array([0, 0, 0]), up=np.array([0, 0, 1]))
    scene.set_pose(nc, pose=camera_pose1)
    scene.set_pose(nl, pose=camera_pose1)
    color1, depth1 = r.render(scene)

    ### Second camera pose
    camera_pose2 = look_at(elev=50, azim=-150, roll=0, distance=1, target=np.array([0, 0, 0]), up=np.array([0, 0, 1]))
    scene.set_pose(nc, pose=camera_pose2)
    scene.set_pose(nl, pose=camera_pose2)
    color2, depth2 = r.render(scene)

    ### Stack the two images horizontally
    img = np.hstack((color1, color2))
    depth = np.hstack((depth1, depth2))

    r.delete()

    return img, depth


### Khoa: updated version of look_at with elevation, azimuth, roll, distance, target, up
def look_at(elev: float = 0, 
            azim: float = 0, 
            roll: float = 0, 
            distance: float = 1, 
            target: np.ndarray | list | tuple = np.array([0, 0, 0]), 
            up: np.ndarray | list | tuple = np.array([0, 1, 0])):
    """Compute camera-to-world transformation matrix for an orbit camera with an arbitrary up vector.
    
    Args:
        elev (float): Elevation angle in degrees, measured as the angle above the horizontal plane. Defaults to __0__.
        azim (float): Azimuth angle in degrees, measured in the horizontal plane relative to a reference direction. Defaults to __0__.
        roll (float): Roll angle in degrees applied about the camera's viewing axis. Defaults to __0__.
        distance (float): Distance from the target point. Defaults to __1__.
        target (np.ndarray | list | tuple): Target point coordinates in world space. Defaults to __np.array([0, 0, 0])__.
        up (np.ndarray | list | tuple): Up vector for the orbit. Defaults to __np.array([0, 1, 0])__.
    
    Returns:
        np.ndarray: 4x4 camera-to-world (pose) transformation matrix.
    """
    if not isinstance(up, np.ndarray):
        if isinstance(up, (list, tuple)) or hasattr(up, "__len__"):
            up = np.array(up)
    if not isinstance(target, np.ndarray):
        if isinstance(target, (list, tuple)) or hasattr(target, "__len__"):
            target = np.array(target)

    ###------ Normalize and choose horizontal reference ------###
    ### Normalize the provided up vector.
    up_norm = up / np.linalg.norm(up)
    
    ### Based on common conventions, choose a horizontal reference vector:
    ### • if up is [0, 1, 0] (Y up): use horizontal_ref = [0, 0, 1]
    ### • if up is [0, 0, 1] (Z up): use horizontal_ref = [1, 0, 0]
    ### • if up is [1, 0, 0] (X up): use horizontal_ref = [0, 1, 0]
    if np.allclose(up_norm, np.array([0, 1, 0])):
        horizontal_ref = np.array([0, 0, 1])
    elif np.allclose(up_norm, np.array([0, 0, 1])):
        horizontal_ref = np.array([1, 0, 0])
    elif np.allclose(up_norm, np.array([1, 0, 0])):
        horizontal_ref = np.array([0, 1, 0])
    else:
        ### For a general up vector, try to pick a reference that isn't colinear.
        tmp = np.cross(up_norm, np.array([0, 0, 1]))
        if np.linalg.norm(tmp) < 1e-8:
            horizontal_ref = np.array([0, 1, 0])
        else:
            horizontal_ref = tmp / np.linalg.norm(tmp)
    
    ###------ Convert and compute spherical offset ------###
    ### Convert angles from degrees to radians.
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    roll_rad = np.radians(roll)
    
    ### Compute horizontal offset in the plane perpendicular to up.
    ### Here the azimuth rotates the horizontal reference vector around up:
    ### offset_horizontal = cos(azim)*horizontal_ref + sin(azim)*cross(up, horizontal_ref)
    offset_horizontal = (np.cos(azim_rad) * horizontal_ref +
                         np.sin(azim_rad) * np.cross(up_norm, horizontal_ref))
    
    ### The full offset adds in the elevation (vertical) component along up.
    offset = distance * (np.cos(elev_rad) * offset_horizontal +
                         np.sin(elev_rad) * up_norm)
    
    ### Compute the camera (eye) position.
    eye = target + offset
    
    ###------ Build the camera frame ------###
    ### The camera will look toward the target.
    forward = target - eye
    norm_forward = np.linalg.norm(forward)
    if norm_forward < 1e-8:
        raise ValueError(f"forward={forward} has near zero norm.")
    forward = forward / norm_forward
    
    ### Compute the right vector (orthogonal to both forward and provided up).
    right = np.cross(forward, up_norm)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-8:
        right = np.array([1, 0, 0])
    else:
        right = right / norm_right
    
    ### Recompute the camera up vector to ensure orthogonality.
    cam_up = np.cross(right, forward)
    
    ### Apply roll rotation around the forward axis if needed.
    if np.abs(roll_rad) > 1e-8:
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)
        new_right = right * cos_roll - cam_up * sin_roll
        new_up = right * sin_roll + cam_up * cos_roll
        right, cam_up = new_right, new_up
    
    ###------ Construct the camera-to-world matrix ------###
    ### Here the first three columns are (right, up, -forward) and the translation is eye.
    cam2world = np.eye(4)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = cam_up
    cam2world[:3, 2] = -forward
    cam2world[:3, 3] = eye
    
    return cam2world


### Class to render fixed camera poses
class LiverRender:
    def __init__(self, 
                 n_cameras: int, 
                 camera_poses: list[dict], 
                 view_size: tuple[int, int] = (800, 800),
                 mesh_color: list[float] | list[list[float]] = [1, 0.3, 0.3, 1.0],
                 pc_radius: float = 0.01,
                 pc_color: list[float] | list[list[float]] = [1.0, 0.0, 0.0],
                 camera_yfov: float = np.pi / 3.0,
                 camera_aspect_ratio: float = 1.0,
                 bg_color: list[float] = [1.0, 1.0, 1.0, 1.0],
                 ):
        """
        Args:
            n_cameras (int): Number of cameras to render.
            camera_poses (list[dict]): List of camera poses.
            view_size (tuple[int, int]): Size (width, height) of the rendered image.
            mesh_color (list[float]): Color of the mesh. Set single color as default.
            pc_radius (float): Radius of the point cloud.
            pc_color (list[float]): Color of the point cloud.
            camera_yfov (float): Field of view of the camera.
            camera_aspect_ratio (float): Aspect ratio of the camera.
        """

        ### Assign the parameters
        self.n_cameras = n_cameras
        self.camera_poses = camera_poses
        self.view_size = view_size
        self.pc_radius = pc_radius
        self.camera_yfov = camera_yfov
        self.camera_aspect_ratio = camera_aspect_ratio
        self.bg_color = bg_color
        if self.bg_color[-1] == 0:
            self.is_transparent_bg = True
        else:
            self.is_transparent_bg = False

        ### Mesh color
        if isinstance(mesh_color, list): 
            if not isinstance(mesh_color[0], list):
                ### If not nested list, make it a nested list
                self.mesh_color: list[list[float]] = [mesh_color]
                self.is_single_mesh_color = True
            else:
                self.mesh_color: list[list[float]] = mesh_color
                self.is_single_mesh_color = False
        else:
            raise ValueError(f"mesh_color must be a list, got {type(mesh_color)}")

        ### Point cloud color
        if isinstance(pc_color, list):
            if not isinstance(pc_color[0], list):
                ### If not nested list, make it a nested list
                self.pc_color: list[list[float]] = [pc_color]
                self.is_single_pc_color = True
            else:
                self.pc_color: list[list[float]] = pc_color
                self.is_single_pc_color = False
        else:
            raise ValueError(f"pc_color must be a list, got {type(pc_color)}")


        ### Initialize the renderer
        self.reset()


    def reset(self):
        ### Create a scene
        self.scene = pyrender.Scene(bg_color=self.bg_color)

        ### Create a camera
        self.camera = pyrender.PerspectiveCamera(yfov=self.camera_yfov, aspectRatio=self.camera_aspect_ratio)

        ### Create a light
        self.light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)

        ### Add the camera and light to the scene
        self.nc = pyrender.Node(camera=self.camera, matrix=np.eye(4))
        self.nl = pyrender.Node(light=self.light, matrix=np.eye(4))
        self.scene.add_node(self.nc)
        self.scene.add_node(self.nl)

        ### Create camera poses
        self.camera_poses = self.create_camera_pose()
    

    def create_camera_pose(self):
        camera_poses = []

        ### Create camera poses
        for i in range(self.n_cameras):
            camera_pose = look_at(**self.camera_poses[i])
            camera_poses.append(camera_pose)

        return camera_poses


    def render_meshes(self, meshes: list[trimesh.Trimesh | np.ndarray], 
                      recreate_meshes: bool = False
                      ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            meshes (list[trimesh.Trimesh | np.ndarray]): The meshes to render.
            recreate_meshes (bool): Whether to recreate a plain mesh from the vertices and faces. Defaults to False. Only used for input mesh rendering.

        Returns:
            tuple[np.ndarray, np.ndarray]: The rendered images and the depth images.
        """
        imgs = []
        depths = []
        
        idx_mesh_color = 0
        idx_pc_color = 0
        
        for mesh in meshes:
            ### Process index of mesh color and point cloud color
            if isinstance(mesh, trimesh.Trimesh):
                idx = 0 if self.is_single_mesh_color else idx_mesh_color
                idx_mesh_color += 1
            else:
                idx = 0 if self.is_single_pc_color else idx_pc_color
                idx_pc_color += 1
            
            ### Render the mesh
            img, depth = self.render_mesh(mesh, idx=idx, recreate_mesh=recreate_meshes)

            ### Append the rendered image and depth image
            imgs.append(img)
            depths.append(depth)

        return imgs, depths
    
    
    def render_mesh(self, obj: trimesh.Trimesh | np.ndarray, 
                    idx: int = 0,
                    recreate_mesh: bool = False,
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            obj (trimesh.Trimesh | np.ndarray): The mesh/point cloud to render.
            idx (int): The index of the input mesh/point cloud.
            recreate_mesh (bool): Whether to recreate a plain mesh from the vertices and faces. Defaults to False. Only used for input mesh rendering.

        Returns:
        """
        ### Check if obj is a trimesh.Trimesh or a numpy array
        if isinstance(obj, trimesh.Trimesh):
            # Handle mesh rendering
            if recreate_mesh:
                ### Recreate a plain mesh from the vertices and faces to avoid rendering issues. 
                ### For example, if the mesh already has vertex colors, the rendered image will be overlapped by the vertex colors + pyrender's material.
                obj = trimesh.Trimesh(vertices=obj.vertices, faces=obj.faces)

            pyrender_mesh = pyrender.Mesh.from_trimesh(
                obj,
                material=pyrender.MetallicRoughnessMaterial(
                    alphaMode="BLEND",
                    baseColorFactor=self.mesh_color[idx],
                    metallicFactor=0.2,
                    roughnessFactor=0.8,
                ),
            )

        else:
            # Handle point cloud rendering, (converting it into a mesh instance)
            ### Ref: https://pyrender.readthedocs.io/en/latest/examples/models.html#point-spheres
            pts = obj
            # sm = trimesh.creation.uv_sphere(radius=0.01)
            sm = trimesh.creation.uv_sphere(radius=self.pc_radius)
            sm.visual.vertex_colors = self.pc_color[idx] ### assign red color to the sphere

            ### Create a transformation matrix for each point
            tfs = np.tile(np.eye(4), (len(pts), 1, 1))  ### shape (N_points, 4, 4)
            tfs[:, :3, 3] = pts
            pyrender_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        ### Add the mesh to the scene
        nm = pyrender.Node(mesh=pyrender_mesh, matrix=np.eye(4))
        self.scene.add_node(nm)

        ### Create a renderer
        r = pyrender.OffscreenRenderer(self.view_size[0], self.view_size[1])

        ### Render the scene from each camera pose
        imgs = []
        depths = []
        for camera_pose in self.camera_poses:
            ### Set the camera pose
            self.scene.set_pose(self.nc, pose=camera_pose)
            self.scene.set_pose(self.nl, pose=camera_pose)

            ### Render the scene
            if self.is_transparent_bg:
                img, depth = r.render(self.scene, flags=pyrender.RenderFlags.RGBA)
            else:
                img, depth = r.render(self.scene)
            imgs.append(img)
            depths.append(depth)

        ### Delete the renderer
        r.delete()

        ### Remove the mesh from the scene
        self.scene.remove_node(nm)

        if len(imgs) == 1:
            return imgs[0], depths[0]
        else:
            ### Stack the images horizontally
            img = np.hstack(imgs)
            depth = np.hstack(depths)

            return img, depth
    




