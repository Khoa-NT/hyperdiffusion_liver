"""
This file contains some useful functions for chamfer distance.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Disclaimer: This file is taken and modified from many sources.

References:
+ Occupancy Networks: https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/eval.py#L29
+ Shape as Points: https://github.com/autonomousvision/shape_as_points/blob/main/src/eval.py#L23
+ DeepSDF: https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/metrics/chamfer.py#L9
+ KDTree: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
+ KDTree.query: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
+ Point Cloud Utils: https://github.com/fwilliams/point-cloud-utils/blob/master/point_cloud_utils/__init__.py#L84
"""


### --------------------------------------------------------- Occupancy Chamfer Distance --------------------------------------------------------- ###
### After SciPy v1.6.0, cKDTree is functionally identical to KDTree. 
from scipy.spatial import KDTree
import numpy as np
import trimesh

def OccNet_CD(pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray, 
              pred_normals: np.ndarray = None, gt_normals: np.ndarray = None,
              Fscore_thresholds: np.ndarray = np.linspace(1./1000, 1, 1000)
              ) -> dict:
    """
    Compute the Chamfer Distance between two occupancy grids based on Occupancy Networks paper.
    """
    ### -------------------------- Completeness -------------------------- ###
    ### Completeness: how far are the points of the target point cloud from thre predicted point cloud
    ### Calculate KDTree(pred_pointcloud) then query(gt_pointcloud)
    ### For each point in `gt_pointcloud`, the KDTree finds the closest point in `pred_pointcloud`.
    completeness_raw, completeness_normals_raw = distance_p2p(
        points_src=gt_pointcloud, normals_src=gt_normals,
        points_tgt=pred_pointcloud, normals_tgt=pred_normals
    )
    completeness2 = completeness_raw**2

    completeness = completeness_raw.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals_raw.mean()


    ### -------------------------- Accuracy -------------------------- ###
    ### Accuracy: how far are the points of the predicted pred_pointcloud from the target pred_pointcloud
    ### Calculate KDTree(gt_pointcloud) then query(pred_pointcloud)
    ### For each point in `pred_pointcloud`, the KDTree finds the closest point in `gt_pointcloud`.
    accuracy_raw, accuracy_normals_raw = distance_p2p(
        points_src=pred_pointcloud, normals_src=pred_normals,
        points_tgt=gt_pointcloud, normals_tgt=gt_normals
    )
    accuracy2 = accuracy_raw**2

    accuracy = accuracy_raw.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals_raw.mean()


    ### -------------------------- Chamfer distance -------------------------- ###
    chamferL1 = 0.5 * (completeness + accuracy)
    chamferL2 = 0.5 * (completeness2 + accuracy2)

    ### -------------------------- Normals correctness -------------------------- ###
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals


    ### -------------------------- F-score -------------------------- ###
    recall = get_threshold_percentage(completeness_raw, Fscore_thresholds)
    precision = get_threshold_percentage(accuracy_raw, Fscore_thresholds)
    F_scores = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]


    ### Output dictionary
    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'chamfer-L1': chamferL1,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer-L2': chamferL2,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normal consistency': normals_correctness,
        'f-scores': F_scores,
    }

    return out_dict



def distance_p2p(points_src, normals_src, points_tgt, normals_tgt) -> tuple[np.ndarray, np.ndarray]:
    ''' Computes minimal distances of each point in points_src to points_tgt.

    TL;DR: For each point in `source`, the KDTree finds the closest point in `target`.

    Args:
        points_src (numpy array): source points. These are the "source" points for which you are querying the nearest neighbors from the KDTree.
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points. This is the set of points used to build the KDTree, representing the "target" points in the space.
        normals_tgt (numpy array): target normals

    Returns:
        dist (numpy array): For each point in points_src, the KDTree finds the closest point in points_tgt and calculates the distance between them. This distance is stored in the dist array.
        normals_dot_product (numpy array): The dot product between the normals of the source and target points.
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to method not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


### From Shape as Points
def get_threshold_percentage(dist: np.ndarray, thresholds: np.ndarray) -> list[float]:
    """Evaluates a point cloud by calculating percentage of points within thresholds.
    
    Args:
        dist (np.ndarray): Calculated distances between points.
        thresholds (np.ndarray): Threshold values for the F-score calculation. Defaults to np.linspace(1./1000, 1, 1000)
    
    Returns:
        list[float]: For each threshold t, returns mean percentage of points where dist <= t
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]

    return in_threshold

### --------------------------------------------------------- DeepSDF --------------------------------------------------------- ###
### The implementation of DeepSDF chamfer distance is not clear with the Occupancy Networks paper.
### DeepSDF says it normalizes by the number of points but the code does not seem to do that.
### https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/metrics/chamfer.py#L9

def DeepSDF_CD(pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray) -> float:
    # one direction
    gen_points_kd_tree = KDTree(pred_pointcloud)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_pointcloud)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_pointcloud)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_pointcloud)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


### --------------------------------------------------------- Point Cloud Utils --------------------------------------------------------- ###
### There are concerns about the point cloud utils chamfer distance:
### https://github.com/fwilliams/point-cloud-utils/issues/89

