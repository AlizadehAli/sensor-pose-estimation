"""
This module provides utility functions for point cloud processing, registration,
and calibration scoring using Open3D.

Functions:
    - load_config: Loads configuration parameters from a YAML file.
    - euler_to_transformation_YPR: Converts Euler angles and translation to a 4x4 transformation matrix.
    - compute_fpfh: Computes FPFH features for point clouds.
    - global_registration_with_initial: Performs global registration with an initial transformation.
    - extract_pose: Extracts translation and Euler angles from a transformation matrix.
    - compute_calibration_score: Computes a combined fitness-RMSE calibration score.
"""

import open3d as o3d
import numpy as np
import copy
import yaml

def load_config(yaml_path):
    """
    Load configuration parameters from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def euler_to_transformation_YPR(yaw, pitch, roll, translation):
    """
    Compute a 4x4 homogeneous transformation matrix from Euler angles and translation.

    Args:
        yaw (float): Rotation around the Z-axis (radians).
        pitch (float): Rotation around the Y-axis (radians).
        roll (float): Rotation around the X-axis (radians).
        translation (list of float): Translation vector [tx, ty, tz].

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])
    
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [ 0,             1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rx = np.array([[1, 0,            0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(translation)
    return T

def compute_fpfh(pcd, voxel_size):
    """
    Downsample a point cloud, estimate normals, and compute FPFH features.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Voxel size for downsampling and feature extraction.

    Returns:
        tuple: Downsampled point cloud and computed FPFH features.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, fpfh

def global_registration_with_initial(source, target, initial_transformation, voxel_size=0.05):
    """
    Perform global registration with an initial transformation.

    Args:
        source (o3d.geometry.PointCloud): Source point cloud.
        target (o3d.geometry.PointCloud): Target point cloud.
        initial_transformation (np.ndarray): 4x4 initial transformation matrix.
        voxel_size (float): Voxel size for feature computation.

    Returns:
        tuple: Final transformation matrix and RANSAC registration result.
    """
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(initial_transformation)
    
    source_down, source_fpfh = compute_fpfh(source_transformed, voxel_size)
    target_down, target_fpfh = compute_fpfh(target, voxel_size)
    
    distance_threshold = voxel_size * 1.5
    print("Running RANSAC-based registration...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return np.dot(result.transformation, initial_transformation), result

def extract_pose(T):
    """
    Extract translation vector and Euler angles from a transformation matrix.

    Args:
        T (np.ndarray): 4x4 transformation matrix.

    Returns:
        tuple: Translation vector, Euler angles in radians, Euler angles in degrees.
    """
    t = T[:3, 3]
    R = T[:3, :3]
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return t, (yaw, pitch, roll), np.degrees([yaw, pitch, roll])

def compute_calibration_score(fitness, rmse, alpha=1.0, beta=1.0):
    """
    Compute a calibration score based on fitness and RMSE.

    Args:
        fitness (float): Fitness score (higher is better).
        rmse (float): Root Mean Square Error (lower is better).
        alpha (float): Weight for fitness contribution.
        beta (float): Weight for RMSE penalty.

    Returns:
        float: Computed calibration score.
    """
    return (alpha * fitness - beta * rmse) / (1 + rmse)