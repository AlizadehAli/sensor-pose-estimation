"""
Calibration pipeline for point cloud alignment using Open3D.

This script reads point cloud files, applies transformations, performs registration,
and evaluates calibration quality based on fitness and RMSE scores.

Usage:
    python calibration_pipeline.py --config config.yml --data /path/to/calib_data

Configuration:
    Parameters are loaded from a YAML configuration file.

Dependencies:
    - Open3D
    - NumPy
    - YAML
    - argparse
"""

import os
import argparse
import open3d as o3d
import numpy as np
from utils import (
    load_config,
    euler_to_transformation_YPR,
    global_registration_with_initial,
    compute_calibration_score,
    extract_pose
)

def get_valid_file_indices(calib_datapath):
    """Retrieve valid file indices from the specified directory."""
    files_idx = set()
    for file in os.listdir(calib_datapath):
        if file.endswith(".ply"):
            idx = file.split(".")[0].split("_")[-1]
            if os.path.exists(os.path.join(calib_datapath, f'point_cloud_0_{idx}.ply')) and \
               os.path.exists(os.path.join(calib_datapath, f'point_cloud_1_{idx}.ply')):
                files_idx.add(int(idx))
    return sorted(files_idx)

def process_calibration(config_file, calib_datapath):
    """Main calibration processing loop."""
    config = load_config(config_file)
    
    calib_voxels = config['calibration']['voxels']
    calib_dist_thr = config['calibration']['distance_thresholds']
    visualization = config['calibration']['visualization']
    
    yaw, pitch, roll = np.deg2rad([
        config['transformation']['euler_angles']['yaw'],
        config['transformation']['euler_angles']['pitch'],
        config['transformation']['euler_angles']['roll']
    ])
    
    translation = [
        config['transformation']['translation']['tx'],
        config['transformation']['translation']['ty'],
        config['transformation']['translation']['tz']
    ]
    
    initial_transformation = euler_to_transformation_YPR(yaw, pitch, roll, translation)
    
    files_idx = get_valid_file_indices(calib_datapath)
    best_calib_score = None
    
    for voxel_size, distance_threshold in zip(calib_voxels, calib_dist_thr):
        for idx in files_idx:
            print(f"\nProcessing file: {idx} | Voxel size: {voxel_size} | Distance threshold: {distance_threshold}\n")
            
            source = o3d.io.read_point_cloud(os.path.join(calib_datapath, f'point_cloud_0_{idx}.ply'))
            target = o3d.io.read_point_cloud(os.path.join(calib_datapath, f'point_cloud_1_{idx}.ply'))
            
            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)
            
            source_down, _ = source_down.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
            target_down, _ = target_down.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
            
            final_T, _ = global_registration_with_initial(source_down, target_down, initial_transformation, voxel_size)
            
            if visualization:
                o3d.visualization.draw_geometries([source_down, target_down], window_name="Filtered Point Clouds")
            
            print("Running ICP-based registration...")
            result = o3d.pipelines.registration.registration_icp(
                target_down, source_down, distance_threshold, final_T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=999999, relative_fitness=1, relative_rmse=1e-6))
            
            calib_score = compute_calibration_score(result.fitness, result.inlier_rmse)
            
            if best_calib_score is None or calib_score >= best_calib_score:
                best_calib_score = calib_score
                best_idx = idx
                
                translation, _, euler_deg = extract_pose(result.transformation)
                
                with open(os.path.join(calib_datapath, "tr_rot.txt"), "w") as f:
                    f.write(f"[tx, ty, tz]: {translation}\n[yaw, pitch, roll]: {euler_deg}")
                
                np.savetxt(os.path.join(calib_datapath, "transformation_mtx.txt"), result.transformation, delimiter=",", fmt="%.16f")
                
                target_t = target.transform(result.transformation)
                merged = source + target_t
                
                if visualization:
                    o3d.visualization.draw_geometries([merged], window_name="Merged Point Cloud")
                
                o3d.io.write_point_cloud(os.path.join(calib_datapath, "Orbbec_merged.ply"), merged)
                initial_transformation = result.transformation
                
                print(f'Best calibration score: {best_calib_score:.4f} | Fitness: {result.fitness:.4f} | RMSE: {result.inlier_rmse:.4f} | Index: {best_idx}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Calibration Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--data", type=str, required=True, help="Path to the calibration data directory")
    args = parser.parse_args()
    
    process_calibration(args.config, args.data)
