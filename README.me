# Calibration Pipeline for Point Cloud Registration and Alignment of KinectV2 and Orbbec Femto Bolt

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Calibration Data Structure](#calibration-data-structure)
6. [Configuration File](#configuration-file-configyml)
7. [Pipeline Workflow](#pipeline-workflow)
8. [Example Output](#example-output)

## Overview
This project provides a **calibration pipeline** for aligning point clouds using **Open3D**. The script reads 3D point cloud files, applies transformations, performs **global registration**, and evaluates the calibration quality based on **fitness** and **RMSE (Root Mean Square Error)** scores.

## Features
- **Automatic Point Cloud Registration**: Uses **ICP (Iterative Closest Point)** and **RANSAC-based global registration**.
- **Transformation Handling**: Converts Euler angles into a **4x4 transformation matrix**.
- **Noise Filtering**: Removes outliers using statistical methods.
- **Configurable Parameters**: Uses a **YAML configuration file** for easy tuning.
- **Supports Command-line Arguments**: Allows dynamic selection of **configuration files** and **data directories**.

## Installation
### **Prerequisites**
Ensure you have installed the required dependencies:
```bash
pip install open3d numpy pyyaml argparse
```

## Usage
Run the calibration pipeline with:
```bash
python calibration_pipeline.py --config config.yml --data /path/to/calib_data
```

### **Command-line Arguments**
| Argument       | Description                              | Required |
|---------------|------------------------------------------|----------|
| `--config`    | Path to the YAML configuration file      | ✅        |
| `--data`      | Path to the calibration data directory   | ✅        |

## Calibration Data Structure
The input **point cloud data** should be structured as follows:
```
/path/to/calib_data/
│── point_cloud_0_1.ply
│── point_cloud_0_2.ply
│── point_cloud_0_3.ply
│── point_cloud_0_4.ply
│── point_cloud_1_1.ply
│── point_cloud_1_2.ply
│── point_cloud_1_3.ply
│── point_cloud_1_4.ply
```
Each file represents a **point cloud capture** from different sensors (`0` and `1`), indexed sequentially.

## Configuration File (`config.yml`)
The pipeline settings are defined in a **YAML configuration file**.

Example:
```yaml
calibration:
  voxels: [5, 1]
  distance_thresholds: [20, 10]
  visualization: false

dataset:
  path: "calibration_dataset"

transformation:
  euler_angles:
    yaw: -45.065
    pitch: -66.796
    roll: 30.832
  translation:
    x: 1662.90391
    y: -431.823753
    z: 1668.25336
```

## Pipeline Workflow
1. **Load Configuration**: Reads parameters from the **YAML file**.
2. **Preprocess Point Clouds**:
   - Loads `.ply` files.
   - Downsamples using voxel grid filtering.
   - Removes noise using statistical outlier removal.
3. **Apply Initial Transformation**:
   - Converts **Euler angles** into a transformation matrix.
   - Transforms the source point cloud accordingly.
4. **Global Registration**:
   - Uses **RANSAC** for rough alignment.
   - Runs **ICP** for fine alignment.
5. **Evaluate Calibration**:
   - Computes **fitness** and **RMSE** scores.
   - Saves the best transformation.
6. **Save Results**:
   - Stores the transformation matrix.
   - Saves the merged point cloud.

## Example Output
```bash
Processing file: 12 | Voxel size: 5 | Distance threshold: 20
Running RANSAC-based registration...
Running ICP-based registration...
Best calibration score: 0.9423 | Fitness: 0.9852 | RMSE: 0.0021 | Index: 12
```