"""
Rigid Atlas to X-ray Registration Script

This script aligns a 3D atlas volume to an X-ray volume via anisotropic scaling,
3D rotation, and translation using paired landmark CSV files.

Usage examples:
  python register_atlas_rigid.py
  python register_atlas_rigid.py --load-params path/to/params.json
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import tifffile
from scipy.ndimage import affine_transform
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rigid atlas registration")
    parser.add_argument(
        "--atlas",
        default=os.path.join("C:\\Users\\hemesath\\python_experiments\\napari_ng\\Finch_atlas", "atlas.npy"),
        help="Path to atlas volume (.npy).",
    )
    parser.add_argument(
        "--points-atlas",
        default=os.path.join("C:\\Users\\hemesath\\python_experiments\\napari_ng\\Finch_atlas\\raw", "Points_atlas.csv"),
        help="CSV file containing atlas landmark coordinates.",
    )
    parser.add_argument(
        "--points-xray",
        default=os.path.join("C:\\Users\\hemesath\\python_experiments\\napari_ng\\Finch_atlas\\raw", "Points_xray.csv"),
        help="CSV file containing X-ray landmark coordinates.",
    )
    parser.add_argument(
        "--xray-shape",
        type=int,
        nargs=3,
        default=(1388, 2622, 3195),
        metavar=("Z", "Y", "X"),
        help="Target X-ray volume shape.",
    )
    parser.add_argument(
        "--save-dir",
        default="C:\\Users\\hemesath\\python_experiments\\napari_ng\\Finch_atlas",
        help="Directory for registered volumes and parameters.",
    )
    parser.add_argument(
        "--load-params",
        default=None,
        help="Optional JSON file with precomputed rigid parameters to skip optimization.",
    )
    parser.add_argument(
        "--save-params",
        default=None,
        help="Optional JSON path to save rigid parameters (defaults to save-dir).",
    )
    parser.add_argument(
        "--mip",
        type=int,
        default=3,
        help="Scale",
    )
    return parser.parse_args()


def load_points_from_csv(csv_path: str, mip: int) -> list:
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return [data[:, 0:3].astype(np.float64) * (1/(2**mip)), np.fliplr(data[:, 3:6].astype(np.float64))]


def apply_transformation(points: np.ndarray, params: List[float]) -> np.ndarray:
    scale = np.array(params[0:3], dtype=np.float64)
    rotation = Rotation.from_euler("xyz", params[3:6], degrees=True)
    translation = np.array(params[6:9], dtype=np.float64)
    scaled = points * scale
    rotated = rotation.apply(scaled)
    return rotated + translation


def objective_function(params, source_points, target_points):
    transformed = apply_transformation(source_points, params)
    return np.mean(np.sum((transformed - target_points) ** 2, axis=1))


def get_transformed_points(params, source_points):
    transformed = apply_transformation(source_points, params)
    return transformed


def build_transformed_csv(points, save_path):
    transformed = {'index': [i for i in range(points.shape[0])], 
                   'axis-0': [i for i in points[:,0]],
                   'axis-1': [i for i in points[:,1]],
                   'axis-2': [i for i in points[:,2]]}
    transformed_pd = pd.DataFrame(transformed)
    transformed_pd.to_csv(save_path, index=False)


def build_inverse_affine(params: List[float]):
    scale_matrix = np.diag(params[0:3])
    rotation_matrix = Rotation.from_euler("xyz", params[3:6], degrees=True).as_matrix()
    forward_matrix = scale_matrix @ rotation_matrix
    translation = np.array(params[6:9])
    inverse_matrix = np.linalg.inv(forward_matrix)
    inverse_translation = -inverse_matrix @ translation
    return inverse_matrix, inverse_translation


def load_precomputed_params(path: str) -> List[float]:
    with open(path, "r") as f:
        data = json.load(f)
    rigid = data.get("rigid_transformation")
    if rigid is None:
        raise ValueError(f"File {path} missing 'rigid_transformation' section.")
    scale = rigid["scale"]
    rotation = rigid["rotation_degrees"]
    translation = rigid["translation"]
    return [*scale, *rotation, *translation]


def save_params(path: str, params_dict: Dict):
    with open(path, "w") as f:
        json.dump(params_dict, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("RIGID ATLAS REGISTRATION")
    print("=" * 60)

    print("Loading atlas volume...")
    if args.atlas.endswith('.npy'):
        atlas = np.load(args.atlas)
        atlas_name = str(args.atlas).strip('.npy').split('\\')[-1]
    elif args.atlas.endswith('.tiff') or args.atlas.endswith('.tif'):
        atlas = tifffile.imread(args.atlas)
        atlas_name = str(args.atlas).strip('.tiff').strip('.tif').split('\\')[-1]
    elif args.atlas.endswith('.raw'):
        shape = [int(x) for x in args.atlas.split('_')[-1].strip('.raw').split('x')]
        atlas = np.fromfile(args.atlas, dtype=np.uint16).reshape(shape)
        atlas_name = str(args.atlas).strip('.raw').split('\\')[-1]
        atlas = np.transpose(atlas, (2, 1, 0))  # Ensure ZYX order
    print(f"Atlas shape: {atlas.shape}")
    print(f"Target X-ray shape: {tuple(args.xray_shape)}")
    
    print("\nLoading landmark correspondences from CSV...")
    atlas_points, xray_points = load_points_from_csv(args.points_atlas, args.mip)
    print(f"Loaded {len(atlas_points)} point pairs.")

    used_precomputed = False
    optimization_success = False
    rigid_error = None

    if args.load_params:
        print(f"\nLoading precomputed parameters from {args.load_params}...")
        optimal_params = load_precomputed_params(args.load_params)
        used_precomputed = True
        optimization_success = True
    else:
        print("\nOptimizing rigid parameters...")
        atlas_center = np.array(atlas.shape[:3]) / 2
        xray_center = np.array(args.xray_shape) / 2
        initial_scale = xray_center / atlas_center
        initial_trans = xray_center - atlas_center * initial_scale
        initial_params = [*initial_scale, 0.0, 0.0, 0.0, *initial_trans]
        print(f"Initial guess: {initial_params}")
        result = minimize(
            objective_function,
            initial_params,
            args=(atlas_points, xray_points),
            method="Powell",
            options={"maxiter": 10000, "disp": True},
        )
        optimal_params = result.x.tolist()
        optimization_success = bool(result.success)
        rigid_error = float(result.fun)
        print(f"Optimization success: {optimization_success}")
        print(f"Final error: {rigid_error:.4f}")

    print("\nExporting transformed points...")
    transformed_points = get_transformed_points(optimal_params, atlas_points)
    csv_path = os.path.join(args.save_dir, atlas_name + '_registered_rigid.csv')
    print('Exporting to', csv_path)
    build_transformed_csv(transformed_points, csv_path)


    print("\nBuilding affine transform...")
    inverse_matrix, inverse_translation = build_inverse_affine(optimal_params)

    print("Applying rigid transform to atlas volume...")
    transformed_atlas = np.zeros((*args.xray_shape, 3), dtype=np.uint8)
    for channel in tqdm(range(3), desc="Channels"):
        transformed_atlas[:, :, :, channel] = affine_transform(
            atlas[:, :, :, channel],
            inverse_matrix,
            offset=inverse_translation,
            output_shape=tuple(args.xray_shape),
            order=0,
            mode="constant",
            cval=0,
        )

    base_name = atlas_name + "_registered_rigid"
    output_npy = os.path.join(args.save_dir, f"{base_name}.npy")
    output_tiff = os.path.join(args.save_dir, f"{base_name}.tiff")
    params_path = args.save_params or os.path.join(args.save_dir, "transformation_params_rigid.json")

    print(f"\nSaving registered volume to {output_npy}")
    np.save(output_npy, transformed_atlas)
    print(f"Saving TIFF to {output_tiff}")
    tifffile.imwrite(output_tiff, transformed_atlas, imagej=True)

    params_dict = {
        "transformation_type": "rigid",
        "rigid_transformation": {
            "scale": [float(x) for x in optimal_params[0:3]],
            "rotation_degrees": [float(x) for x in optimal_params[3:6]],
            "translation": [float(x) for x in optimal_params[6:9]],
        },
        "used_precomputed_params": used_precomputed,
        "rigid_error": rigid_error,
        "optimization_success": optimization_success,
        "atlas_volume": args.atlas,
        "points_atlas_csv": args.points_atlas,
        "points_xray_csv": args.points_xray,
        "xray_shape": list(map(int, args.xray_shape)),
    }
    print(f"Saving parameters to {params_path}")
    save_params(params_path, params_dict)

    print("\nRegistration complete.")
    print("=" * 60)
    print(f"Outputs:\n  - {output_npy}\n  - {output_tiff}\n  - {params_path}")


if __name__ == "__main__":
    main()

