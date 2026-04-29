#!/usr/bin/env python3
"""
Rotate points using the *rotation-only* part of an elastix 3D AffineTransform.

Elastix/ITK affine (with center c):
    y = A @ (x - c) + c + t

This script:
  1) parses A and center c from the elastix parameter file,
  2) extracts the closest proper rotation R to A (polar decomposition via SVD),
  3) applies rotation about c:
        MOVING->FIXED: y = R   @ (x - c) + c
        FIXED->MOVING: y = R.T @ (x - c) + c   (inverse for pure rotation)

By default, assumes the elastix transform maps MOVING -> FIXED.
"""

import re
import sys
import argparse
import numpy as np
import json


def read_elastix_params(txt_path: str):
    with open(txt_path, "r") as f:
        txt = json.load(f)

    c = np.array(txt["center_of_rotation"], dtype=float)
    p = np.array(txt["transform_parameters"], dtype=float)

    A = p[:9].reshape(3, 3)  # row-major in elastix parameter files
    t = p[9:]                # translation (ignored here)
    return A, t, c


def closest_rotation(A: np.ndarray) -> np.ndarray:
    """Closest proper rotation matrix to A (SVD / polar decomposition)."""
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # fix reflection
        U[:, -1] *= -1
        R = U @ Vt
    return R


def apply_rotation(points_xyz, R, center):
    """Apply y = R @ (x-center) + center to one point (3,) or array (N,3)."""
    P = np.asarray(points_xyz, dtype=float)
    if P.ndim == 1:
        return (R @ (P - center)) + center
    # row-wise: (R @ v).T == v @ R.T
    return ((P - center) @ R.T) + center


def parse_points_from_cli_or_stdin(point_args):
    # If provided as x y z [x y z ...]
    if point_args is not None and len(point_args) > 0:
        coords = list(map(float, point_args))
        if len(coords) % 3 != 0:
            raise ValueError("Point coordinates must be multiples of 3 (x y z)")
        return np.array(coords, dtype=float).reshape(-1, 3)

    # Otherwise read from stdin: one "x y z" per line
    pts = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        arr = np.fromstring(line, sep=" ", dtype=float)
        if arr.size != 3:
            raise ValueError("Each input line must contain exactly 3 numbers: x y z")
        pts.append(arr)
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.vstack(pts)


def main():
    ap = argparse.ArgumentParser(
        description="Apply elastix affine rotation-only (no scaling/shear, no translation) to 3D points."
    )
    ap.add_argument("param_file", default="zf13/transform_parameters.json", help="elastix TransformParameters.json file")
    ap.add_argument(
        "--direction",
        choices=["moving2fixed", "fixed2moving"],
        default="moving2fixed",
        help="Transform direction (default: moving2fixed).",
    )
    ap.add_argument(
        "points",
        nargs="*",
        help="Optional point coordinates: x y z [x y z ...]. If omitted, read points from stdin.",
    )
    args = ap.parse_args()

    A, t, c = read_elastix_params(args.param_file)
    R = closest_rotation(A)

    # Choose forward/inverse rotation
    R_use = R if args.direction == "moving2fixed" else R.T

    pts = parse_points_from_cli_or_stdin(args.points)
    if pts.size == 0:
        return

    out = apply_rotation(pts, R_use, c)

    # Print mapping
    for p_in, p_out in zip(pts, out):
        print(
            f"{p_in[0]:.8f} {p_in[1]:.8f} {p_in[2]:.8f}"
            f"  ->  {p_out[0]:.8f} {p_out[1]:.8f} {p_out[2]:.8f}"
        )


if __name__ == "__main__":
    main()