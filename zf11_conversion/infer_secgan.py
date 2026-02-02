from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from cloudvolume import CloudVolume

from secgan_models import Generator3D


# Generator with VALID convs:
# in_conv (k=3 valid) shrinks by 2
# each resblock = 2 valid convs => shrinks by 4; 8 blocks => 32
# total shrink = 34 voxels per axis
SHRINK = 34
MARGIN = SHRINK // 2  # 17 voxels each side


def parse_xyz(s: str) -> Tuple[int, int, int]:
    a = tuple(int(x) for x in s.split(","))
    if len(a) != 3:
        raise ValueError("Expected 'x,y,z'")
    return a  # xyz


def parse_bbox(s: Optional[str]):
    # "x0,x1,y0,y1,z0,z1"
    if s is None:
        return None
    p = [int(x) for x in s.split(",")]
    if len(p) != 6:
        raise ValueError("bbox must be x0,x1,y0,y1,z0,z1")
    x0, x1, y0, y1, z0, z1 = p
    return (x0, x1, y0, y1, z0, z1)


def cv_bounds(vol: CloudVolume) -> Tuple[int, int, int, int, int, int]:
    if hasattr(vol, "bounds") and hasattr(vol.bounds, "minpt") and hasattr(vol.bounds, "maxpt"):
        mn, mx = vol.bounds.minpt, vol.bounds.maxpt
        return int(mn.x), int(mx.x), int(mn.y), int(mx.y), int(mn.z), int(mx.z)
    sx, sy, sz = vol.shape[:3]
    return 0, int(sx), 0, int(sy), 0, int(sz)


def ensure_out_volume(
    out_url: str,
    ref_vol: CloudVolume,
    mip: int,
    dtype: str,
    layer_type: str = "image",
    encoding: str = "raw",
    num_channels: int = 1,
):
    """
    Create an output precomputed volume if it doesn't exist.
    If it exists, just open it.
    """
    try:
        out_vol = CloudVolume(out_url, mip=mip, progress=False, fill_missing=True)
        # will error if info missing
        _ = out_vol.info
        return out_vol
    except Exception:
        pass

    # create info based on reference
    ref_info = ref_vol.info
    scale = ref_info["scales"][mip]
    resolution = scale["resolution"]
    voxel_offset = scale.get("voxel_offset", [0, 0, 0])
    volume_size = scale["size"]
    chunk_size = scale["chunk_sizes"][0]

    info = CloudVolume.create_new_info(
        num_channels=num_channels,
        layer_type=layer_type,
        data_type=dtype,
        encoding=encoding,
        resolution=resolution,
        voxel_offset=voxel_offset,
        chunk_size=chunk_size,
        volume_size=volume_size,
    )
    CloudVolume(out_url, info=info, mip=mip, progress=False).commit_info()
    out_vol = CloudVolume(out_url, mip=mip, progress=False, fill_missing=True)
    out_vol.commit_provenance()
    return out_vol


def load_segmenter(module_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    import importlib

    m = importlib.import_module(module_path)
    S: nn.Module = m.build_segmenter()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        S.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        S.load_state_dict(ckpt, strict=False)
    S.to(device)
    S.eval()
    for p in S.parameters():
        p.requires_grad_(False)
    return S


def load_generator_yx(secgan_ckpt: str, device: torch.device) -> nn.Module:
    G_YX = Generator3D(in_ch=1, base_ch=32, num_blocks=8).to(device)
    ckpt = torch.load(secgan_ckpt, map_location="cpu")
    if "G_YX" not in ckpt:
        raise ValueError("Checkpoint does not contain key 'G_YX'.")
    G_YX.load_state_dict(ckpt["G_YX"], strict=True)
    G_YX.eval()
    for p in G_YX.parameters():
        p.requires_grad_(False)
    return G_YX


def read_block_u16_with_pad(
    vol: CloudVolume,
    x0: int, x1: int, y0: int, y1: int, z0: int, z1: int,
    bounds: Tuple[int, int, int, int, int, int],
    pad_mode: str = "reflect",
):
    """
    Reads an XYZ block, clipping to volume bounds, and pads back to requested shape.
    Returns numpy uint16 array shaped (X,Y,Z).
    """
    bx0, bx1, by0, by1, bz0, bz1 = bounds

    rx0, rx1 = max(x0, bx0), min(x1, bx1)
    ry0, ry1 = max(y0, by0), min(y1, by1)
    rz0, rz1 = max(z0, bz0), min(z1, bz1)

    blk = vol[rx0:rx1, ry0:ry1, rz0:rz1]
    blk = np.asarray(blk)
    if blk.ndim == 4 and blk.shape[-1] == 1:
        blk = blk[..., 0]
    blk = blk.astype(np.uint16, copy=False)

    # pad amounts in xyz
    pad_left_x = rx0 - x0
    pad_right_x = x1 - rx1
    pad_left_y = ry0 - y0
    pad_right_y = y1 - ry1
    pad_left_z = rz0 - z0
    pad_right_z = z1 - rz1

    if any(p != 0 for p in [pad_left_x, pad_right_x, pad_left_y, pad_right_y, pad_left_z, pad_right_z]):
        # numpy pad uses ((before, after), ...) in array axis order (X,Y,Z)
        pad_width = ((pad_left_x, pad_right_x), (pad_left_y, pad_right_y), (pad_left_z, pad_right_z))
        if pad_mode == "reflect":
            blk = np.pad(blk, pad_width, mode="reflect")
        elif pad_mode == "edge":
            blk = np.pad(blk, pad_width, mode="edge")
        else:
            blk = np.pad(blk, pad_width, mode="constant", constant_values=0)

    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_y", required=True, help="CloudVolume URL for domain Y (uint16)")
    ap.add_argument("--mip", type=int, default=0)
    ap.add_argument("--bbox_y", default=None, help="x0,x1,y0,y1,z0,z1 (optional). Default: full bounds.")
    ap.add_argument("--tile_out", default="128,128,64", help="Output tile size in XYZ (recommended z smaller if anisotropic)")
    ap.add_argument("--pad_mode", default="reflect", choices=["reflect", "edge", "constant"])

    ap.add_argument("--secgan_ckpt", required=True, help="Checkpoint produced by training script (contains G_YX)")
    ap.add_argument("--segmenter_module", required=True, help="Python module path providing build_segmenter()")
    ap.add_argument("--segmenter_ckpt", required=True, help="Pretrained segmenter checkpoint (logits output)")

    ap.add_argument("--out_translated", default=None, help="Output CloudVolume URL for translated Y->X image (uint16)")
    ap.add_argument("--out_logits", default=None, help="Output CloudVolume URL for segmentation logits (float32)")

    ap.add_argument("--init_outputs", action="store_true", help="Create output CloudVolumes if missing.")
    ap.add_argument("--amp", action="store_true", help="Use autocast mixed precision on GPU.")
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tile_out_xyz = parse_xyz(args.tile_out)
    tile_in_xyz = (tile_out_xyz[0] + SHRINK, tile_out_xyz[1] + SHRINK, tile_out_xyz[2] + SHRINK)

    in_vol = CloudVolume(args.vol_y, progress=True, mip=args.mip)
    bounds = cv_bounds(in_vol)

    bbox = parse_bbox(args.bbox_y)
    if bbox is None:
        x0, x1, y0, y1, z0, z1 = bounds
    else:
        x0, x1, y0, y1, z0, z1 = bbox

    # Outputs
    out_img = None
    out_logits = None
    if args.out_translated:
        if args.init_outputs:
            out_img = ensure_out_volume(args.out_translated, in_vol, args.mip, dtype="uint16")
        else:
            out_img = CloudVolume(args.out_translated, progress=True, mip=args.mip, fill_missing=True)

    if args.out_logits:
        if args.init_outputs:
            out_logits = ensure_out_volume(args.out_logits, in_vol, args.mip, dtype="float32")
        else:
            out_logits = CloudVolume(args.out_logits, progress=True, mip=args.mip, fill_missing=True)

    # Models
    G_YX = load_generator_yx(args.secgan_ckpt, device)
    S = load_segmenter(args.segmenter_module, args.segmenter_ckpt, device)

    # Iterate tiles in XYZ over bbox
    ox_step, oy_step, oz_step = tile_out_xyz

    torch.set_grad_enabled(False)

    for oz in range(z0, z1, oz_step):
        for oy in range(y0, y1, oy_step):
            for ox in range(x0, x1, ox_step):
                # actual output region (clip at bbox end)
                oxe = min(ox + ox_step, x1)
                oye = min(oy + oy_step, y1)
                oze = min(oz + oz_step, z1)

                out_shape_xyz = (oxe - ox, oye - oy, oze - oz)

                # Corresponding input region with margin; keep output centered
                ix0 = ox - MARGIN
                iy0 = oy - MARGIN
                iz0 = oz - MARGIN
                ix1 = ix0 + (out_shape_xyz[0] + SHRINK)
                iy1 = iy0 + (out_shape_xyz[1] + SHRINK)
                iz1 = iz0 + (out_shape_xyz[2] + SHRINK)

                blk_u16_xyz = read_block_u16_with_pad(
                    in_vol, ix0, ix1, iy0, iy1, iz0, iz1, bounds=bounds, pad_mode=args.pad_mode
                )  # (X,Y,Z) uint16

                # Convert to torch (N,1,D,H,W) with D=Z, H=Y, W=X
                blk_u16_zyx = np.transpose(blk_u16_xyz, (2, 1, 0))  # (Z,Y,X)
                y_u16 = torch.from_numpy(blk_u16_zyx).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # Segmenter expects [0,1] float32 (your setup)
                y01 = y_u16 / 65535.0

                # Generator expects [-1,1]
                y_gan = y01 * 2.0 - 1.0

                with autocast(enabled=(args.amp and device.type == "cuda")):
                    fake_x = G_YX(y_gan)  # [-1,1], shape should be (1,1, outZ,outY,outX)
                    fake_x01 = ((fake_x + 1.0) * 0.5).clamp(0.0, 1.0)

                    # Segmenter outputs logits
                    logits = S(fake_x01)  # (1,1,D,H,W)

                # Move to CPU numpy and transpose back to XYZ for writing
                fake_x01_np_zyx = fake_x01.squeeze(0).squeeze(0).float().cpu().numpy()  # (Z,Y,X)
                fake_x_u16_xyz = np.transpose((fake_x01_np_zyx * 65535.0).round().astype(np.uint16), (2, 1, 0))  # (X,Y,Z)

                logits_np_zyx = logits.squeeze(0).squeeze(0).float().cpu().numpy()  # (Z,Y,X)
                logits_xyz = np.transpose(logits_np_zyx, (2, 1, 0)).astype(np.float32)  # (X,Y,Z)

                # Write clipped output region (in case last tile smaller)
                if out_img is not None:
                    out_img[ox:oxe, oy:oye, oz:oze] = fake_x_u16_xyz[: out_shape_xyz[0], : out_shape_xyz[1], : out_shape_xyz[2]]

                if out_logits is not None:
                    out_logits[ox:oxe, oy:oye, oz:oze] = logits_xyz[: out_shape_xyz[0], : out_shape_xyz[1], : out_shape_xyz[2]]

                print(f"Wrote tile XYZ [{ox}:{oxe}, {oy}:{oye}, {oz}:{oze}] (in with margin)")

    print("Done.")


if __name__ == "__main__":
    main()