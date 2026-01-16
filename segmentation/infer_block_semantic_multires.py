import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
import torch

import tifffile
import torch.nn.functional as F
from monai.data import ImageReader

from monai.transforms import MapTransform, Randomizable
from typing import Dict, Hashable, List, Sequence, Tuple
#from monai.data import RepeatDataset


from torch.utils.data import Dataset

import torch.nn as nn

from cloudvolume import CloudVolume

import math
import time



def center_crop_3d(x, target_zyx):
    tz, ty, tx = target_zyx
    _, _, D, H, W = x.shape
    sz = (D - tz) // 2
    sy = (H - ty) // 2
    sx = (W - tx) // 2
    return x[:, :, sz:sz+tz, sy:sy+ty, sx:sx+tx]

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(c_out)
        self.act  = nn.LeakyReLU(inplace=True)
        self.block = ConvBlock(c_out, c_out)
    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return self.block(x)

class UpCtx(nn.Module):
    """
    Upsample + concat(skip) + concat(context) + convblock.
    """
    def __init__(self, c_in, c_skip, c_ctx, c_out):
        super().__init__()
        self.up = nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)
        self.block = ConvBlock(c_out + c_skip + c_ctx, c_out)

    def forward(self, x, skip, ctx):
        x = self.up(x)
        # ctx must already be same spatial size as x and skip
        x = torch.cat([x, skip, ctx], dim=1)
        return self.block(x)

class MultiPathUNet_ds2_ds4_DecCtx(nn.Module):
    def __init__(self, out_channels=3, c_base=16, c_ds2=32, c_ds4=64, c_ctx=16):
        super().__init__()

        # -------------------------
        # High-res UNet (s0)
        # -------------------------
        self.e0 = ConvBlock(1, c_base)           # 16, 96
        self.e1 = Down(c_base, c_base*2)         # 32, 48
        self.e2 = Down(c_base*2, c_base*4)       # 64, 24  (stride 4)
        self.e3 = Down(c_base*4, c_base*8)       # 128, 12
        self.e4 = Down(c_base*8, c_base*16)      # 256, 6   (stride 16, bottleneck)

        # -------------------------
        # ds2 context path (96³ grid, already coarse physically)
        # -------------------------
        self.ds2_ctx = nn.Sequential(
            ConvBlock(1, c_ds2),
            ConvBlock(c_ds2, c_ds2),
        )
        self.ds2_proj = nn.Conv3d(c_ds2, c_ctx, kernel_size=1, bias=False)

        # fuse ds2 at encoder stage s2 (24³)
        self.fuse2 = ConvBlock(c_base*4 + c_ctx, c_base*4)

        # -------------------------
        # ds4 context path
        # -------------------------
        self.ds4_ctx = nn.Sequential(
            ConvBlock(1, c_ds4),
            ConvBlock(c_ds4, c_ds4),
        )
        self.ds4_proj = nn.Conv3d(c_ds4, c_ctx, kernel_size=1, bias=False)

        # fuse ds4 at bottleneck (6³)
        self.fuse4 = ConvBlock(c_base*16 + c_ctx, c_base*16)

        # -------------------------
        # Decoder with context injection
        # -------------------------
        # u3: 6->12, inject ds4 context at 12
        self.u3 = UpCtx(c_base*16, c_base*8,  c_ctx,      c_base*8)  # out 128
        # u2: 12->24, inject BOTH ds2@24 and ds4@24 (concat them => 2*c_ctx)
        self.u2 = UpCtx(c_base*8,  c_base*4,  2*c_ctx,    c_base*4)  # out 64
        # u1: 24->48, inject ds2@48
        self.u1 = UpCtx(c_base*4,  c_base*2,  c_ctx,      c_base*2)  # out 32
        # u0: 48->96, inject ds2@96
        self.u0 = UpCtx(c_base*2,  c_base,    c_ctx,      c_base)    # out 16

        self.out = nn.Conv3d(c_base, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B,3,96,96,96) with channels [s0, ds2, ds4]
        x_s0  = x[:, 0:1]
        x_ds2 = x[:, 1:2]
        x_ds4 = x[:, 2:3]

        # -------------------------
        # Compute context features
        # -------------------------
        f2 = self.ds2_proj(self.ds2_ctx(x_ds2))   # (B,c_ctx,96,96,96)
        f4 = self.ds4_proj(self.ds4_ctx(x_ds4))   # (B,c_ctx,96,96,96)

        # Anchor crops that correspond to the same physical 96³ high-res region
        f2_24 = center_crop_3d(f2, (24,24,24))  # stride-4 aligned context
        f4_6  = center_crop_3d(f4, (6,6,6))     # stride-16 aligned context

        # Prepare decoder-scale context maps by interpolation (no new info, but aligned conditioning)
        # If you ever suspect a 1-voxel shift, try align_corners=True consistently.
        f2_48 = F.interpolate(f2_24, size=(48,48,48), mode="trilinear", align_corners=False)
        f2_96 = F.interpolate(f2_24, size=(96,96,96), mode="trilinear", align_corners=False)

        f4_12 = F.interpolate(f4_6,  size=(12,12,12), mode="trilinear", align_corners=False)
        f4_24 = F.interpolate(f4_6,  size=(24,24,24), mode="trilinear", align_corners=False)

        # -------------------------
        # High-res encoder
        # -------------------------
        s0 = self.e0(x_s0)  # 96
        s1 = self.e1(s0)    # 48
        s2 = self.e2(s1)    # 24

        # Fuse ds2 at s2 (24³)
        s2 = self.fuse2(torch.cat([s2, f2_24], dim=1))

        s3 = self.e3(s2)    # 12
        b  = self.e4(s3)    # 6

        # Fuse ds4 at bottleneck (6³)
        b = self.fuse4(torch.cat([b, f4_6], dim=1))

        # -------------------------
        # Decoder with context injection
        # -------------------------
        x = self.u3(b,  s3, f4_12)
        x = self.u2(x,  s2, torch.cat([f2_24, f4_24], dim=1))
        x = self.u1(x,  s1, f2_48)
        x = self.u0(x,  s0, f2_96)

        return self.out(x)


class AttnGate3D(nn.Module):
    """
    Additive attention gate (Attention U-Net style), extended with optional context term.
    Produces alpha in [0,1] with shape (B,1,D,H,W), broadcast-multipliable with skip.
    """
    def __init__(self, c_skip, c_g, c_ctx, c_int):
        super().__init__()
        self.Ws = nn.Conv3d(c_skip, c_int, kernel_size=1, bias=False)
        self.Wg = nn.Conv3d(c_g,    c_int, kernel_size=1, bias=False)
        self.Wc = nn.Conv3d(c_ctx,  c_int, kernel_size=1, bias=False) if c_ctx > 0 else None

        self.norm = nn.InstanceNorm3d(c_int)
        self.act = nn.LeakyReLU(inplace=True)

        self.psi = nn.Conv3d(c_int, 1, kernel_size=1, bias=True)

    def forward(self, skip, g, ctx=None):
        # skip, g, ctx expected same spatial shape
        x = self.Ws(skip) + self.Wg(g)
        if self.Wc is not None:
            if ctx is None:
                raise ValueError("ctx is required but was None")
            x = x + self.Wc(ctx)
        x = self.act(self.norm(x))
        alpha = torch.sigmoid(self.psi(x))  # (B,1,D,H,W)
        return skip * alpha

class UpAttnCtx(nn.Module):
    """
    Upsample decoder feature, attention-gate the skip using (upsampled feature + ctx),
    then concat and conv.
    """
    def __init__(self, c_in, c_skip, c_ctx, c_out, attn_int=None, concat_ctx=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)
        self.concat_ctx = bool(concat_ctx)

        # attention internal channels
        if attn_int is None:
            attn_int = max(8, c_out // 2)

        self.attn = AttnGate3D(c_skip=c_skip, c_g=c_out, c_ctx=c_ctx, c_int=attn_int)

        cat_in = c_out + c_skip + (c_ctx if self.concat_ctx else 0)
        self.block = ConvBlock(cat_in, c_out)

    def forward(self, x, skip, ctx):
        x = self.up(x)  # gating signal g at target spatial size
        skip_att = self.attn(skip, x, ctx)

        if self.concat_ctx:
            x = torch.cat([x, skip_att, ctx], dim=1)
        else:
            x = torch.cat([x, skip_att], dim=1)

        return self.block(x)

class MultiPathUNet_Attn_ds2_ds4(nn.Module):
    def __init__(self, out_channels=3, c_base=16, c_ds2=32, c_ds4=64, c_ctx=16, concat_ctx=True):
        super().__init__()

        # -------------------------
        # High-res encoder (s0)
        # -------------------------
        self.e0 = ConvBlock(1, c_base)           # 16, 96
        self.e1 = Down(c_base, c_base*2)         # 32, 48
        self.e2 = Down(c_base*2, c_base*4)       # 64, 24  (stride 4)
        self.e3 = Down(c_base*4, c_base*8)       # 128, 12
        self.e4 = Down(c_base*8, c_base*16)      # 256, 6   (stride 16, bottleneck)

        # -------------------------
        # ds2 context path
        # -------------------------
        self.ds2_ctx = nn.Sequential(
            ConvBlock(1, c_ds2),
            ConvBlock(c_ds2, c_ds2),
        )
        self.ds2_proj = nn.Conv3d(c_ds2, c_ctx, kernel_size=1, bias=False)
        self.fuse2 = ConvBlock(c_base*4 + c_ctx, c_base*4)  # encoder fusion at 24³

        # -------------------------
        # ds4 context path
        # -------------------------
        self.ds4_ctx = nn.Sequential(
            ConvBlock(1, c_ds4),
            ConvBlock(c_ds4, c_ds4),
        )
        self.ds4_proj = nn.Conv3d(c_ds4, c_ctx, kernel_size=1, bias=False)
        self.fuse4 = ConvBlock(c_base*16 + c_ctx, c_base*16)  # bottleneck fusion at 6³

        # -------------------------
        # Decoder with attention-gated skips
        # -------------------------
        # u3: 6->12, gate skip s3 using ds4@12
        self.u3 = UpAttnCtx(c_in=c_base*16, c_skip=c_base*8,  c_ctx=c_ctx,    c_out=c_base*8,  concat_ctx=concat_ctx)

        # u2: 12->24, gate skip s2 using (ds2@24 + ds4@24) as ctx (2*c_ctx)
        self.u2 = UpAttnCtx(c_in=c_base*8,  c_skip=c_base*4,  c_ctx=2*c_ctx,  c_out=c_base*4,  concat_ctx=concat_ctx)

        # u1: 24->48, gate skip s1 using ds2@48
        self.u1 = UpAttnCtx(c_in=c_base*4,  c_skip=c_base*2,  c_ctx=c_ctx,    c_out=c_base*2,  concat_ctx=concat_ctx)

        # u0: 48->96, gate skip s0 using ds2@96
        self.u0 = UpAttnCtx(c_in=c_base*2,  c_skip=c_base,    c_ctx=c_ctx,    c_out=c_base,    concat_ctx=concat_ctx)

        self.out = nn.Conv3d(c_base, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B,3,96,96,96) channels [s0, ds2, ds4]
        x_s0  = x[:, 0:1]
        x_ds2 = x[:, 1:2]
        x_ds4 = x[:, 2:3]

        # ---- context features (projected to c_ctx)
        f2 = self.ds2_proj(self.ds2_ctx(x_ds2))  # (B,c_ctx,96,96,96)
        f4 = self.ds4_proj(self.ds4_ctx(x_ds4))  # (B,c_ctx,96,96,96)

        # anchor crops corresponding to the physical 96³ high-res region:
        f2_24 = center_crop_3d(f2, (24,24,24))
        f4_6  = center_crop_3d(f4, (6,6,6))

        # resize context to decoder levels
        f2_48 = F.interpolate(f2_24, size=(48,48,48), mode="trilinear", align_corners=False)
        f2_96 = F.interpolate(f2_24, size=(96,96,96), mode="trilinear", align_corners=False)

        f4_12 = F.interpolate(f4_6,  size=(12,12,12), mode="trilinear", align_corners=False)
        f4_24 = F.interpolate(f4_6,  size=(24,24,24), mode="trilinear", align_corners=False)

        # ---- high-res encoder
        s0 = self.e0(x_s0)  # 96
        s1 = self.e1(s0)    # 48
        s2 = self.e2(s1)    # 24

        # encoder fusion at stride-4
        s2 = self.fuse2(torch.cat([s2, f2_24], dim=1))

        s3 = self.e3(s2)    # 12
        b  = self.e4(s3)    # 6

        # bottleneck fusion at stride-16
        b = self.fuse4(torch.cat([b, f4_6], dim=1))

        # ---- decoder with attention-gated skips
        x = self.u3(b,  s3, f4_12)
        x = self.u2(x,  s2, torch.cat([f2_24, f4_24], dim=1))
        x = self.u1(x,  s1, f2_48)
        x = self.u0(x,  s0, f2_96)

        return self.out(x)

class RepeatDataset(Dataset):
    """Repeat an indexable dataset `times` times via modulo indexing."""
    def __init__(self, dataset, times: int):
        self.dataset = dataset
        self.times = int(times)
        if self.times <= 0:
            raise ValueError("times must be >= 1")

    def __len__(self):
        return len(self.dataset) * self.times

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]

class ApplyTransformDataset(Dataset):
    """
    Wrap a base dataset (e.g., CacheDataset) and apply an additional transform
    on-the-fly (not cached).
    """
    def __init__(self, base_ds, transform=None):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, index):
        item = self.base_ds[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

# --------------------------
# IO: TIFF reader for MONAI
# --------------------------
class TifffileReader(ImageReader):
    """A custom MONAI ImageReader for reading TIFF files using tifffile."""
    def read(self, data, **kwargs):
        return tifffile.imread(data)

    def get_data(self, img):
        return img, {}

    def verify_suffix(self, filename: str) -> bool:
        return filename.lower().endswith((".tif", ".tiff"))


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def worker_init_fn(_):
    torch.set_num_threads(1)

def _as_3tuple(x) -> Tuple[int, int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return int(x[0]), int(x[1]), int(x[2])
    raise ValueError(f"Expected a 3-tuple, got: {x}")


def _center_index_1d(length: int) -> float:
    # index-space center (works for even/odd): (L-1)/2
    return (float(length) - 1.0) / 2.0


def _float_patch_in_bounds(start_f_zyx: Tuple[float, float, float], shape_zyx: Tuple[int, int, int], roi_zyx: Tuple[int, int, int]) -> bool:
    """Check that sampling coordinates [start, start+roi-1] lie within [0, size-1]."""
    z0, y0, x0 = start_f_zyx
    rz, ry, rx = roi_zyx
    D, H, W = shape_zyx
    z1 = z0 + (rz - 1)
    y1 = y0 + (ry - 1)
    x1 = x0 + (rx - 1)
    return (z0 >= 0.0) and (y0 >= 0.0) and (x0 >= 0.0) and (z1 <= D - 1) and (y1 <= H - 1) and (x1 <= W - 1)


def extract_patch_trilinear(img_1dhw: torch.Tensor, start_f_zyx: Tuple[float, float, float], roi_size_zyx: Tuple[int, int, int],
                            padding_mode: str = "border") -> torch.Tensor:
    """
    Trilinear patch extraction at (possibly fractional) start coordinates using grid_sample.

    img_1dhw: (1, D, H, W) tensor
    returns:  (1, rz, ry, rx)
    """
    if img_1dhw.ndim != 4 or img_1dhw.shape[0] != 1:
        raise ValueError(f"Expected img shape (1,D,H,W), got {tuple(img_1dhw.shape)}")

    rz, ry, rx = roi_size_zyx
    _, D, H, W = img_1dhw.shape
    device = img_1dhw.device
    dtype = img_1dhw.dtype

    z0, y0, x0 = start_f_zyx

    # Build coordinate vectors in voxel index space (float)
    z = torch.arange(rz, device=device, dtype=dtype) + torch.tensor(z0, device=device, dtype=dtype)
    y = torch.arange(ry, device=device, dtype=dtype) + torch.tensor(y0, device=device, dtype=dtype)
    x = torch.arange(rx, device=device, dtype=dtype) + torch.tensor(x0, device=device, dtype=dtype)

    # Normalize to [-1,1] for align_corners=True
    z_norm = 2.0 * z / (D - 1) - 1.0
    y_norm = 2.0 * y / (H - 1) - 1.0
    x_norm = 2.0 * x / (W - 1) - 1.0

    zz, yy, xx = torch.meshgrid(z_norm, y_norm, x_norm, indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1)[None, ...]  # (1, rz, ry, rx, 3)

    out = F.grid_sample(
        img_1dhw.unsqueeze(0),  # (1, 1, D, H, W)
        grid,
        mode="bilinear",        # trilinear for 3D
        padding_mode=padding_mode,
        align_corners=True,
    )
    return out[0]  # (1, rz, ry, rx)


# ---------------------------------------------------------
# Multi-resolution aligned random crop (center in phys space)
#   - ref + label: integer slicing
#   - ds volumes: trilinear interpolation
# ---------------------------------------------------------
class MultiResAlignedRandSpatialCropd(MapTransform, Randomizable):
    """
    Random crop on ref_key at integer coordinates; aligned crops on other resolutions
    with patch centers corresponding in physical space.

    For downsampled keys, allows fractional aligned positions and extracts patches via
    trilinear interpolation (grid_sample).

    Raises if any required aligned patch would be out-of-bounds (float-bounds check).
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        ref_key: Hashable,
        factors: Dict[Hashable, float],
        roi_size: Sequence[int],
        max_tries: int = 100,
        interp_keys: Sequence[Hashable] | None = None,
        padding_mode: str = "border",
    ):
        super().__init__(keys)
        self.ref_key = ref_key
        self.factors = dict(factors)
        self.roi_size = _as_3tuple(roi_size)
        self.max_tries = int(max_tries)
        self.padding_mode = padding_mode

        if self.ref_key not in self.factors:
            raise ValueError(f"ref_key '{ref_key}' must be present in factors.")
        if abs(self.factors[self.ref_key] - 1.0) > 1e-12:
            raise ValueError("Reference key factor must be 1.0 (highest resolution).")

        # default: interpolate everything except ref and label
        if interp_keys is None:
            self.interp_keys = set(k for k in keys if k != ref_key and k != "label")
        else:
            self.interp_keys = set(interp_keys)

    @staticmethod
    def _get_spatial_shape(img: torch.Tensor) -> Tuple[int, int, int]:
        # expects (C, D, H, W)
        if not isinstance(img, torch.Tensor) or img.ndim != 4:
            raise ValueError(f"Expected torch.Tensor with shape (C,D,H,W), got {type(img)} {getattr(img, 'shape', None)}")
        return int(img.shape[1]), int(img.shape[2]), int(img.shape[3])

    @staticmethod
    def _compute_start_from_center(center: Tuple[float, float, float], roi: Tuple[int, int, int]) -> Tuple[float, float, float]:
        off = ((roi[0] - 1) / 2.0, (roi[1] - 1) / 2.0, (roi[2] - 1) / 2.0)
        return (center[0] - off[0], center[1] - off[1], center[2] - off[2])

    def _mapped_start_float_for_key(
        self,
        start_ref_int: Tuple[int, int, int],
        shape_ref: Tuple[int, int, int],
        shape_k: Tuple[int, int, int],
        factor_k: float,
    ) -> Tuple[float, float, float]:
        rz, ry, rx = self.roi_size
        center_off = ((rz - 1) / 2.0, (ry - 1) / 2.0, (rx - 1) / 2.0)
        pH = (start_ref_int[0] + center_off[0], start_ref_int[1] + center_off[1], start_ref_int[2] + center_off[2])

        Hc = (_center_index_1d(shape_ref[0]), _center_index_1d(shape_ref[1]), _center_index_1d(shape_ref[2]))
        Lc = (_center_index_1d(shape_k[0]), _center_index_1d(shape_k[1]), _center_index_1d(shape_k[2]))

        # pL = Lc + f*(pH - Hc)
        pL = (
            Lc[0] + factor_k * (pH[0] - Hc[0]),
            Lc[1] + factor_k * (pH[1] - Hc[1]),
            Lc[2] + factor_k * (pH[2] - Hc[2]),
        )
        return self._compute_start_from_center(pL, self.roi_size)

    def __call__(self, data):
        d = dict(data)

        ref = d[self.ref_key]
        shape_ref = self._get_spatial_shape(ref)
        rz, ry, rx = self.roi_size

        if rz > shape_ref[0] or ry > shape_ref[1] or rx > shape_ref[2]:
            raise RuntimeError(f"ROI {self.roi_size} larger than reference image shape {shape_ref}.")

        for _ in range(self.max_tries):
            z0 = int(self.R.randint(0, shape_ref[0] - rz + 1))
            y0 = int(self.R.randint(0, shape_ref[1] - ry + 1))
            x0 = int(self.R.randint(0, shape_ref[2] - rx + 1))
            start_ref = (z0, y0, x0)

            # compute float starts for ds keys and check bounds
            mapped_float: Dict[Hashable, Tuple[float, float, float]] = {}
            ok = True

            for k in self.keys:
                if k == self.ref_key or k == "label":
                    continue
                if k not in d:
                    raise KeyError(f"Key '{k}' not found in data dict.")
                if k not in self.factors:
                    raise KeyError(f"Key '{k}' not found in factors dict.")

                shape_k = self._get_spatial_shape(d[k])
                start_f = self._mapped_start_float_for_key(start_ref, shape_ref, shape_k, self.factors[k])

                if not _float_patch_in_bounds(start_f, shape_k, self.roi_size):
                    ok = False
                    break
                mapped_float[k] = start_f

            if not ok:
                continue

            # crop ref and label by slicing (integer)
            for k in self.keys:
                if k == self.ref_key:
                    img = d[k]
                    d[k] = img[:, z0:z0 + rz, y0:y0 + ry, x0:x0 + rx]
                elif k == "label":
                    lab = d[k]
                    d[k] = lab[:, z0:z0 + rz, y0:y0 + ry, x0:x0 + rx]
                else:
                    img = d[k]
                    if k in self.interp_keys:
                        d[k] = extract_patch_trilinear(img, mapped_float[k], self.roi_size, padding_mode=self.padding_mode)
                    else:
                        # fallback: round to nearest integer (not recommended)
                        sf = mapped_float[k]
                        zi, yi, xi = int(round(sf[0])), int(round(sf[1])), int(round(sf[2]))
                        d[k] = img[:, zi:zi + rz, yi:yi + ry, xi:xi + rx]

            return d

        raise RuntimeError(
            f"Could not find an in-bounds aligned crop across all resolutions after {self.max_tries} tries. "
            f"Your multi-res cubes likely don't provide enough margin for roi_size={self.roi_size}."
        )


# --------------------------
# Multi-res file helpers
# --------------------------
def ds_level_to_factor(level: int) -> float:
    if level <= 0:
        raise ValueError("Downsample levels must be positive integers (1,2,3,...)")
    return 1.0 / (2 ** level)


def build_multires_paths(raw_path: str, ds_levels: List[int]) -> Dict[str, str]:
    if not raw_path.endswith("_raw.tif"):
        raise ValueError(f"Expected raw file to end with '_raw.tif', got: {raw_path}")

    base = raw_path[:-len("_raw.tif")]
    out = {"image_s0": raw_path}
    for lvl in ds_levels:
        out[f"image_ds{lvl}"] = f"{base}_ds{lvl}.tif"
    return out


# --------------------------
# Multi-res sliding window inference (aligned centers, ds via interpolation)
# --------------------------
def multires_sliding_window_inference(
    vols: Dict[str, torch.Tensor],
    factors: Dict[str, float],
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    model: torch.nn.Module,
    device: torch.device,
    overlap: float = 0.25,
    ref_key: str = "image_s0",
    keys_order: List[str] | None = None,   # IMPORTANT: enforce consistent channel order
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    vols: dict key->tensor, each shape (1, D, H, W) typically on CPU
    returns: logits tensor shape (1, out_channels, D, H, W) on device
    """
    from monai.inferers.utils import dense_patch_slices, compute_importance_map

    roi_size = _as_3tuple(roi_size)

    if keys_order is None:
        # fallback (but you should pass image_keys from main)
        keys_order = list(vols.keys())

    ref = vols[ref_key]
    if ref.ndim != 4:
        raise ValueError(f"Expected ref volume shape (1,D,H,W), got {ref.shape}")

    _, D, H, W = ref.shape
    image_size = (D, H, W)

    # compute scan interval like MONAI
    scan_interval = tuple(max(1, int(r * (1.0 - overlap))) for r in roi_size)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    imp = compute_importance_map(roi_size, mode="gaussian", sigma_scale=0.125, device=device)  # (roiD,roiH,roiW)

    def mapped_start_float(start_ref: Tuple[int, int, int], shape_ref: Tuple[int, int, int], shape_k: Tuple[int, int, int], f: float):
        rz, ry, rx = roi_size
        center_off = ((rz - 1) / 2.0, (ry - 1) / 2.0, (rx - 1) / 2.0)

        pH = (start_ref[0] + center_off[0], start_ref[1] + center_off[1], start_ref[2] + center_off[2])
        Hc = (_center_index_1d(shape_ref[0]), _center_index_1d(shape_ref[1]), _center_index_1d(shape_ref[2]))
        Lc = (_center_index_1d(shape_k[0]), _center_index_1d(shape_k[1]), _center_index_1d(shape_k[2]))

        pL = (Lc[0] + f * (pH[0] - Hc[0]), Lc[1] + f * (pH[1] - Hc[1]), Lc[2] + f * (pH[2] - Hc[2]))
        startLf = (pL[0] - center_off[0], pL[1] - center_off[1], pL[2] - center_off[2])
        return startLf

    def extract_patch_batch(batch_slices):
        patch_list = []
        shape_ref = (D, H, W)

        for slc in batch_slices:
            z0, y0, x0 = int(slc[0].start), int(slc[1].start), int(slc[2].start)
            start_ref = (z0, y0, x0)

            patches_per_scale = []
            for k in keys_order:
                v = vols[k]  # (1,Dk,Hk,Wk)
                _, Dk, Hk, Wk = v.shape

                if k == ref_key:
                    # integer slice
                    patch = v[:, z0:z0 + roi_size[0], y0:y0 + roi_size[1], x0:x0 + roi_size[2]]
                else:
                    sf = mapped_start_float(start_ref, shape_ref, (Dk, Hk, Wk), factors[k])
                    if not _float_patch_in_bounds(sf, (Dk, Hk, Wk), roi_size):
                        raise RuntimeError(
                            f"Aligned inference patch out of bounds for key={k}. "
                            f"start_float={sf}, roi={roi_size}, shape={(Dk,Hk,Wk)}. "
                            "Provide larger multi-res cubes (more margin)."
                        )
                    # interpolate at float start
                    patch = extract_patch_trilinear(v, sf, roi_size, padding_mode=padding_mode)

                patches_per_scale.append(patch)

            patchC = torch.cat(patches_per_scale, dim=0)  # (C,roiD,roiH,roiW)
            patch_list.append(patchC)

        return torch.stack(patch_list, dim=0)  # (B,C,roiD,roiH,roiW)

    # dry run for out_channels
    with torch.no_grad():
        test_in = extract_patch_batch([slices[0]]).to(device)
        test_out = model(test_in)
        out_channels = int(test_out.shape[1])

    output = torch.zeros((1, out_channels, D, H, W), device=device, dtype=torch.float32)
    count_map = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)

    imp = imp.unsqueeze(0).unsqueeze(0)  # (1,1,roiD,roiH,roiW)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(slices), sw_batch_size):
            batch_slices = slices[i:i + sw_batch_size]
            x = extract_patch_batch(batch_slices).to(device)
            y = model(x)

            for b, slc in enumerate(batch_slices):
                zslice, yslice, xslice = slc
                output[..., zslice, yslice, xslice] += y[b:b + 1] * imp
                count_map[..., zslice, yslice, xslice] += imp

    output = output / torch.clamp_min(count_map, 1e-8)
    return output

def read_from_multires(path, ds, block_org, block_size, patch_size):
    patch_size += 10
    image = CloudVolume(path, mip=ds)
    data_shape = image.info['scales'][ds]['size']
    data_type_str = image.info['data_type']
    if data_type_str == 'uint8':
        data_type = np.uint8
    elif data_type_str == 'uint16':
        data_type = np.uint16
    ds_block_size = [math.ceil(block_size[0] / (2**ds)) + patch_size - math.floor(patch_size/(2**ds)),
                     math.ceil(block_size[1] / (2**ds)) + patch_size - math.floor(patch_size/(2**ds)),
                     math.ceil(block_size[2] / (2**ds)) + patch_size - math.floor(patch_size/(2**ds))]

    ds_block_org = [round(((2 * block_org[0] + block_size[0])/2)/(2**ds))-ds_block_size[0]//2, 
                    round(((2 * block_org[1] + block_size[1])/2)/(2**ds))-ds_block_size[1]//2, 
                    round(((2 * block_org[2] + block_size[2])/2)/(2**ds))-ds_block_size[2]//2]
    print(ds_block_org)

    ds_block_max = [ds_block_org[0]+ds_block_size[0],
                    ds_block_org[1]+ds_block_size[1],
                    ds_block_org[2]+ds_block_size[2]]
    print(ds_block_max)
    vol_out = np.zeros((ds_block_size[0], ds_block_size[1], ds_block_size[2]), dtype=data_type)

    ds_block_org_adjust = []
    ds_block_max_adjust = []
    vol_org = []
    vol_max = []

    for i in range(3):
        if ds_block_org[i] < 0:
            ds_block_org_adjust.append(0)
            vol_org.append(-ds_block_org[i])
        else:
            ds_block_org_adjust.append(ds_block_org[i])
            vol_org.append(0)
        if ds_block_max[i] > data_shape[i]:
            ds_block_max_adjust.append(data_shape[i])
            vol_max.append(ds_block_size[i] - (ds_block_max[i] - data_shape[i]))
        else:
            ds_block_max_adjust.append(ds_block_max[i])
            vol_max.append(ds_block_size[i])

    vol_xyz = np.squeeze(image[ds_block_org_adjust[0]:ds_block_max_adjust[0],
                            ds_block_org_adjust[1]:ds_block_max_adjust[1],
                            ds_block_org_adjust[2]:ds_block_max_adjust[2]])
    
    vol_out[vol_org[0]:vol_max[0],
            vol_org[1]:vol_max[1],
            vol_org[2]:vol_max[2]] = vol_xyz
    return vol_out


# --------------------------
# Main
# --------------------------
def main(args):
    t1 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds_levels = sorted(list(set(args.ds_levels)))
    image_keys = ["image_s0"] + [f"image_ds{lvl}" for lvl in ds_levels]
    image_levels = [0] + ds_levels
    factors = {"image_s0": 1.0}
    for lvl in ds_levels:
        factors[f"image_ds{lvl}"] = ds_level_to_factor(lvl)

    print(f"Using resolution levels (as input channels): {image_keys}")
    print(f"Downsample factors: {factors}")

    #model = UNet(
    #    spatial_dims=3,
    #    in_channels=len(image_keys),
    #    out_channels=3,
    #    channels=(16, 32, 64, 128, 256),
    #    strides=(2, 2, 2, 2),
    #    num_res_units=2,
    #).to(device)

    if args.mode == 'myelin_BV':
        out_channels = 3
    elif args.mode == 'soma':
        out_channels = 2
    elif args.mode == 'markers':
        out_channels = 1

    model = MultiPathUNet_ds2_ds4_DecCtx(out_channels=out_channels).to(device)
    if args.attention:
        model = MultiPathUNet_Attn_ds2_ds4(out_channels=out_channels).to(device)
    

    print("--- Starting Prediction ---")
    if not args.predict_image or not os.path.exists(args.predict_image):
        raise ValueError("Prediction image must be provided and exist.")
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError("Model path must be provided and exist for prediction.")


    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    #multires_paths = build_multires_paths(args.predict_image, ds_levels)
    #for k, p in multires_paths.items():
    #    if not os.path.exists(p):
    #        raise FileNotFoundError(f"Missing prediction input for {k}: {p}")

    vols: Dict[str, torch.Tensor] = {}
    print('loading data...')
    for i, k in enumerate(image_keys):
        #arr = tifffile.imread(multires_paths[k]).astype(np.float32)
        
        if i == 0:
            image = CloudVolume(args.predict_image, mip=0)
            print('highres image opened')
            arr = np.squeeze(image[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
                        args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
                        args.block_origin[2]:args.block_origin[2]+args.block_shape[2]]).astype(np.float32)
            print('highres loaded, shape: ', arr.shape)
            sum_val = np.sum(arr)
            if sum_val == 0:
                if args.debug_path is not None:
                    print('Empty block detected, skipping process_id ', args.process_id)
                    msg = 'Empty block detected, skipping process_id ' + str(args.process_id)
                    with open(os.path.join(args.debug_path, str(args.process_id) + '.txt'), 'w') as f:
                        f.write(msg)
                return
        else:
            arr = read_from_multires(args.predict_image, image_levels[i], block_org=args.block_origin, block_size=args.block_shape, patch_size=args.patch_size).astype(np.float32)
            print(f'lowres ds{image_levels[i]} loaded, shape: {arr.shape}')
        arr = np.clip(arr / 65535.0, 0.0, 1.0)
        arr = arr[None, ...]  # (1,D,H,W)
        vols[k] = torch.from_numpy(arr)  # keep on CPU
    print('data loaded, starting inference...')
    t2 = time.time()
    roi_size = (args.patch_size, args.patch_size, args.patch_size)
    logits = multires_sliding_window_inference(
        vols=vols,
        factors=factors,
        roi_size=roi_size,
        sw_batch_size=args.sw_batch_size,
        model=model,
        device=device,
        overlap=args.overlap,
        ref_key="image_s0",
        keys_order=image_keys,       # enforce same channel order as training
        padding_mode="border",
    )

    if out_channels > 1:
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1).squeeze(0)
        pred_np = pred.cpu().numpy().astype(np.uint64)

        
    
    elif out_channels == 1:
        pred_np = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32) 

    t3 = time.time()
        
    out_vol = CloudVolume(args.output_dir, parallel=1, non_aligned_writes=True, fill_missing=True)
    out_vol[args.block_origin[0]+50:args.block_origin[0]+args.block_shape[0]-50,
            args.block_origin[1]+50:args.block_origin[1]+args.block_shape[1]-50,
            args.block_origin[2]+50:args.block_origin[2]+args.block_shape[2]-50] = pred_np[50:args.block_shape[0]-50, 50:args.block_shape[1]-50, 50:args.block_shape[2]-50]
    

    t4 = time.time()

    if args.debug_path is not None:
        msg = 'Time for reading: ' + str(round(t2-t1)) + ' s\nTime for inference: ' + str(round(t3-t2)) + ' s\nTime for writing: ' + str(round(t4-t3)) + ' s'
        with open(os.path.join(args.debug_path, str(args.process_id) + '.txt'), 'w') as f:
            f.write(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation using a 3D U-Net (multi-resolution inputs)")

    parser.add_argument('--mode', type=str, choices=['myelin_BV', 'soma', 'marker'], required=True, 
                        help='Mode of the segmentation')

    parser.add_argument(
        "--ds_levels",
        type=int,
        nargs="*",
        default=[],
        help="Downsample levels to include as extra input channels. "
             "Each level n expects '*_ds{n}.tif' and corresponds to factor 1/(2^n).",
    )

 
    # Prediction args
    parser.add_argument("--predict_image", type=str, help="Path to the HIGH-res volume '*_raw.tif'. Lower-res inputs are inferred via naming.")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save prediction masks.")
    parser.add_argument("--sw_batch_size", type=int, default=4, help="Sliding window batch size.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding window overlap in [0,1).")
    parser.add_argument("--attention", type=bool, default=False,
                    help="Use attention gates.")
    parser.add_argument('--block_origin', nargs=3, type=int, required=True, 
                        help='Origin of the block to load')
    parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load')
    parser.add_argument('--patch_size', type=int, default=96, 
                        help='Patch size for prediction')
    parser.add_argument('--process_id', type=int, required=True, 
                        help='Process ID')
    parser.add_argument('--debug_path', type=str, default=None, 
                        help='Path to the debug_folder')

    # Model path
    parser.add_argument("--model_path", type=str, default="segmentation_model.pth", help="Path to save the trained model or load it for prediction.")

    args = parser.parse_args()
    if args.block_shape[0] > 100 and args.block_shape[1] > 100 and args.block_shape[2] > 100:
        main(args)  