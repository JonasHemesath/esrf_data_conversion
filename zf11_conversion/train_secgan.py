# train_secgan.py
from __future__ import annotations

import argparse
import os
from typing import Iterator, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from secgan_data import CloudVolumeRandomBlockDataset, VolumeSpec
from secgan_models import (
    Generator3D,
    PatchDiscriminator3D,
    LossWeights,
    cycle_l1,
    lsgan_mse,
    to_gan,
    from_gan,
)


def center_crop_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, _, d, h, w = x.shape
    _, _, rd, rh, rw = ref.shape
    sd = (d - rd) // 2
    sh = (h - rh) // 2
    sw = (w - rw) // 2
    return x[:, :, sd:sd + rd, sh:sh + rh, sw:sw + rw]


@torch.no_grad()
def freeze_(m: nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def build_loader(
    url: str,
    block_shape_xyz: Tuple[int, int, int],
    bbox: str | None,
    mip: int,
    num_workers: int,
    seed: int,
    batch_size: int,
    max_val: float,
) -> DataLoader:
    spec = VolumeSpec(url=url, mip=mip, bboxes=None if bbox is None else [bbox])
    ds = CloudVolumeRandomBlockDataset(
        spec,
        block_shape_xyz=block_shape_xyz,
        seed=seed,
        return_u16=False,
        max_val=max_val,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def infinite_zip(loader_a: DataLoader, loader_b: DataLoader) -> Iterator[Tuple[dict, dict]]:
    ita = iter(loader_a)
    itb = iter(loader_b)
    while True:
        try:
            a = next(ita)
        except StopIteration:
            ita = iter(loader_a)
            a = next(ita)
        try:
            b = next(itb)
        except StopIteration:
            itb = iter(loader_b)
            b = next(itb)
        yield a, b


def save_ckpt(path: str, step: int, G_XY, G_YX, D_X, D_Y, D_S, opt_G, opt_D, scaler):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "G_XY": G_XY.state_dict(),
            "G_YX": G_YX.state_dict(),
            "D_X": D_X.state_dict(),
            "D_Y": D_Y.state_dict(),
            "D_S": D_S.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_x", required=True)
    ap.add_argument("--vol_y", required=True)
    ap.add_argument("--bbox_x", default=None, help="x0,x1,y0,y1,z0,z1")
    ap.add_argument("--bbox_y", default=None, help="x0,x1,y0,y1,z0,z1")
    ap.add_argument("--mip", type=int, default=0)
    ap.add_argument("--block_shape", default="196,196,64", help="XYZ; consider smaller Z if anisotropic")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="./runs/secgan")
    ap.add_argument("--lambda_cyc", type=float, default=2.5)
    ap.add_argument("--lambda_seg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_val", type=float, default=65535.0, help="uint16 max (typically 65535)")
    ap.add_argument("--segmenter_ckpt", type=str, required=True)
    ap.add_argument("--segmenter_module", type=str, required=True, help="module that provides build_segmenter()")

    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    block_shape_xyz = tuple(int(x) for x in args.block_shape.split(","))
    if len(block_shape_xyz) != 3:
        raise ValueError("--block_shape must be 'sx,sy,sz' in XYZ order")

    loader_x = build_loader(args.vol_x, block_shape_xyz, args.bbox_x, args.mip, args.num_workers, args.seed + 1, args.batch_size, args.max_val)
    loader_y = build_loader(args.vol_y, block_shape_xyz, args.bbox_y, args.mip, args.num_workers, args.seed + 2, args.batch_size, args.max_val)

    # Models
    G_XY = Generator3D(in_ch=1, base_ch=32, num_blocks=8).to(device)
    G_YX = Generator3D(in_ch=1, base_ch=32, num_blocks=8).to(device)
    D_X = PatchDiscriminator3D(in_ch=1).to(device)
    D_Y = PatchDiscriminator3D(in_ch=1).to(device)
    D_S = PatchDiscriminator3D(in_ch=1).to(device)

    # Segmenter (frozen). Expects [0,1], returns logits.
    import importlib
    m = importlib.import_module(args.segmenter_module)
    S: nn.Module = m.build_segmenter()
    ckpt = torch.load(args.segmenter_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        S.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        S.load_state_dict(ckpt, strict=False)
    S = S.to(device)
    freeze_(S)

    opt_G = torch.optim.Adam(list(G_XY.parameters()) + list(G_YX.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()) + list(D_S.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))

    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
    w = LossWeights(lambda_cyc=args.lambda_cyc, lambda_seg=args.lambda_seg)

    it = infinite_zip(loader_x, loader_y)

    for step in range(1, args.steps + 1):
        batch_x, batch_y = next(it)

        # float32 [0,1], shape (N,1,D,H,W)
        x01 = batch_x["image01"].to(device, non_blocking=True)
        y01 = batch_y["image01"].to(device, non_blocking=True)

        # GAN inputs in [-1,1]
        x = to_gan(x01)
        y = to_gan(y01)

        # -------------------------
        # Update Generators
        # -------------------------
        opt_G.zero_grad(set_to_none=True)
        with autocast(enabled=(args.amp and device.type == "cuda")):
            fake_y = G_XY(x)  # [-1,1]
            fake_x = G_YX(y)  # [-1,1]

            rec_x = G_YX(fake_y)
            rec_y = G_XY(fake_x)

            # crop targets due to VALID conv shrink
            x_c = center_crop_like(x, rec_x)
            y_c = center_crop_like(y, rec_y)

            loss_cyc = cycle_l1(rec_x, x_c) + cycle_l1(rec_y, y_c)

            loss_adv_x = lsgan_mse(D_X(fake_x), True)
            loss_adv_y = lsgan_mse(D_Y(fake_y), True)

            # Segmentation enhancement on fake_x
            fake_x01 = from_gan(fake_x)  # [0,1]
            logits_fake = S(fake_x01)    # logits
            prob_fake = torch.sigmoid(logits_fake)
            loss_adv_s = lsgan_mse(D_S(prob_fake - 0.5), True)

            loss_G = w.lambda_cyc * loss_cyc + w.lambda_adv_x * loss_adv_x + w.lambda_adv_y * loss_adv_y + w.lambda_seg * loss_adv_s

        scaler.scale(loss_G).backward()
        scaler.step(opt_G)

        # -------------------------
        # Update Discriminators
        # -------------------------
        opt_D.zero_grad(set_to_none=True)
        with autocast(enabled=(args.amp and device.type == "cuda")):
            # D_X real/fake
            x_real_for_dx = center_crop_like(x, fake_x)  # match spatial support roughly
            loss_DX = lsgan_mse(D_X(x_real_for_dx.detach()), True) + lsgan_mse(D_X(fake_x.detach()), False)

            # D_Y real/fake
            y_real_for_dy = center_crop_like(y, fake_y)
            loss_DY = lsgan_mse(D_Y(y_real_for_dy.detach()), True) + lsgan_mse(D_Y(fake_y.detach()), False)

            # D_S: segmentations from real X vs fake_x
            x01_crop = center_crop_like(x01, fake_x)     # crop to generator output size
            logits_real = S(x01_crop)
            prob_real = torch.sigmoid(logits_real)

            prob_fake_det = prob_fake.detach()
            loss_DS = lsgan_mse(D_S(prob_real - 0.5), True) + lsgan_mse(D_S(prob_fake_det - 0.5), False)

            loss_D = loss_DX + loss_DY + loss_DS

        scaler.scale(loss_D).backward()
        scaler.step(opt_D)
        scaler.update()

        if step % 50 == 0:
            print(
                f"[{step:07d}] "
                f"loss_G={loss_G.item():.4f} (cyc={loss_cyc.item():.4f}, advX={loss_adv_x.item():.4f}, advY={loss_adv_y.item():.4f}, advS={loss_adv_s.item():.4f}) "
                f"loss_D={loss_D.item():.4f}"
            )

        if step % args.save_every == 0:
            save_ckpt(
                os.path.join(args.outdir, f"ckpt_{step:07d}.pt"),
                step, G_XY, G_YX, D_X, D_Y, D_S, opt_G, opt_D, scaler
            )


if __name__ == "__main__":
    main()