# secgan_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

BBox = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]  # ((x0,x1),(y0,y1),(z0,z1))


def _parse_bbox(b: Optional[Union[str, Sequence[int], BBox]]) -> Optional[BBox]:
    if b is None:
        return None
    if isinstance(b, str):
        parts = [int(x) for x in b.split(",")]
        if len(parts) != 6:
            raise ValueError("bbox string must be 'x0,x1,y0,y1,z0,z1'")
        x0, x1, y0, y1, z0, z1 = parts
        return ((x0, x1), (y0, y1), (z0, z1))
    if isinstance(b, (list, tuple)) and len(b) == 6 and all(isinstance(x, (int, np.integer)) for x in b):
        x0, x1, y0, y1, z0, z1 = map(int, b)
        return ((x0, x1), (y0, y1), (z0, z1))
    if isinstance(b, (list, tuple)) and len(b) == 3:
        (x0, x1), (y0, y1), (z0, z1) = b  # type: ignore[misc]
        return ((int(x0), int(x1)), (int(y0), int(y1)), (int(z0), int(z1)))
    raise ValueError(f"Unsupported bbox format: {type(b)}")


@dataclass
class VolumeSpec:
    url: str
    mip: int = 0
    bboxes: Optional[List[BBox]] = None
    assume_cv_returns_xyz: bool = True


class CloudVolumeRandomBlockDataset(IterableDataset):
    """
    Yields random 3D blocks from CloudVolume.

    Returns:
      dict(image01=torch.float32 in [0,1], shape (1,D,H,W))
      optionally: image_u16=torch.uint16 shape (1,D,H,W)
    """

    def __init__(
        self,
        spec: VolumeSpec,
        block_shape_xyz: Tuple[int, int, int],
        samples_per_worker: Optional[int] = None,
        return_u16: bool = False,
        max_val: float = 65535.0,
        seed: int = 0,
    ):
        super().__init__()
        self.spec = spec
        self.block_shape_xyz = tuple(int(x) for x in block_shape_xyz)
        self.samples_per_worker = samples_per_worker
        self.return_u16 = return_u16
        self.max_val = float(max_val)
        self.seed = int(seed)

        if self.spec.bboxes is not None:
            self.spec.bboxes = [_parse_bbox(bb) for bb in self.spec.bboxes]  # type: ignore[assignment]

    def _make_volume(self):
        from cloudvolume import CloudVolume
        return CloudVolume(self.spec.url, progress=False, mip=self.spec.mip)

    def _get_default_bbox(self, vol) -> BBox:
        if hasattr(vol, "bounds") and hasattr(vol.bounds, "minpt") and hasattr(vol.bounds, "maxpt"):
            mn = vol.bounds.minpt
            mx = vol.bounds.maxpt
            return ((int(mn.x), int(mx.x)), (int(mn.y), int(mx.y)), (int(mn.z), int(mx.z)))
        shape = vol.shape  # xyz
        return ((0, int(shape[0])), (0, int(shape[1])), (0, int(shape[2])))

    def _sample_origin(self, rng: np.random.Generator, bbox: BBox) -> Tuple[int, int, int]:
        (x0, x1), (y0, y1), (z0, z1) = bbox
        sx, sy, sz = self.block_shape_xyz
        if (x1 - x0) < sx or (y1 - y0) < sy or (z1 - z0) < sz:
            raise ValueError(f"Block {self.block_shape_xyz} does not fit in bbox {bbox}")
        ox = int(rng.integers(x0, x1 - sx + 1))
        oy = int(rng.integers(y0, y1 - sy + 1))
        oz = int(rng.integers(z0, z1 - sz + 1))
        return ox, oy, oz

    def __iter__(self) -> Iterator[dict]:
        wi = get_worker_info()
        worker_id = 0 if wi is None else wi.id
        rng = np.random.default_rng(self.seed + 10007 * worker_id)

        vol = self._make_volume()

        if self.spec.bboxes is None or len(self.spec.bboxes) == 0:
            bboxes = [self._get_default_bbox(vol)]
        else:
            bboxes = self.spec.bboxes

        n = self.samples_per_worker
        produced = 0

        while True:
            if n is not None and produced >= n:
                return

            bbox = bboxes[int(rng.integers(0, len(bboxes)))]
            ox, oy, oz = self._sample_origin(rng, bbox)
            sx, sy, sz = self.block_shape_xyz

            blk = np.asarray(vol[ox:ox + sx, oy:oy + sy, oz:oz + sz])
            if blk.ndim == 4 and blk.shape[-1] == 1:
                blk = blk[..., 0]

            if self.spec.assume_cv_returns_xyz:
                blk = np.transpose(blk, (2, 1, 0))  # (Z,Y,X)

            # enforce uint16
            blk_u16 = blk.astype(np.uint16, copy=False)

            # torch (1,D,H,W)
            t_u16 = torch.from_numpy(blk_u16).unsqueeze(0)  # uint16
            t01 = t_u16.to(torch.float32) / self.max_val    # float32 [0,1]

            out = {"image01": t01}
            if self.return_u16:
                out["image_u16"] = t_u16
            produced += 1
            yield out