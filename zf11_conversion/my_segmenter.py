# my_segmenter.py
import torch
import torch.nn as nn
from monai.networks.nets import UNet

def build_segmenter() -> nn.Module:
    # Example placeholder. Replace with your real architecture.
    # Must accept (N,1,D,H,W) and return (N,1,D,H,W) logits (preferred) or probabilities.
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )