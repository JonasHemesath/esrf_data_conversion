import os
import sys

import argparse

import numpy as np

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.inferers import sliding_window_inference
sys.path.append("/cajal/nvmescratch/users/johem/pi2_4_5/pi2/bin-linux64/release-nocl")
from pi2py2 import *

pi = Pi2()


parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation")
parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset')
parser.add_argument('--dataset_shape', nargs=3, type=int, required=True, 
                        help='Shape of the dataset')
parser.add_argument('--dataset_dtype', type=str, choices=['uint8', 'uint16'], required=True, 
                        help='Datatype of the dataset')
parser.add_argument('--block_origin', nargs=3, type=int, required=True, 
                        help='Origin of the block to load')
parser.add_argument('--block_shape', nargs=3, type=int, required=True, 
                        help='Shape of the block to load')
parser.add_argument('--output_name', type=str, required=True, 
                        help='Filename of the output')
parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model')

args = parser.parse_args()

if args.dataset_dtype == 'uint8':
    dataset_dtype = np.uint8
elif args.dataset_dtype == 'uint16':
    dataset_dtype = np.uint16


output_dtype = ImageDataType.UINT64
output_dtype_np = np.uint64


data = np.memmap(args.data_path, dtype=dataset_dtype, mode='r', shape=tuple(args.dataset_shape), order='F')
vol = data[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
           args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
           args.block_origin[2]:args.block_origin[2]+args.block_shape[2]]


if np.sum(vol) > 0 and args.block_shape[0] > 100 and args.block_shape[1] > 100 and args.block_shape[2] > 100:

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    # --- Model Definition ---
    # This is a SegResNet, configured to match the pretrained model architecture.
    # The out_channels is set to 4 for your specific task (BG, vessels, myelin, somata).
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)



    #print("--- Starting Prediction ---")
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError("Model path must be provided and exist for prediction.")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- Prediction ---
    a_max_val = 255.0 if dataset_dtype == np.uint8 else 65535.0

    pred_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=a_max_val, b_min=0.0, b_max=1.0, clip=True
            ),
            ToTensord(keys=["image"]),
        ]
    )

    input_data = pred_transforms({"image": vol})
    input_tensor = input_data["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        # Sliding window inference for large images
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        pred_output = sliding_window_inference(
            input_tensor, roi_size, sw_batch_size, model
        )

        # Process output: apply softmax, get argmax, and convert to numpy
        pred_output = torch.argmax(F.softmax(pred_output, dim=1), dim=1).squeeze(0)
        pred_output_np = pred_output.cpu().numpy().astype(output_dtype_np)



    img_pi = pi.newimage(output_dtype, args.block_shape[0]-100, args.block_shape[1]-100, args.block_shape[2]-100)
    img_pi.from_numpy(pred_output_np[50:args.block_shape[0]-50, 50:args.block_shape[1]-50, 50:args.block_shape[2]-50])

    #out_name = dataset_name + '_semantic_seg_' + str(out_shape[0]) + 'x' + str(out_shape[1]) + 'x' + str(out_shape[2]) + '.raw'


    pi.writerawblock(img_pi, args.output_name, [args.block_origin[0]+50, args.block_origin[1]+50, args.block_origin[2]+50], [0, 0, 0], [0, 0, 0], [args.block_shape[0]-100, args.block_shape[1]-100, args.block_shape[2]-100])
    #print('done')