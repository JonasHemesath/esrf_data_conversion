import os

import argparse

import numpy as np
import time

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
import zarr
from filelock import FileLock

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
parser.add_argument('--zarr_path', type=str, required=True, 
                        help='Path to the zarr array for output')
parser.add_argument('--process_id', type=int, required=True, 
                        help='Process ID')
parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model')
parser.add_argument('--debug_path', type=str, default=None, 
                        help='Path to the debug_folder')

args = parser.parse_args()

if args.dataset_dtype == 'uint8':
    dataset_dtype = np.uint8
elif args.dataset_dtype == 'uint16':
    dataset_dtype = np.uint16

output_dtype_np = np.uint64

t1 = time.time()
data = np.memmap(args.data_path, dtype=dataset_dtype, mode='r', shape=tuple(args.dataset_shape), order='F')
vol = data[args.block_origin[0]:args.block_origin[0]+args.block_shape[0],
           args.block_origin[1]:args.block_origin[1]+args.block_shape[1],
           args.block_origin[2]:args.block_origin[2]+args.block_shape[2]].copy()  # Copy to make writable and avoid warnings

t2 = time.time()
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
        # Reduced sw_batch_size to save memory
        roi_size = (96, 96, 96)
        sw_batch_size = 1  # Reduced from 4 to 1 to avoid OOM
        overlap = 0.25  # Overlap between windows
        
        pred_output = sliding_window_inference(
            input_tensor, roi_size, sw_batch_size, model, overlap=overlap
        )

        # Process output: use logits directly with argmax to avoid creating full softmax tensor
        # This is more memory efficient than softmax + argmax
        pred_output = torch.argmax(pred_output, dim=1).squeeze(0)
        
        # Clear GPU cache before moving to CPU
        torch.cuda.empty_cache()
        
        pred_output_np = pred_output.cpu().numpy().astype(output_dtype_np)
        
        # Clear more memory
        del pred_output
        del input_tensor
        torch.cuda.empty_cache()

    t3 = time.time()

    # Trim 50 pixels from each side (100 total reduction per dimension)
    # Extract the valid region (excluding borders)
    output_block = pred_output_np[50:args.block_shape[0]-50, 
                                   50:args.block_shape[1]-50, 
                                   50:args.block_shape[2]-50]
    
    # Calculate output dimensions and write location
    output_shape = (args.block_shape[0]-100, args.block_shape[1]-100, args.block_shape[2]-100)
    write_origin = (args.block_origin[0]+50, args.block_origin[1]+50, args.block_origin[2]+50)
    write_end = (write_origin[0] + output_shape[0], 
                 write_origin[1] + output_shape[1], 
                 write_origin[2] + output_shape[2])

    # Open zarr array and write with FileLock for thread safety
    z = zarr.open_array(args.zarr_path, mode='a')
    
    # FileLock creates a file-based lock that locks the ENTIRE zarr array, not just a portion.
    # When one process acquires the lock, all other processes must wait, even if they're
    # writing to different (non-overlapping) regions. This is more conservative than needed
    # since our blocks don't overlap, but it ensures safety. Zarr itself supports concurrent
    # writes to different chunks, but FileLock provides an extra safety layer.
    # Note: Since blocks are non-overlapping, FileLock could be removed for better performance,
    # but it's kept here for safety.
    if args.block_shape[0]-100 > 512 and args.block_shape[1]-100 > 512 and args.block_shape[2]-100 > 512:
        z[write_origin[0]:write_end[0],
          write_origin[1]:write_end[1],
          write_origin[2]:write_end[2]] = output_block
    else:
        lock_file = f"{args.zarr_path}.lock"
        with FileLock(lock_file):
            z[write_origin[0]:write_end[0],
            write_origin[1]:write_end[1],
            write_origin[2]:write_end[2]] = output_block
    
    t4 = time.time()

    if args.debug_path is not None:
        msg = 'Time for reading: ' + str(round(t2-t1)) + ' s\nTime for inference: ' + str(round(t3-t2)) + ' s\nTime for writing: ' + str(round(t4-t3)) + ' s'
        with open(os.path.join(args.debug_path, str(args.process_id) + '.txt'), 'w') as f:
            f.write(msg)
    #print('done')

