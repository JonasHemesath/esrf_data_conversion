import numpy as np
import tifffile
import tensorstore as ts
import sys
import os
import torch
import torch.nn.functional as F
from monai.data import ImageReader
from monai.networks.nets import SegResNet
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



si = 0

dataset_name = sys.argv[1]

block_org = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
block_size = int(sys.argv[5])
model_path = sys.argv[6]

iden = 'https://syconn.esc.mpcdf.mpg.de/johem/ng/' + dataset_name + '/'
#iden = '/cajal/nvmescratch/projects/from_ssdscratch/songbird/johem/ng/' + dataset_name + '/'

dataset_future = ts.open({
     'driver':
        'neuroglancer_precomputed',
    'kvstore':
         #'https://syconn.esc.mpcdf.mpg.de/johem/ng/zf13_hr2/',
        iden,
    'scale_index':
        si,
     # Use 100MB in-memory cache.
     'context': {
         'cache_pool': {
             'total_bytes_limit': 100_000_000
         }
     },
     'recheck_cached_data':
         'open',
})

dataset = dataset_future.result()

#print(dataset)

dataset_3d = dataset[ts.d['channel'][0]]

out_shape = [dataset_3d.shape[2], dataset_3d.shape[1], dataset_3d.shape[0]]
#print(dataset_3d.shape)
#print(dataset_3d.dtype)

if str(dataset_3d.dtype) == 'dtype("uint8")':
    data_type = np.uint8
elif str(dataset_3d.dtype) == 'dtype("uint16")':
    data_type = np.uint16

block_size_x = min(block_size, dataset_3d.shape[0]-block_org[0])
block_size_y = min(block_size, dataset_3d.shape[1]-block_org[1])
block_size_z = min(block_size, dataset_3d.shape[2]-block_org[2])

vol = dataset_3d[
                block_org[0]:block_org[0]+block_size_x,
                block_org[1]:block_org[1]+block_size_y,
                block_org[2]:block_org[2]+block_size_z
                ].read().result()

vol = vol.transpose(2,1,0)
#print(type(vol))

if np.sum(vol) > 0 and block_size_x > 100 and block_size_y > 100 and block_size_z > 100:

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Definition ---
    # This is a SegResNet, configured to match the pretrained model architecture.
    # The out_channels is set to 4 for your specific task (BG, vessels, myelin, somata).
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)



    print("--- Starting Prediction ---")
    if not model_path or not os.path.exists(model_path):
        raise ValueError("Model path must be provided and exist for prediction.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Prediction ---
    a_max_val = 255.0 if data_type == np.uint8 else 65535.0

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
        pred_output_np = pred_output.cpu().numpy().astype(np.uint32)



    img_pi = pi.newimage(ImageDataType.UINT32, block_size, block_size, block_size)
    img_pi.from_numpy(pred_output_np[50:block_size_z-50, 50:block_size_y-50, 50:block_size_x-50])

    out_name = dataset_name + '_semantic_seg_' + str(out_shape[0]) + 'x' + str(out_shape[1]) + 'x' + str(out_shape[2]) + '.raw'


    pi.writerawblock(img_pi, out_name, [block_org[2]+50, block_org[1]+50, block_org[0]+50], [0, 0, 0], [0, 0, 0], [block_size_z-100, block_size_y-100, block_size_x-100])
    print('done')