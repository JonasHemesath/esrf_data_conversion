import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SegResNet # Changed from UNet to SegResNet
from monai.losses import DiceLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandFlipd,
    ToTensord,
)
from monai.data import Dataset as MonaiDataset
from monai.inferers import sliding_window_inference
import tifffile
import torch.nn.functional as F
from monai.data import ImageReader
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


class TifffileReader(ImageReader):
    """A custom MONAI ImageReader for reading TIFF files using tifffile."""
    def read(self, data, **kwargs):
        """Reads a TIFF file and returns the numpy array."""
        return tifffile.imread(data)

    def get_data(self, img):
        """The image is already a numpy array, so just return it with empty metadata."""
        return img, {}

    def verify_suffix(self, filename: str) -> bool:
        """Verify the filename extension is supported by this reader."""
        return filename.lower().endswith((".tif", ".tiff"))


def main(args):
    """
    Main function to run the training and prediction pipeline.
    """
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

    # --- Load pretrained model if specified (for transfer learning) ---
    if args.mode == 'train' and args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            print(f"Loading pretrained model from {args.pretrained_path}")
            pretrained_dict = torch.load(args.pretrained_path, map_location=device)
            model_dict = model.state_dict()
            
            # Filter out unnecessary keys and keys with mismatched sizes.
            # This is important if the pretrained model has a different number of output channels.
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            # Update the current model's state dict with the pretrained weights.
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            num_loaded = len(pretrained_dict)
            num_total = len(model_dict)
            print(f"Loaded {num_loaded}/{num_total} matching layers from pretrained model.")
            if num_loaded == 0:
                print("Warning: No matching layers found in the pretrained model. Check compatibility. Training from scratch.")
        else:
            print(f"Warning: --pretrained_path was provided, but file not found at {args.pretrained_path}. Training from scratch.")

    # --- Training Mode ---
    if args.mode == 'train':
        print("--- Starting Training ---")

        # --- Load existing model if resuming ---
        if args.resume_training:
            if os.path.exists(args.model_path):
                print(f"Resuming training, loading model from {args.model_path}")
                model.load_state_dict(torch.load(args.model_path, map_location=device))
            else:
                print(f"Warning: --resume_training flag was passed, but model file was not found at {args.model_path}. Starting training from scratch.")

        # --- Set up output directory ---
        model_dir = os.path.dirname(args.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        else: # If no directory is in the path, save to the current directory
            model_dir = '.'
        
        # --- Find and process all training data ---
        print(f"Scanning for training data in: {args.train_data_dir}")
        
        # Directory to store combined labels
        preprocessed_label_dir = os.path.join(model_dir, "preprocessed_labels")
        os.makedirs(preprocessed_label_dir, exist_ok=True)
        
        train_files = []
        # Recursively walk through the training data directory
        for root, _, files in os.walk(args.train_data_dir):
            for file in files:
                if file.endswith("_raw.tif"):
                    raw_path = os.path.join(root, file)
                    print(raw_path)
                    base_name = file.replace("_raw.tif", "")
                    
                    # Construct paths for label files
                    vessels_path = os.path.join(root, f"{base_name}_BV.tif")
                    myelin_path = os.path.join(root, f"{base_name}_Myelin.tif")
                    somata_path = os.path.join(root, f"{base_name}_Soma.tif")
                    
                    # Check if all corresponding label files exist
                    if os.path.exists(vessels_path) and os.path.exists(myelin_path) and os.path.exists(somata_path):
                        print(f"Found training set: {base_name}")
                        
                        # --- Pre-process and Combine Labels for this set ---
                        vessels = tifffile.imread(vessels_path).astype(np.uint8)
                        myelin = tifffile.imread(myelin_path).astype(np.uint8)
                        somata = tifffile.imread(somata_path).astype(np.uint8)
                        
                        labels = np.zeros_like(vessels, dtype=np.uint8)
                        labels[myelin > 0] = 2
                        labels[somata > 0] = 3
                        labels[vessels > 0] = 1
                        
                        # Save the combined label to the preprocessed directory
                        combined_label_path = os.path.join(preprocessed_label_dir, f"{base_name}_combined_label.tif")
                        tifffile.imwrite(combined_label_path, labels)
                        
                        # Add to the list of files for the MONAI dataset
                        train_files.append({"image": raw_path, "label": combined_label_path})

        if not train_files:
            raise FileNotFoundError(f"No training sets found in {args.train_data_dir}. Ensure files follow the naming convention (*_raw.tif, *_BV.tif, etc.)")
        
        print(f"\nFound {len(train_files)} training sets.")

        # --- Training Transforms ---
        # These transforms augment the data during training to make the model more robust.
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], reader=TifffileReader()),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True
                ),
                # Crop 96x96x96 patches for training to fit in memory and for augmentation
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1, neg=1, num_samples=4,
                    image_key="image", image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                ToTensord(keys=["image", "label"]),
            ]
        )

        # --- Dataset and DataLoader ---
        train_ds = MonaiDataset(data=train_files, transform=train_transforms)
        # The RandCrop.. transform returns a list of samples, so we can't use a batch_size > 1 here.
        # The num_samples in RandCrop.. acts as an effective batch size.
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

        # --- Loss and Optimizer ---
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        # --- Training Loop ---
        model.train()
        epoch_losses = []
        loss_plot_path = os.path.join(model_dir, "training_loss_curve.png")
        best_loss = float('inf')
        best_model_path = os.path.join(model_dir, "current_best_model.pth")

        for epoch in range(args.epochs):
            epoch_loss = 0
            for i, batch_data in enumerate(train_loader):
                # The loader yields a list of dictionaries, get the first one
                data = batch_data[0]
                inputs, labels = data["image"].to(device), data["label"].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

            # --- Save best model checkpoint ---
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path} with loss: {best_loss:.4f}")

            # --- Save checkpoint and loss curve periodically ---
            if (epoch + 1) % 500 == 0:
                checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")

                plt.figure("train_loss", (12, 6))
                plt.title("Epoch Average Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(epoch_losses)
                plt.savefig(loss_plot_path)
                plt.close()
                print(f"Loss curve updated at {loss_plot_path}")

        print("Training finished.")
        torch.save(model.state_dict(), args.model_path)
        print(f"Final model saved to {args.model_path}")

        # --- Save final loss curve ---
        plt.figure("train_loss", (12, 6))
        plt.title("Epoch Average Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_losses)
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Final loss curve saved to {loss_plot_path}")

    # --- Prediction Mode ---
    elif args.mode == 'predict':
        print("--- Starting Prediction ---")
        if not args.predict_image or not os.path.exists(args.predict_image):
            raise ValueError("Prediction image must be provided and exist.")
        if not args.model_path or not os.path.exists(args.model_path):
             raise ValueError("Model path must be provided and exist for prediction.")

        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        # --- Prediction Transforms ---
        pred_transforms = Compose(
            [
                LoadImaged(keys=["image"], reader=TifffileReader()),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
                ToTensord(keys=["image"]),
            ]
        )
        
        pred_dict = [{"image": args.predict_image}]
        pred_ds = MonaiDataset(data=pred_dict, transform=pred_transforms)
        pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=0)
        
        with torch.no_grad():
            for pred_data in pred_loader:
                image = pred_data["image"].to(device)
                
                # Sliding window inference for large images
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                pred_output = sliding_window_inference(
                    image, roi_size, sw_batch_size, model
                )
                
                # Process output: apply softmax, get argmax, and convert to numpy
                pred_output = torch.argmax(F.softmax(pred_output, dim=1), dim=1).squeeze(0)
                pred_output_np = pred_output.cpu().numpy().astype(np.uint8)

                # Split back into semantic layers for each class
                vessels_pred = (pred_output_np == 1).astype(np.uint8)
                myelin_pred = (pred_output_np == 2).astype(np.uint8)
                somata_semantic_pred = (pred_output_np == 3).astype(np.uint8)
                
                # --- Instance Segmentation for Somata using Watershed ---
                print("Performing instance segmentation on somata using watershed...")
                
                # 1. Calculate the distance transform
                distance = distance_transform_edt(somata_semantic_pred)
                
                # 2. Find markers for the watershed using peak_local_max
                # This finds the local maxima in the distance transform, which are good markers for the centers of objects.
                peak_coords = peak_local_max(distance, min_distance=args.soma_min_distance, labels=somata_semantic_pred)
                markers_mask = np.zeros(distance.shape, dtype=bool)
                markers_mask[tuple(peak_coords.T)] = True
                markers = label(markers_mask)[0]

                # 3. Apply the watershed algorithm
                # The watershed algorithm finds basins in the inverted distance transform
                # The markers guide the flooding process.
                somata_instances = watershed(-distance, markers, mask=somata_semantic_pred)
                
                num_instances = somata_instances.max()
                print(f"Found {num_instances} individual somata instances.")

                # Save the prediction files
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                    
                tifffile.imwrite(os.path.join(args.output_dir, "vessels_prediction.tif"), vessels_pred)
                tifffile.imwrite(os.path.join(args.output_dir, "myelin_prediction.tif"), myelin_pred)
                tifffile.imwrite(os.path.join(args.output_dir, "somata_prediction.tif"), somata_semantic_pred)
                tifffile.imwrite(os.path.join(args.output_dir, "somata_instances_prediction.tif"), somata_instances.astype(np.uint16))
                
                print(f"Semantic predictions for vessels/myelin/somata and instance predictions for somata saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Brain Tissue Segmentation using a 3D SegResNet")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], 
                        help='Mode to run: "train" to train a new model, "predict" to run inference.')
    
    # --- Training Arguments ---
    parser.add_argument('--train_data_dir', type=str, help='(train mode) Path to the directory containing training images and labels.')
    parser.add_argument('--epochs', type=int, default=100, help='(train mode) Number of training epochs.')
    parser.add_argument('--resume_training', action='store_true', help='(train mode) If specified, load the model weights from --model_path and continue training.')
    parser.add_argument('--pretrained_path', type=str, help='(train mode) Path to a pretrained model to start from (transfer learning).')
    
    # --- Prediction Arguments ---
    parser.add_argument('--predict_image', type=str, help='(predict mode) Path to the image volume for prediction.')
    parser.add_argument('--output_dir', type=str, default='predictions', 
                        help='(predict mode) Directory to save prediction masks.')
    parser.add_argument('--soma_min_distance', type=int, default=15, 
                        help='(predict mode) The minimum distance between peaks of identified somata for watershed. (in pixels)')

    # --- Model Path ---
    parser.add_argument('--model_path', type=str, default='segmentation_model.pth', 
                        help='Path to save the trained model or load it for prediction.')
    
    args = parser.parse_args()
    main(args)
