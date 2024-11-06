import glob
import multiprocessing
import numpy as np
import rasterio
import sys
import os
import logging
from os import path

# Setup logging
logging.basicConfig(filename='tif_to_npy_errors.log', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

# Setup
in_directory = "D:/NinjaLABO/HySpecNet-11k/hsi-compression/datasets/hyspecnet-11k/patches/"

if not os.path.exists(in_directory):
    print("Directory does not exist:", in_directory)
    sys.exit(1)  # Exit the script if the directory does not exist
else:
    print("Directory found:", in_directory)

# Filtering out invalid channels
invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
valid_channels_ids = [c for c in range(1, 225) if c-1 not in invalid_channels]  # zero-based to one-based indexing

minimum_value = 0
maximum_value = 10000

# Conversion function
def convert(patch_path):
    if not path.exists(patch_path):
        logging.error(f"File not found: {patch_path}")
        return
    
    try:
        with rasterio.open(patch_path) as dataset:
            src = dataset.read(valid_channels_ids)
            clipped = np.clip(src, a_min=minimum_value, a_max=maximum_value)
            normalized = (clipped - minimum_value) / (maximum_value - minimum_value)
            out_data = normalized.astype(np.float32)
            
            # save npy
            out_path = patch_path.replace("SPECTRAL_IMAGE", "DATA").replace("TIF", "npy")
            np.save(out_path, out_data)
            
            if not path.exists(out_path):
                raise FileNotFoundError(f"Failed to create {out_path}")

            print(f"Converted and saved: {out_path}")
    except Exception as e:
        logging.error(f"Error processing {patch_path}: {str(e)}")

# Discover all spectral image files recursively
in_patches = glob.glob(path.join(in_directory, "**", "*SPECTRAL_IMAGE.TIF"), recursive=True)

# Execute conversions with multiprocessing
if __name__ == '__main__':
    with multiprocessing.Pool(processes=min(63, multiprocessing.cpu_count())) as pool:
        pool.map(convert, in_patches)
