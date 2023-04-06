import os
from PIL import Image, ImageEnhance
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
import subprocess
import cv2
import sys
sys.path = ["code/"] + sys.path
from src.utils.const import SEGM_IDS
from tqdm import tqdm

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--out_mask_path', type=str, default='', help='merged mask folder path')
parser.add_argument('--object_mask_path', type=str, default='', help='object mask folder path')
parser.add_argument('--hand_mask_path', type=str, default='', help='hand mask folder path')

# Parse arguments
args = parser.parse_args()


# out_mask_path = '/home/simba/Documents/project/hold-private/code/data/hezi/processed/masks'
# object_mask_path = '/home/simba/Documents/project/hold-private/code/data/hezi/processed/masks_object'
# hand_mask_path = '/home/simba/Documents/project/hold-private/code/data/hezi/processed/masks_hand'

out_mask_path = args.out_mask_path
object_mask_path = args.object_mask_path
hand_mask_path = args.hand_mask_path

# Ensure the output folder exists
os.makedirs(out_mask_path, exist_ok=True)

# List all files in the directory and filter out those with a .png extension
object_mask_files = [file for file in os.listdir(object_mask_path) if file.endswith('.png')]

object_mask_files.sort()  # Sort the png_files list

hand_mask_files = [file for file in os.listdir(hand_mask_path) if file.endswith('.png')]

hand_mask_files.sort()  # Sort the png_files list

assert len(object_mask_files) == len(hand_mask_files), "Number of object files and hand files do not match"


# Iterate through the list of PNG files and read each one
for i, object_mask_item in tqdm(enumerate(object_mask_files), desc="merging masks", total=len(object_mask_files)):
    # Construct the full file path
    object_mask_file = os.path.join(object_mask_path, object_mask_item)
    hand_mask_file = os.path.join(hand_mask_path, object_mask_item)  # Assuming color images are in .jpg
    # Check if SamTrack_file_path and color_file_path exist
    if os.path.exists(hand_mask_file) and os.path.exists(object_mask_file):
        object_mask = cv2.imread(object_mask_file, cv2.IMREAD_GRAYSCALE)
        _, object_binary_mask = cv2.threshold(object_mask, 127, 255, cv2.THRESH_BINARY)
        hand_mask = cv2.imread(hand_mask_file, cv2.IMREAD_GRAYSCALE)
        _, hand_binary_mask = cv2.threshold(hand_mask, 127, 255, cv2.THRESH_BINARY)
        merged_mask = SEGM_IDS["object"] * (object_binary_mask > 0) + SEGM_IDS["right"] * (hand_binary_mask > 0)
        merged_mask[np.where(merged_mask == (SEGM_IDS["object"] + SEGM_IDS["right"]))] = SEGM_IDS["right"]
        # masked_color = cv2.bitwise_and(color, color, mask=binary_mask)
        merged_mask_file = os.path.join(out_mask_path, object_mask_item)
        cv2.imwrite(merged_mask_file, merged_mask)
    else:
        print(f"File not found: {hand_mask_file} or {object_mask_file}")
