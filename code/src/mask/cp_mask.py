import os
import argparse
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--source_dir', type=str)
parser.add_argument('--target_dir', type=str)

# Parse arguments
args = parser.parse_args()

source_dir = args.source_dir
target_dir = args.target_dir


def change_file_name(source_dir, target_dir):
    for filename in tqdm(sorted(os.listdir(source_dir)), desc="copying images", total=len(os.listdir(source_dir))):
        if filename.endswith(".png"):
            index = int(filename.split(".")[0])
            new_filename = f'{index:04d}.png'
            old_file = os.path.join(source_dir, filename)
            new_file = os.path.join(target_dir, new_filename)
            cmd = f"cp {old_file} {new_file}"
            os.system(cmd)
        elif filename.endswith(".jpg"):
            index = int(filename.split(".")[0])
            new_filename = f'{index:04d}.png'
            old_file = os.path.join(source_dir, filename)
            new_file = os.path.join(target_dir, new_filename)
            with Image.open(old_file) as img:
                img = img.convert("RGB")  # Ensure proper format
                img.save(new_file, "PNG")


os.makedirs(target_dir,exist_ok=True)

change_file_name(source_dir, target_dir)
print("Done")