import os
from PIL import Image
from tqdm import tqdm

def downscale_crop_images(input_folder, output_folder, scaler, crop_size):
    """
    Downscale all images in the input folder by a specified scaler and save to the output folder.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save downscaled images.
        scaler (float): Scale factor for downscaling (default is 0.5).
    """
    if not os.path.exists(input_folder):
        assert False, f"Input folder '{input_folder}' does not exist"
    
    # list all sub folders  
    sub_folders = ["image", "mask"]
    for sub_folder in tqdm(sub_folders, desc="Processing sub folders", total=len(sub_folders)):
        os.makedirs(os.path.join(output_folder, sub_folder), exist_ok=True)
        # list all files in the sub folder
        files = os.listdir(f"{input_folder}/{sub_folder}")
        for file_name in tqdm(files, desc="downscale and crop images", total=len(files)):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_folder, sub_folder, file_name)
                output_path = os.path.join(output_folder, sub_folder, os.path.splitext(file_name)[0] + '.png')

                try:
                    with Image.open(input_path) as img:
                        new_size = (int(img.width * scaler), int(img.height * scaler))
                        downscaled_img = img.resize(new_size, Image.Resampling.LANCZOS)
                        crop_pos = (new_size[0] // 2 - crop_size[0] // 2, new_size[1] // 2 - crop_size[1] // 2)
                        crop_img = downscaled_img.crop((crop_pos[0], crop_pos[1], crop_pos[0] + crop_size[0], crop_pos[1] + crop_size[1]))        
                        crop_img.save(output_path)
                except Exception as e:
                    print(f"Error processing '{input_path}': {e}")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--sub_folders", type=str, required=True)
    parser.add_argument("--scaler", type=float, required=True)
    parser.add_argument("--crop_size", type=str, required=True)
    args = parser.parse_args()
    crop_size = tuple(int(size) for size in args.crop_size.split(','))
    args.crop_size = crop_size
    downscale_crop_images(args.source_dir, args.target_dir, args.scaler, crop_size)