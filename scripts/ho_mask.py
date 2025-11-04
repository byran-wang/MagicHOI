import os
import subprocess
import argparse

# Define the scenes


def main(args):
    seq_name = args.seq_name
    CUTIE_PATH = args.cutie_path
    DATA_PATH = args.data_path
    CUTIE_CONDA_ENV_PATH = os.path.expanduser(os.environ.get("PYCUTIE"))
    #check whether CUTIE_CONDA_ENV_PATH is exist
    if CUTIE_CONDA_ENV_PATH is None:
        raise ValueError("Please set the PYCUTIE environment variable to the path of the Cutie conda environment python executable.")
    elif os.path.exists(CUTIE_CONDA_ENV_PATH) is False:
        raise ValueError(f"The path {CUTIE_CONDA_ENV_PATH} set in PYCUTIE does not exist.")
    
    scaler = args.scaler
    crop_size = args.crop_size

    # Preprocessing - generate object and hand masks using Cutie
    current_dir = os.getcwd()
    workspace_object = f"{current_dir}/{CUTIE_PATH}/workspace/{seq_name}_object"
    workspace_hand = f"{current_dir}/{CUTIE_PATH}/workspace/{seq_name}_hand"
    video_path = f"{current_dir}/{DATA_PATH}/{seq_name}/video.mp4"


    # Generate object mask
    subprocess.run(["rm", "-rf", workspace_object])
    subprocess.run([CUTIE_CONDA_ENV_PATH, "./interactive_demo.py", "--video", video_path, "--workspace", workspace_object, "--num_objects", "1"], cwd=CUTIE_PATH)
    
    # Generate hand mask
    subprocess.run(["rm", "-rf", workspace_hand])
    subprocess.run([CUTIE_CONDA_ENV_PATH, "./interactive_demo.py", "--video", video_path, "--workspace", workspace_hand, "--num_objects", "1"], cwd=CUTIE_PATH)

    # Copy Cutie generated masks and images
    source_dir = f"{CUTIE_PATH}/workspace/{seq_name}_hand/images/"
    target_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/images/"
    subprocess.run(["python", "scripts/cp_mask.py", "--source_dir", source_dir, "--target_dir", target_dir])
    
    source_dir = f"{CUTIE_PATH}/workspace/{seq_name}_hand/binary_masks/"
    target_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_hand/"
    subprocess.run(["python", "scripts/cp_mask.py", "--source_dir", source_dir, "--target_dir", target_dir])

    source_dir = f"{CUTIE_PATH}/workspace/{seq_name}_object/binary_masks/"
    target_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_object/"
    subprocess.run(["python", "scripts/cp_mask.py", "--source_dir", source_dir, "--target_dir", target_dir])

    # Merge object and hand masks
    out_mask_path = f"{DATA_PATH}/{seq_name}/build_origin_size/masks"
    object_mask_path = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_object"
    hand_mask_path = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_hand"
    subprocess.run(["python", "scripts/merge_mask.py", "--out_mask_path", out_mask_path, "--object_mask_path", object_mask_path, "--hand_mask_path", hand_mask_path])

    # Downscale and crop images
    source_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/"
    target_dir = f"{DATA_PATH}/{seq_name}/processed/"
    subprocess.run(["python", "scripts/down_crop_seq.py", 
                    "--source_dir", source_dir, 
                    "--target_dir", target_dir, 
                    "--sub_folders", "images,masks",
                    "--scaler", str(scaler),
                    "--crop_size", crop_size])
    
    # Generate RGBA images for the objects
    source_dir = f"{DATA_PATH}/{seq_name}/processed/"
    target_dir = f"{DATA_PATH}/{seq_name}/processed/rgbas/"
    subprocess.run(["python", "scripts/object_mask_gen.py", "--seq_name", seq_name, "--source_dir", source_dir, "--target_dir", target_dir, "--rgba_format"])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, required=True, help="Name of the sequence to process")
    parser.add_argument("--cutie_path", type=str, default="./third_party/Cutie", help="Path to the Cutie repository")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--scaler", type=float, default=1.0, help="Downscaling factor for images")
    parser.add_argument("--crop_size", type=str, default="640,480", help="Crop size as width,height (e.g., 640,480)")
    args = parser.parse_args()


    main(args)