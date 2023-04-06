import os
import subprocess


# Define the scenes

from pathlib import Path

# Get current working directory
current_dir = Path.cwd()

CUTIE_PATH = f"{current_dir}/third_party/Cutie"
DATA_PATH = f"{current_dir}/data"

CUTIE_CONDA_ENV_PATH = "/home/simba/anaconda3/envs/py38cu118/bin/python"
# SCENES = ["cat_cup", "pig", "dragon"]
# SCENES = [
#           {"name": "pig", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "cat_cup", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "dragon", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "crocodile", "scaler": 0.7, "crop_size": "640,480"},
#           {"name": "baichai", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "banshou", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "bilang", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "blue_moon", "scaler": 0.7, "crop_size": "640,480"},
#           {"name": "bottle1", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "bread", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "driller", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "clamp", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "fan", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "jiaoqiang", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "kettle", "scaler": 0.6, "crop_size": "640,480"},
#           {"name": "libai_cleaner", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "naicha", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "old_phone", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "pig_1", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "pingdiguo", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "toy_banshou", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "toy_car", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "toy_kettle", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "toy_plane", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "wanyongbiao", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "yangtao", "scaler": 0.8, "crop_size": "640,480"},
#           ]

# SCENES = [
#           {"name": "pig_2", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "pig_3", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "pig_4", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "flower_bottle_1", "scaler": 0.7, "crop_size": "640,480"},
#           {"name": "flower_bottle_2", "scaler": 0.7, "crop_size": "640,480"},
#           {"name": "flower_bottle_3", "scaler": 0.7, "crop_size": "640,480"},          
#           {"name": "flower_bottle_4", "scaler": 0.7, "crop_size": "640,480"},
#           {"name": "lufei_1", "scaler": 0.8, "crop_size": "640,480"},          
#           {"name": "lufei_2", "scaler": 0.8, "crop_size": "640,480"},    
#           {"name": "suolong_1", "scaler": 0.8, "crop_size": "640,480"},          
#           {"name": "suolong_2", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "banshou", "scaler": 0.8, "crop_size": "640,480"},                    
#           ]

# SCENES = [
#           {"name": "pig_5", "scaler": 0.6, "crop_size": "640,480"},
#           {"name": "pig_6", "scaler": 0.7, "crop_size": "640,480"},       
#           ]

import os


# mp4_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.mp4')]

# print("MP4 files in the Data directory:")
# # breakpoint()
# for file in mp4_files:
#     print(file)
#     os.makedirs(f"{DATA_PATH}/{file.split('.')[0]}")
#     os.rename(f"{DATA_PATH}/{file}", f"{DATA_PATH}/{file.split('.')[0]}/{file}")


SCENES = [
        #   {"name": "pig_7", "scaler": 0.8, "crop_size": "640,480"},
        #   {"name": "pig_8", "scaler": 0.8, "crop_size": "640,480"},       
        #   {"name": "pig_9", "scaler": 0.8, "crop_size": "640,480"},       
        # {"name": "bottle_1_left_processed", "scaler": 0.7, "crop_size": "480,640"},
        # {"name": "bottle_2_processed", "scaler": 0.6, "crop_size": "480,640"},
        # {"name": "camera_processed", "scaler": 0.8, "crop_size": "640,480"},
        # {"name": "computer_mouse_1_processed", "scaler": 0.7, "crop_size": "640,480"},
        {"name": "controller_1_left_processed", "scaler": 0.6, "crop_size": "480,640"},
        # {"name": "controller_2_left_processed", "scaler": 0.6, "crop_size": "480,640"},

        # {"name": "cube_1_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "cube_2_processed", "scaler": 1.0, "crop_size": "640,480"},
        # {"name": "game_controller_processed", "scaler": 0.6, "crop_size": "640,480"},
        # {"name": "hair_dryer_left_processed", "scaler": 0.4, "crop_size": "480,640"},
        # {"name": "hand_fan_processed", "scaler": 0.6, "crop_size": "640,480"},
        # {"name": "osmo_pocket_left_processed", "scaler": 0.6, "crop_size": "480,640"},
        # {"name": "rubiks_cube_processed", "scaler": 1.0, "crop_size": "640,480"},
        # {"name": "toy_car_1_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_2_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_3_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_4_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_5_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_6_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_7_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_8_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_9_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_10_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_11_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_12_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_13_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_14_left_processed", "scaler": 0.5, "crop_size": "640,480"},
        # {"name": "toy_car_15_left_processed", "scaler": 0.5, "crop_size": "640,480"}, #delete last no hand and left hand pictures
        
        
          ]

# SCENES = [
#           {"name": "camera_1", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "game_controller_1", "scaler": 0.8, "crop_size": "640,480"},
#           {"name": "hand_fan_1", "scaler": 0.8, "crop_size": "640,480"},          
#           ]

# SCENES = [        
#           {"name": "camera_4_processed", "scaler": 0.8, "crop_size": "800,1200"},
#           {"name": "camera_3_processed", "scaler": 0.8, "crop_size": "800,1200"},
#           {"name": "camera_2_left_processed", "scaler": 0.8, "crop_size": "800,1200"},       
#           {"name": "camera_1_left_processed", "scaler": 0.8, "crop_size": "800,1200"},       
#           {"name": "computer_mouse_2_left_processed", "scaler": 0.8, "crop_size": "800,1200"},       
#           {"name": "game_controller_2_left_processed", "scaler": 0.8, "crop_size": "800,1200"},  
#           {"name": "game_controller_1_processed", "scaler": 0.8, "crop_size": "800,1200"},  
#           {"name": "mug_2_left_processed", "scaler": 0.8, "crop_size": "800,1200"},  
#           {"name": "mug_1_processed", "scaler": 0.8, "crop_size": "800,1200"},  
# ]

# SCENES = [        
#           {"name": "hand_fan_1", "scaler": 0.8, "crop_size": "800,1200"},
#           {"name": "hand_big_fan", "scaler": 0.8, "crop_size": "800,1200"},
#           {"name": "hand_driller", "scaler": 0.8, "crop_size": "800,1200"},       
#           {"name": "hand_driller2", "scaler": 0.8, "crop_size": "800,1200"},       

# ]
# SCENES = [        
#           {"name": "face_id_drug_box", "scaler": 1.0, "crop_size": "480,640"},
# ]
# ]
# Loop over each sequence name

for scene in SCENES:
    seq_name = scene["name"]
    scaler = scene["scaler"]
    crop_size = scene["crop_size"]
    # Preprocessing - generate object and hand masks using Cutie
    workspace_object = f"{CUTIE_PATH}/workspace/{seq_name}_object"
    workspace_hand = f"{CUTIE_PATH}/workspace/{seq_name}_hand"
    video_path = f"{DATA_PATH}/{seq_name}/{seq_name}.mp4"
    

    # Generate object mask
    subprocess.run(["rm", "-rf", workspace_object])
    subprocess.run([CUTIE_CONDA_ENV_PATH, "./interactive_demo.py", "--video", video_path, "--workspace", workspace_object, "--num_objects", "1"], cwd=CUTIE_PATH)
    
    # Generate hand mask
    subprocess.run(["rm", "-rf", workspace_hand])
    subprocess.run([CUTIE_CONDA_ENV_PATH, "./interactive_demo.py", "--video", video_path, "--workspace", workspace_hand, "--num_objects", "1"], cwd=CUTIE_PATH)

    # Copy Cutie generated masks and images
    source_dir = f"{CUTIE_PATH}/workspace/{seq_name}_hand/images/"
    target_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/image/"
    subprocess.run(["python", "code/src/mask/cp_mask.py", "--source_dir", source_dir, "--target_dir", target_dir])
    
    source_dir = f"{CUTIE_PATH}/workspace/{seq_name}_hand/binary_masks/"
    target_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_hand/"
    subprocess.run(["python", "code/src/mask/cp_mask.py", "--source_dir", source_dir, "--target_dir", target_dir])
    
    source_dir = f"{CUTIE_PATH}/workspace/{seq_name}_object/binary_masks/"
    target_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_object/"
    subprocess.run(["python", "code/src/mask/cp_mask.py", "--source_dir", source_dir, "--target_dir", target_dir])
    
    # Merge object and hand masks
    out_mask_path = f"{DATA_PATH}/{seq_name}/build_origin_size/mask"
    object_mask_path = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_object"
    hand_mask_path = f"{DATA_PATH}/{seq_name}/build_origin_size/mask_hand"
    subprocess.run(["python", "code/src/mask/merge_mask.py", "--out_mask_path", out_mask_path, "--object_mask_path", object_mask_path, "--hand_mask_path", hand_mask_path])

    # Downscale and crop images
    source_dir = f"{DATA_PATH}/{seq_name}/build_origin_size/"
    target_dir = f"{DATA_PATH}/{seq_name}/build/"
    subprocess.run(["python", "code/src/mask/scale_crop_seq.py", 
                    "--source_dir", source_dir, 
                    "--target_dir", target_dir, 
                    "--sub_folders", "image,mask",
                    "--scaler", str(scaler),
                    "--crop_size", crop_size])