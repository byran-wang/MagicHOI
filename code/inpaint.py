import os
import sys
import subprocess
import numpy as np
import cv2
import json
from pathlib import Path
import argparse

def inpaint_input_views(config, do_inpaint=True, do_mask=True, do_center=True, write_pose=True):
    if config.inpaint_select_strategy == "manual":
        inpaint_f = Path(config.image_dir) / f"{config.cond_view:04d}.png"
    else:
        max_value = 0
        hand_bbox_f = Path(config.image_dir) / "../boxes.npy"
        hand_bboxes = np.load(hand_bbox_f)
        valid_index = -1
        for i, ref_view in enumerate(config.ref_views):
            image_f = Path(config.image_dir) / f"{ref_view:04d}.png"
            # check image_f exists
            if not image_f.exists():
                continue
            valid_index += 1
            mask_f = image_f.parent.parent / "masks" / f"{ref_view:04d}.png"
            mask = cv2.imread(str(mask_f), cv2.IMREAD_UNCHANGED)
            SEGM_IDS = {"bg": 0, "object": 50, "right": 150, "left": 250}
            object_pixels = (mask == SEGM_IDS["object"]).sum()
            hand_pixels = (mask == SEGM_IDS["right"]).sum()
            if config.inpaint_select_strategy == "object_hand_ratio":
                # object_hand_ratio = object_pixels / (hand_pixels in the bbox range)
                # hand_bboxes is a list of bboxes, each bbox is a tuple (x1, y1, x2, y2)
                hand_bbox = hand_bboxes[valid_index].copy()
                # clip the bbox x1, x2 to image width, y1, y2 to image height
                mask_height, mask_width = mask.shape[:2]
                hand_bbox[0] = np.clip(hand_bbox[0], 0, mask_width - 1)
                hand_bbox[2] = np.clip(hand_bbox[2], 0, mask_width - 1)
                hand_bbox[1] = np.clip(hand_bbox[1], 0, mask_height - 1)
                hand_bbox[3] = np.clip(hand_bbox[3], 0, mask_height - 1)
                hand_bbox = hand_bbox.astype(np.int32)
                hand_pixels_in_bbox = (mask[hand_bbox[1]:hand_bbox[3], hand_bbox[0]:hand_bbox[2]] == SEGM_IDS["right"]).sum()
                # show the mask and the bbox by plotting
                # show_mask_and_bbox(mask, hand_bbox)
                object_hand_ratio = object_pixels / hand_pixels_in_bbox
                if object_hand_ratio > max_value:
                    max_value = object_hand_ratio
                    inpaint_f = image_f
            elif config.inpaint_select_strategy == "object_pixel_max":
                if object_pixels > max_value:
                    max_value = object_pixels
                    inpaint_f = image_f
            else:
                # assert and print error message
                assert False, "Unknown inpaint_select_strategy"
    print(f"Selected inpaint view: {inpaint_f}")
    # save the inpaint file index to config.inpaint_select_strategy.txt
    inpaint_f_index = os.path.basename(inpaint_f).split(".")[0].split("_rgba")[0]
    inpaint_f_index_f = f"{config.out_dir}/{config.inpaint_select_strategy}_selected.txt"
    os.makedirs(os.path.dirname(inpaint_f_index_f), exist_ok=True)
    with open(inpaint_f_index_f, "w") as f:
        f.write(f"{inpaint_f_index}")

    InpaintAny_dir = "./third_party/Inpaint-Anything"
    InpaintAny_py = "/home/simba/anaconda3/envs/chatcap/bin/python"
    InpaintAny_script = "remove_anything.py"

    Cutie_dir = "./third_party/Cutie"
    Cutie_py = "/home/simba/anaconda3/envs/py38cu118/bin/python"
    Cutie_script = "interactive_demo.py"
    Cutie_workspace_dir = os.path.join(Cutie_dir, "workspace")
    # for i, inpaint_f in enumerate(inpaint_views):
    if 1:
        out_dir = config.out_dir
        # cameras_dir = os.path.join(os.path.dirname(inpaint_view), "../cameras")
        os.makedirs(out_dir, exist_ok=True)
        image_name = os.path.basename(inpaint_f).split(".")[0]

        inpaint_video_file = os.path.join(out_dir, image_name, "image", image_name + ".mp4")
        inpaint_mask_file = os.path.join(out_dir, image_name, "mask", f"{image_name}.png")
        inpaint_image_file = os.path.join(out_dir, image_name, "image", f"{image_name}.png")

        masked_ip_rgba_f = os.path.join(out_dir, f"{image_name}_rgba.png")       

        if do_inpaint:
            inpainting_command = [
                InpaintAny_py, InpaintAny_script,
                "--input_img", inpaint_f,
                "--coords_type", "click",
                "--point_coords", "200", "450",
                "--point_labels", "1",
                "--dilate_kernel_size", "15",
                "--output_dir", out_dir,
                "--sam_model_type", "vit_t",
                "--sam_ckpt", "./weights/mobile_sam.pt",
                "--lama_config", "./lama/configs/prediction/default.yaml",
                "--lama_ckpt", "./pretrained_models/big-lama"
            ]
            
            subprocess.run(inpainting_command, check=True, cwd=InpaintAny_dir)
            print(f"Finished inpainting {inpaint_f}")
            print(f"Output saved to {out_dir}/{image_name}")
            selected_number = input("Please enter inpaint selected number [0, 1 or 2]:")
            inpaint_file = os.path.join(out_dir, image_name, f"inpainted_with_mask_{selected_number}.png")
            # Create directories
            inpaint_image_dir = os.path.join(os.path.dirname(inpaint_file), "image")
            inpaint_mask_dir = os.path.join(os.path.dirname(inpaint_file), "mask")
            os.makedirs(inpaint_image_dir, exist_ok=True)
            os.makedirs(inpaint_mask_dir, exist_ok=True)
            subprocess.run(["cp", inpaint_file, inpaint_image_file], check=True)

            # Create video from images using ffmpeg
            command = [
                '/usr/bin/ffmpeg -y', '-framerate', '5', '-pattern_type', 'glob', '-i', '\"./*.png\"',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', '\"pad=ceil(iw/2)*2:ceil(ih/2)*2\"',
                f"{image_name}.mp4"
            ]
            shell_command = ' '.join(command)
            subprocess.run(shell_command, shell=True, check=True, cwd=inpaint_image_dir)

        if do_mask:
            # Run Cutie interactive demo
            command = [
                Cutie_py, 
                Cutie_script,
                "--video", inpaint_video_file, 
                "--num_objects", "1"
            ]
            subprocess.run(["rm", "-rf", os.path.join(Cutie_workspace_dir, f"{image_name}")], check=True)
            subprocess.run(command, check=True, cwd=Cutie_dir)     
            cutie_mask_file = os.path.join(Cutie_workspace_dir, f"{image_name}", "binary_masks", "0000000.png")
            # Copy the generated mask
            subprocess.run(["cp", "-rf", cutie_mask_file, inpaint_mask_file], check=True)


            inpaint_image = cv2.imread(inpaint_image_file)
            mask_image = cv2.imread(inpaint_mask_file, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask[:, :, np.newaxis]
            binary_mask = binary_mask[:inpaint_image.shape[0],...]
            binary_mask = binary_mask[:inpaint_image.shape[0],:inpaint_image.shape[1],:]
            masked_inpaint = inpaint_image * (binary_mask // 255) + 255 * (1 - binary_mask// 255)

            masked_ip_f = os.path.join(out_dir, f"{image_name}.png")
            cv2.imwrite(masked_ip_f, inpaint_image)

            alpha_channel = binary_mask
            rgba_image = cv2.merge((inpaint_image[:, :, 0], inpaint_image[:, :, 1], inpaint_image[:, :, 2], alpha_channel))
            cv2.imwrite(masked_ip_rgba_f, rgba_image)
       
        if do_center:
            inpaint_rgba = cv2.imread(masked_ip_rgba_f, cv2.IMREAD_UNCHANGED)
            desired_size = int(config.inpaint_size * (1 - config.border_ratio))
            # Center the inpainted view
            mask = inpaint_rgba[:, :, 3]
            coords = np.nonzero(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            h = y_max - y_min
            w = x_max - x_min
            scale = desired_size / max(h, w)
            h2 = int(h * scale)
            w2 = int(w * scale)
            y2_min = (config.inpaint_size - h2) // 2
            y2_max = y2_min + h2
            x2_min = (config.inpaint_size - w2) // 2
            x2_max = x2_min + w2
            # mask

            inpaint_rgba_center = np.zeros((config.inpaint_size, config.inpaint_size, 4), dtype=np.uint8)
            inpaint_rgba_center[y2_min:y2_max, x2_min:x2_max] = cv2.resize(inpaint_rgba[y_min:y_max, x_min:x_max], (w2, h2), interpolation=cv2.INTER_AREA)
            inpaint_rgba_center_f = os.path.join(out_dir, f"{image_name}_rgba_center.png")
            cv2.imwrite(inpaint_rgba_center_f, inpaint_rgba_center)

            # # algine optical center and scale
            # align_cx = int((x_min + x_max) / 2)
            # align_cy = int((y_min + y_max) / 2)            
            # all_cameras = sorted(glob(f"{cameras_dir}/*.json"))     
            # for camera_f in all_cameras:
            #     camera = json.load(open(camera_f, 'r'))
            #     camera['K_inpaint'] = copy.deepcopy(camera['K'])
            #     camera['K_inpaint'][0][2] = align_cx
            #     camera['K_inpaint'][1][2] = align_cy
            #     camera['K_manual'] = copy.deepcopy(camera['K'])
            #     camera['K_manual'][0][2] = config.manual_cx']
            #     camera['K_manual'][1][2] = config.manual_cy']
            #     camera['K_half_wh'] = copy.deepcopy(camera['K'])
            #     camera['K_half_wh'][0][2] = camera['width'] // 2
            #     camera['K_half_wh'][1][2] = camera['height'] // 2
            #     align_scale = scale * camera['height'] / config.inpaint_size]
            #     camera['scale_inpaint'] = align_scale
            #     save_json(camera_f, camera)

        if write_pose:
            data = {
                "elevation_deg": config.elevation_deg,
                "azimuth_deg": config.azimuth_deg,
                "fovy_deg": config.fovy_deg,
                "distance": config.distance
            }

            pose_f = os.path.join(out_dir, f"{image_name}.json")
            with open(pose_f, 'w') as f:
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ref_views", type=int, nargs='+', required=True)
    parser.add_argument("--inpaint_select_strategy", type=str, required=True)
    parser.add_argument("--cond_view", type=int, required=True)
    parser.add_argument("--inpaint_size", type=int, default=256)
    parser.add_argument("--border_ratio", type=float, default=0.2)
    parser.add_argument("--elevation_deg", type=float, default=30)
    parser.add_argument("--azimuth_deg", type=float, default=-30)
    parser.add_argument("--fovy_deg", type=float, default=41.15)
    parser.add_argument("--distance", type=float, default=2)
    args = parser.parse_args()
    inpaint_input_views(args)