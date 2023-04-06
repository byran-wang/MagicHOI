
import os
import argparse
import json
import sys
sys.path.append("third_party/utils_simba")
from utils_simba.misc import merge_json
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="gpu index")
parser.add_argument("--CFG_scale", default=5.0, help="classifier free scale")
parser.add_argument("--elevation_deg", default=30, help="elevation for condition image")
parser.add_argument("--azimuth_deg", default=-30, help="azimuth for condition image")
parser.add_argument("--camera_distance", default=2, help="camera distance for condition image")
parser.add_argument("--fovy_deg", default=41.15, help="fovy for condition image")
parser.add_argument("--out_dir", default="outputs", help="output directory")
parser.add_argument("--mute", action="store_true", help="mute the output")
parser.add_argument("--debug", action="store_true", help="debug the output")
parser.add_argument("--rebuild", action="store_true", help="rebuild the output")
parser.add_argument("--vis", action="store_true", help="visulize the output")
parser.add_argument("--vis_ait", action="store_true", help="visulize the output")
parser.add_argument("--f", action="store_true", help="quick test")
parser.add_argument("--cond_select_strategy", default="manual", help="choise: manual, object_hand_ratio, object_pixel_max")
parser.add_argument("--isosurface_resolution", default=256, help="isosurface resolution choice: 32, 64, 128, 256, 400; check in load/tets")

parser.add_argument('--execute_list', 
    choices=[
            "only_3d",
            "3d_ref",
            "3d_ref_weight",
            "only_ref",            
            ], 
    help="Specify the execution option.", 
    nargs='+',  # To accept multiple values in a list
    required=False  # This makes the argument mandatory
)

parser.add_argument('--process_list', 
    choices=[
            "vis_ait",
            "rm", 
            "train", 
            "export",
            "validate",
            "gen_cond_depth", 
            "align",
            "save_align",
            "align_hand_object_h",
            "align_hand_object_r",
            "align_hand_object_o",
            "align_hand_object_ho",
            "eval_step_ho_pose_refine",
            "eval_summary_ho",            
             ],
    help="Specify the process option.", 
    nargs='+',  # To accept multiple values in a list
    required=False  # This makes the argument mandatory
)

all_sequences = [
    # ############# HO3D_v3_HOLD ICCV dataset #############    
    "hold_ABF12_ho3d.180",
    "hold_ABF14_ho3d.180",
    "hold_GPMF12_ho3d.90",
    "hold_GPMF14_ho3d.90",
    "hold_MC1_ho3d.0",
    "hold_MC4_ho3d.0",
    "hold_MDF12_ho3d.60",
    "hold_MDF14_ho3d.300",
    "hold_ShSu10_ho3d.30",
    "hold_ShSu12_ho3d.30",
    "hold_SM2_ho3d.90",
    "hold_SM4_ho3d.0",
    "hold_SMu1_ho3d.0",
    "hold_SMu40_ho3d.0",
]

parser.add_argument('--seq_list',
    choices=all_sequences + ['all'],
    help="Specify the sequence list. Use 'all' to select all sequences.",
    nargs='+',  # To accept multiple values in a list
    required=True  # This makes the argument mandatory
)

args = parser.parse_args()

if 'all' in args.seq_list:
    selected_sequences = all_sequences
else:
    selected_sequences = args.seq_list

process_list = args.process_list
excecute_list = args.execute_list

current_path = os.getcwd()
path_prefix = "processed"

def export_cond_cam(args, cam_f):
    cam = {}
    cam['elevation_deg'] = args.elevation_deg
    cam['azimuth_deg'] = args.azimuth_deg
    cam['fovy_deg'] = args.fovy_deg
    cam['distance'] = args.camera_distance

    with open(cam_f, 'w') as f:
        json.dump(cam, f, indent=4)

from sequence_config import sequences
for seq in sequences:
    if seq["id"] not in selected_sequences:
        continue    
    seq_name = seq['id'].split(".")[0]
    prefix = ""
    trainer_max_steps_only_ref = 1000
    trainer_max_steps_only_3d = 1000
    trainer_max_steps_3d_ref = 2000
    trainer_max_steps_3d_ref_weight = 3000
    trainer_3d_ref_only_ref = 1000
    random_camera_resolution_milestones = [400, 700]

    for exe in excecute_list:

        # get the max_steps for subsequence process
        if exe == "only_3d":
            max_steps = trainer_max_steps_only_3d
        elif exe == "only_ref":
            max_steps = trainer_max_steps_only_ref
        elif exe == "3d_ref":
            max_steps = trainer_max_steps_3d_ref
        elif exe == "3d_ref_weight":
            max_steps = trainer_max_steps_3d_ref_weight
        else:
            assert False, "Invalid exe"          
        output_name = f"{seq['id']}"
        tag = f"{prefix}{exe}"
        output_path = f"{current_path}/{args.out_dir}/{output_name}/{tag}"
        align_path = f"{seq['data_path']}/{seq_name}/{path_prefix}/align_{seq['id']}"
        colmap_camera_path = f"{seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/HO3D/cameras"
        if seq.get("consecutive_frame_num", 0) > 0:
            start_frame = seq['consecutive_frame_star']
            ref_views = list(range(start_frame, start_frame + seq['consecutive_frame_num'] * seq['consecutive_frame_interval'], seq['consecutive_frame_interval']))
        else:
            ref_views = seq['ref_views']        

        if "vis_ait" in args.process_list:
            cmd = ""
            cmd += f" python code/visualize_ckpt_magicHOI.py "
            cmd += f" --ours " #--gt_ho3d --headless --out_folder {current_path}/outputs/{seq['id']}/results
            cmd += f" --seq_name {seq_name} "
            cmd += f" --ckpt_p {current_path}/outputs/{seq['id']}/{exe}/mano_fit_ckpt/ho/last.ckpt "
            cmd += f" --data_root data/{seq_name}/processed/ "
            cmd += f" --obj_mesh_f {current_path}/outputs/{seq['id']}/{exe}/save/it{max_steps}-export/model.obj "
            cmd += f" --obj_pose_f data/{seq_name}/processed/colmap_{seq['id']}/sfm_superpoint+superglue/mvs//o2w_normalized_aligned.npy "
            print(f"{cmd}")
            os.system(cmd)


        if "rm" in process_list:
            cmd = ""
            cmd = f"rm -rf {output_path} "
            print(f"{cmd}")
            os.system(cmd)

        if "train" in process_list:
            cmd = ""
            cmd = f"python launch.py --config configs/zero123.yaml --train --gpu {args.gpu}  use_timestamp=False name={output_name} tag={tag} "
            if exe == "3d_ref":
                cmd += f" resume={current_path}/{args.out_dir}/{output_name}/{prefix}only_ref/ckpts/last.ckpt "
            elif exe == "3d_ref_weight":
                cmd += f" resume={current_path}/{args.out_dir}/{output_name}/{prefix}3d_ref/ckpts/last.ckpt "
            cmd += f" exp_root_dir={args.out_dir} "
            cmd += f" data.real_image_foundation_pose.scene={seq_name} "
            cmd += f" data.real_image_foundation_pose.selected_ind={seq['selected_image']} "            
            cmd += f" data.real_image_foundation_pose.data_dir={seq['data_path']} "
            cmd += f" data.real_image_foundation_pose.rgb_path={seq['data_path']}/{seq_name}/{path_prefix}/images "
            # cmd += f" data.real_image_foundation_pose.rgba_path={seq['data_path']}/{seq_name}/{path_prefix}/rgbas_hand "
            cmd += f" data.real_image_foundation_pose.rgba_path={seq['data_path']}/{seq_name}/{path_prefix}/rgbas "
            cmd += f" data.real_image_foundation_pose.camera_path={colmap_camera_path} "
            cmd += f" data.real_image_foundation_pose.camera_align_path={seq['data_path']}/{seq_name}/{path_prefix}/cameras_pred "
            cmd += f" data.real_image_foundation_pose.mask_path={seq['data_path']}/{seq_name}/{path_prefix}/masks "
            cmd += f" data.real_image_foundation_pose.inpaint_f={seq['data_path']}/{seq_name}/{path_prefix}/inpaint/{seq['cond_image']:04d}_rgba_center.png "
            cmd += f" data.real_image_foundation_pose.selected_pose={align_path}/{seq['selected_image']:04d}.json "
            cmd += f" data.cond_pose_from_selected_cam={seq['cond_pose_from_selected_cam']} "
            cmd += f" data.default_azimuth_deg={args.azimuth_deg} "
            cmd += f" data.default_camera_distance={args.camera_distance} "
            cmd += f" data.default_fovy_deg={args.fovy_deg} "  
            cmd += f" data.real_image_foundation_pose.ref_views=\"{ref_views}\" "
            cmd += f" data.HOLD.data_dir={seq['data_path']}/{seq_name}/build/ "
            cmd += f" data.HOLD.preprocess_dir={seq['data_path']}/{seq_name}/{path_prefix}/ "
            cmd += f" data.HOLD.log_dir={output_path} "
            cmd += f" data.HOLD.MANO_dir={current_path}/body_models/ "
            cmd += f" data.colmap_dir={seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/ "
            cmd += f" data.train_mode={exe} "
            cmd += f" system.guidance.guidance_scale={args.CFG_scale} "
            cmd += f" system.hand.data_dir=data/{seq_name}/ "


            cmd += f" data.viz.show_scene=False data.viz.pause=False "
            cmd += f" data.random_camera.viz.show_scene=False data.random_camera.viz.pause=False "
            # cmd += f" resume={args.out_dir}/{output_name}/{tag}/ckpts/last.ckpt "
            

            cmd += f" data.default_elevation_deg={args.elevation_deg} "
            
            cmd += f" trainer.max_steps={max_steps} "
            if exe == "only_ref":
                cmd += f" system.freq.do_ref_only=True data.align.do_3d_guidance_only=False "            
            elif exe == "only_3d":
                cmd += f" system.freq.do_ref_only=False data.align.do_3d_guidance_only=True "
                cmd += f" system.freq.ref_only_steps=-1 "
                cmd += f" 'data.random_camera.resolution_milestones=[{random_camera_resolution_milestones[0]}, {random_camera_resolution_milestones[1]}]' "
            elif exe == "3d_ref" or exe == "3d_ref_weight":
                cmd += f" system.freq.do_ref_only=False data.align.do_3d_guidance_only=False "

                cmd += f" system.freq.ref_only_steps={trainer_3d_ref_only_ref} "
                mile_start = trainer_3d_ref_only_ref + random_camera_resolution_milestones[0]
                mile_end = trainer_3d_ref_only_ref + random_camera_resolution_milestones[1]
                cmd += f" 'data.random_camera.resolution_milestones=[{mile_start}, {mile_end}]' "
                cmd += f" 'system.guidance.min_step_percent=[{trainer_3d_ref_only_ref}, 0.4, 0.2, {trainer_max_steps_3d_ref}]' "
                cmd += f" 'system.guidance.max_step_percent=[{trainer_3d_ref_only_ref}, 0.85, 0.5, {trainer_max_steps_3d_ref}]' "
            else:
                assert False, "Invalid exe"
            
            if exe == "3d_ref_weight":
                cmd += f" data.random_camera.visibility_mesh_f={current_path}/{args.out_dir}/{seq['id']}/3d_ref/visibility/mesh_with_visibility.ply "

            cmd += f" || true"
            print(f"{cmd}")
            os.system(cmd)
        if "export" in process_list:
            cmd = ""
            cmd = f"python launch.py --config {output_path}/configs/parsed.yaml --export --gpu {args.gpu} resume={output_path}/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.context_type=cuda"
            cmd += f" system.geometry.isosurface_resolution={args.isosurface_resolution} "
            cmd += f" system.geometry.isosurface_method=mt "
            cmd += f" system.geometry.radius=0.8 "
            cmd += f" system.geometry.isosurface_coarse_to_fine=True " # hard set bbox to [-0.8, 0.8]
            cmd += f" || true"
            print(f"{cmd}")
            os.system(cmd)
            if exe == "only_3d":
                cam_f = f"{output_path}/save/it{trainer_max_steps_only_3d}-export/camera.json"
                export_cond_cam(args, cam_f)

        if "validate" in process_list:
            if args.rebuild:
                cmd = ""
                cmd += f" cd {output_path}/ && "
                cmd += f" rm -rf visibility "
                print(f"{cmd}")
                os.system(cmd)
                
            cmd = ""
            cmd = f"python launch.py --config {output_path}/configs/parsed.yaml --validate --gpu {args.gpu} resume={output_path}/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.context_type=cuda"
            cmd += f" data.viz.save_cond_point_cloud=True "
            if exe == "3d_ref":
                cmd += f" system.weight.gen_visibility_mesh=True "
            cmd += f" || true"
            print(f"{cmd}")
            os.system(cmd)

        if "gen_cond_depth" in process_list:
            cmd = ""
            cmd += f" python code/gen_depth.py "
            cmd += f" --seqence_name {seq_name} "
            cmd += f" --in_dir {output_path}/save/it{trainer_max_steps_only_3d}-export/ "
            cmd += f" --out_dir {output_path}/cond_depth/ "
            if args.mute:
                cmd += f" --mute "
            cmd += f" || true"
            print(f"{cmd}")
            os.system(cmd)              

        if "align" in process_list:
            if args.rebuild:
                cmd = ""
                cmd += f" cd {seq['data_path']}/ && "
                cmd += f" rm -rf {seq_name}/{path_prefix}/align_{seq['id']}/ {output_path}/features/  {output_path}/align/"
                print(f"{cmd}")
                os.system(cmd)
            
            cmd = ""
            cmd += f"python code/align_corres.py "
            cmd += f" --cond_img_f {seq['data_path']}/{seq_name}/{path_prefix}/inpaint/{seq['cond_image']:04d}_rgba_center.png "
            cmd += f" --cond_depth_f {output_path}/cond_depth/depth.png "
            cmd += f" --cond_camera_f {output_path}/cond_depth/camera.json "

            cmd += f" --query_img_f {seq['data_path']}/{seq_name}/{path_prefix}/rgbas/{seq['selected_image']:04d}.png "
            cmd += f" --query_depth_f {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/HO3D/depths/{seq['selected_image']:04d}.png "
            cmd += f" --query_camera_f {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/HO3D/cameras/{seq['selected_image']:04d}.json "
            cmd += f" --in_dir ./ "
            cmd += f" --out_dir {output_path}/features/ "
            if args.mute:
                cmd += f" --mute "
            cmd += f" || true"
            print(cmd)
            os.system(cmd)

            cmd = ""
            cmd += f" python code/align_pcs.py " 
            cmd += f" --data_dir {output_path}/features/ "
            cmd += f" --query_index {seq['selected_image']:04d} "
            cmd += f" --seqence_name {seq_name} "
            cmd += f" --cond_camera_f {output_path}/cond_depth/camera.json "
            cmd += f" --cond_img_f {seq['data_path']}/{seq_name}/{path_prefix}/inpaint/{seq['cond_image']:04d}_rgba_center.png "
            cmd += f" --out_dir {output_path}/align/ "
            if args.mute:
                cmd += f" --mute "
            cmd += f" || true"
            print(cmd)
            os.system(cmd)
            merge_json(f"{seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/HO3D/cameras/{seq['selected_image']:04d}.json", f"{output_path}/align/{seq['selected_image']:04d}.json", f"{output_path}/align/{seq['selected_image']:04d}.json")   
            
            cmd = ""
            cmd += f" rm {seq['data_path']}/{seq_name}/{path_prefix}/align_{seq['id']}/ -rf && "
            cmd += f" cp {output_path}/align {seq['data_path']}/{seq_name}/{path_prefix}/align_{seq['id']} -rf "
            print(f"{cmd}")
            os.system(cmd)

        if "save_align" in process_list:         
            cmd = ""
            cmd += " python code/save_aligned_poses.py "
            cmd += f" --poses_normalized_f {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/sfm_superpoint+superglue/mvs/o2w_normalized.npy "
            cmd += f" --aligned_pose_f {seq['data_path']}/{seq_name}/{path_prefix}/align_{seq['id']}/{seq['selected_image']:04d}.json "            
            cmd += f" --sparse_pts_normalized_f {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/sfm_superpoint+superglue/mvs/sparse_points_normalized.ply "
            cmd += f" --dense_pts_normalized_f {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/sfm_superpoint+superglue/mvs/fused_normalized.ply "
            cmd += f" --out_dir {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/sfm_superpoint+superglue/mvs/ "
            print(f"{cmd}")
            os.system(cmd)

        def run_align_hand_object(mode, seq_name, seq, exe, args, current_path,  path_prefix):
            if args.rebuild:
                cmd = ""
                cmd += f" rm -rf {current_path}/outputs/{seq['id']}/{exe}/mano_fit_ckpt/{mode}/ "
                print(f"{cmd}")
                os.system(cmd)


            cmd = ""
            cmd += f" python code/align_hands_object.py "
            cmd += f" --seq_name {seq_name} "
            cmd += f" --colmap_path {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/ "
            cmd += f" --object_dir /home/simba/Documents/project/diff_object/threestudio/outputs/{seq['id']}/{exe}/ "
            cmd += f" --data_path {seq['data_path']}/{seq_name}/{path_prefix}/{exe}/ "
            cmd += f" --colmap_k "
            cmd += f" --mode {mode} "
            cmd += f" --out_dir {current_path}/outputs/{seq['id']}/{exe}/ "
            print(f"{cmd}")
            os.system(cmd)

        for mode in ["before", "h", "r", "o", "ho"]: # only save 
            if f"align_hand_object_{mode}" in process_list:
                run_align_hand_object(mode, seq_name, seq, exe, args, current_path, path_prefix)

        def run_eval_step(seq, exe, prefix, max_steps, current_path, args, stage):
            """Run evaluation for a specific stage (before/after pose refinement)"""
            metrics_dir = f"metrics_{stage}_pose_refine"
            # ckpt_dir = "before" if stage == "before" else "ho"
            ckpt_dir = stage
            
            if args.rebuild:
                cmd = ""
                cmd = f"rm -rf {current_path}/outputs/{seq['id']}/{prefix}{exe}/{metrics_dir}/"
                print(cmd)
                os.system(cmd)

            cmd = f"python code/evaluate_step.py"
            cmd += f" --seq_name {seq['id']}"
            cmd += f" --data_root {current_path}/outputs/{seq['id']}/{exe}"
            cmd += f" --mvs_root {seq['data_path']}/{seq_name}/{path_prefix}/colmap_{seq['id']}/sfm_superpoint+superglue/mvs/"
            cmd += f" --object_mesh_f {current_path}/outputs/{seq['id']}/{prefix}{exe}/save/it{max_steps}-export/model.obj"
            cmd += f" --sd_p {current_path}/outputs/{seq['id']}/{exe}/mano_fit_ckpt/{ckpt_dir}/last.ckpt"
            cmd += f" --MANO_f {current_path}/code/body_models/MANO_RIGHT.pkl"
            cmd += f" --out_dir {current_path}/outputs/{seq['id']}/{prefix}{exe}/{metrics_dir}/"
            if stage == "h":
                cmd += f" --only_eval_hand "
            cmd += " || true"
            print(cmd)
            os.system(cmd)

        for stage in ["ho"]:
            if f"eval_step_{stage}_pose_refine" in process_list:
                run_eval_step(seq, exe, prefix, max_steps, current_path, args, stage)                                        
 

        def run_eval_summary(seq, exe, prefix, max_steps, current_path, args, stage):
            """Run evaluation for a specific stage (before/after pose refinement)"""
            metrics_dir = f"metrics_{stage}_pose_refine"
            # ckpt_dir = "before" if stage == "before" else "ho"
            output_dir = f"{current_path}/outputs/metrics_summary/"
            
            if args.rebuild:
                cmd = ""
                cmd = f"rm -rf {output_dir}/metrics_{stage}_pose_refine_results.txt"
                print(cmd)
                os.system(cmd)
            
            cmd = ""
            cmd += f"cd {current_path} && python extract_jsons.py"
            cmd += f" --parent_dir outputs "
            cmd += f" --metric_folder {metrics_dir} "
            cmd += f" --out_dir {output_dir} "
            cmd += " || true"
            print(cmd)
            os.system(cmd)  
        
        for stage in ["ho"]:
            if f"eval_summary_{stage}" in process_list:
                run_eval_summary(seq, exe, prefix, max_steps, current_path, args, stage)