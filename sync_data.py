import os
source_data_dir="/home/simba/Documents/dataset/MagicHOI/data_iccv_submit"
des_data_dir="data_release"

source_result_dir="/home/simba/Documents/MagicHOI/outputs_iccv_release/[105][104][7_29][fix_evaluation]"
des_result_dir="outputs_release"


all_sequences = [
    # ############# HO3D_v3_HOLD ICCV dataset #############    
    "hold_ABF12_ho3d.180", # mask align error
    "hold_ABF14_ho3d.180",
    "hold_GPMF12_ho3d.90", # not frimly grasp from 90~92
    "hold_GPMF14_ho3d.90",
    "hold_MC1_ho3d.0",
    "hold_MC4_ho3d.0", # object scale fail in frame 24, mask align error
    "hold_MDF12_ho3d.60", # object project is fail
    "hold_MDF14_ho3d.300", # mask align error
    "hold_ShSu10_ho3d.30", # not frimly grasp in 52
    "hold_ShSu12_ho3d.30", # reconstruction too fat
    "hold_SM2_ho3d.90", # mask align error
    "hold_SM4_ho3d.0", # mask align error
    "hold_SMu1_ho3d.0",
    "hold_SMu40_ho3d.0", # only 15 frames for mvs due to textureless, severe occlusion and mask error
]

for sequence in all_sequences:
    seq_name = sequence.split('.')[0]
    print(f"process {sequence}")
    # copy data file
    sub_file = f"{seq_name}/processed/hold_fit.slerp.npy"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)

    # copy data file
    sub_file = f"{seq_name}/processed/j2d.full.npy"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)    

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_2d/keypoints.npy"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd) 

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_{sequence}/intrinsic.npy"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)     

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_{sequence}/sfm_superpoint+superglue/mvs/fused_normalized.ply"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)   

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_{sequence}/sfm_superpoint+superglue/mvs/fused_normalized_aligned.ply"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)   

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_{sequence}/sfm_superpoint+superglue/mvs/o2w_normalized.npy"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)     

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_{sequence}/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)      

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_{sequence}/sfm_superpoint+superglue/mvs/sparse_points_normalized.ply"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)      

    # copy data file
    sub_file = f"{seq_name}/processed/colmap_{sequence}/sfm_superpoint+superglue/mvs/sparse_points_normalized_aligned.ply"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_file))} && "
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_file)}" "{os.path.join(des_data_dir, sub_file)}"'''
    print(cmd)
    os.system(cmd)                                   
    
    
    # copy data folder
    sub_folder = f"{seq_name}/processed/images"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_folder))} && "
    cmd += f'''rm "{os.path.join(des_data_dir, sub_folder)}" -rf && '''
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_folder)}" "{os.path.join(des_data_dir, os.path.dirname(sub_folder))}"'''
    print(cmd)
    os.system(cmd)   

    # copy data folder
    sub_folder = f"{seq_name}/processed/masks"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_folder))} && "
    cmd += f'''rm "{os.path.join(des_data_dir, sub_folder)}" -rf && '''
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_folder)}" "{os.path.join(des_data_dir, os.path.dirname(sub_folder))}"'''
    print(cmd)
    os.system(cmd)    

    # copy data folder
    sub_folder = f"{seq_name}/processed/rgbas"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_folder))} && "
    cmd += f'''rm "{os.path.join(des_data_dir, sub_folder)}" -rf && '''
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_folder)}" "{os.path.join(des_data_dir, os.path.dirname(sub_folder))}"'''
    print(cmd)
    os.system(cmd)  

    # copy data folder
    sub_folder = f"{seq_name}/processed/inpaint"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_folder))} && "
    cmd += f'''rm "{os.path.join(des_data_dir, sub_folder)}" -rf && '''
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_folder)}" "{os.path.join(des_data_dir, os.path.dirname(sub_folder))}"'''
    print(cmd)
    os.system(cmd)  

    # copy data folder
    sub_folder = f"{seq_name}/processed/colmap_{sequence}/HO3D"
    cmd = ""
    cmd += f"mkdir -p {os.path.join(des_data_dir, os.path.dirname(sub_folder))} && "
    cmd += f'''rm "{os.path.join(des_data_dir, sub_folder)}" -rf && '''
    cmd += f'''rsync -azvp "{os.path.join(source_data_dir, sub_folder)}" "{os.path.join(des_data_dir, os.path.dirname(sub_folder))}"'''
    print(cmd)
    os.system(cmd)               
    


    # # copy result file
    # sub_file = f"{sequence}/3d_ref_weight/hold_fit.aligned.npy"
    # cmd = ""
    # cmd += f"mkdir -p {os.path.join(des_result_dir, os.path.dirname(sub_file))} && "
    # cmd += f'''rsync -azvp "{os.path.join(source_result_dir, sub_file)}" "{os.path.join(des_result_dir, sub_file)}"'''
    # print(cmd)
    # os.system(cmd)            
    

    # # copy result file
    # sub_file = f"{sequence}/3d_ref_weight/mano_fit_ckpt/ho/last.ckpt"
    # cmd = ""
    # cmd += f"mkdir -p {os.path.join(des_result_dir, os.path.dirname(sub_file))} && "
    # cmd += f'''rsync -azvp "{os.path.join(source_result_dir, sub_file)}" "{os.path.join(des_result_dir, sub_file)}"'''
    # print(cmd)
    # os.system(cmd)     
    
    # # copy result folder
    # sub_folder = f"{sequence}/3d_ref_weight/save/it3000-export"
    # cmd = ""
    # cmd += f"mkdir -p {os.path.join(des_result_dir, os.path.dirname(sub_folder))} && "
    # cmd += f'''rm "{os.path.join(des_data_dir, sub_folder)}" -rf && '''
    # cmd += f'''rsync -azvp "{os.path.join(source_result_dir, sub_folder)}" "{os.path.join(des_result_dir, os.path.dirname(sub_folder))}"'''
    # print(cmd)
    # os.system(cmd)  