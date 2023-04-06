import os
HO3D_v3_HOLD_data_path = f"{os.getcwd()}/data"

sequences = [
    {
    # reconstruction OK after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_ABF12_ho3d.180",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 206,  
    "selected_image": 206,  
    "consecutive_frame_star": 180,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    "id": "hold_ABF14_ho3d.180",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 206,  
    "selected_image": 206,  
    "consecutive_frame_star": 180,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # the same after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_GPMF12_ho3d.90",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 101,  
    "selected_image": 101,  
    "consecutive_frame_star": 90,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # the same after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_GPMF14_ho3d.90",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 94,  
    "selected_image": 94,  
    "consecutive_frame_star": 90,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    #the same after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_MC1_ho3d.0",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 12,  
    "selected_image": 12,  
    "consecutive_frame_star": 0,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # no pose error for back views after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_MC4_ho3d.0",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 12,  
    "selected_image": 12,  
    "consecutive_frame_star": 0,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # the same after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    # However, MVS reconstruction is not good due to the textureless handler which cause wrong correspondences
    "id": "hold_MDF12_ho3d.60",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 89,  
    "selected_image": 89,  
    "consecutive_frame_star": 60,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    "id": "hold_MDF14_ho3d.300",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 329,  
    "selected_image": 329,  
    "consecutive_frame_star": 300,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # the same after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_ShSu10_ho3d.30",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 45,  
    "selected_image": 45,  
    "consecutive_frame_star": 30,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    "id": "hold_ShSu12_ho3d.30",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 53,  
    "selected_image": 53,  
    "consecutive_frame_star": 30,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # less noise after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_SM2_ho3d.90",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 109,  
    "selected_image": 109,  
    "consecutive_frame_star": 90,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # the same after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_SM4_ho3d.0",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 7,  
    "selected_image": 7,  
    "consecutive_frame_star": 0,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    }, 
    {
    # the same after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_SMu1_ho3d.0",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 17,  
    "selected_image": 17,  
    "consecutive_frame_star": 0,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },
    {
    # more accurate pose and reconstruction result after increase matching scores to 0.5 from 0.3 and retrival image number to top 50%
    "id": "hold_SMu40_ho3d.0",
    "cond_select_strategy": "object_hand_ratio", # object_hand_ratio or object_pixel_mask or manual
    "cond_image": 16,  
    "selected_image": 16,  
    "consecutive_frame_star": 0,
    "consecutive_frame_num": 30,
    "consecutive_frame_interval": 1,
    "data_path": HO3D_v3_HOLD_data_path,
    "cond_pose_from_selected_cam": False,
    },                                                          
]