import argparse
import numpy as np
import json
import sys
sys.path.append("third_party/utils_simba")
from utils_simba.pose import get_relative_trans, scale_t
from utils_simba.geometry import convert_point_cloud
import os
def main(args) -> None:
    
    sparse_pts_normalized_f = args.sparse_pts_normalized_f
    dense_pts_normalized_f = args.dense_pts_normalized_f
    poses_normalized_f = args.poses_normalized_f
    aligned_pose_f = args.aligned_pose_f
    out_dir = args.out_dir
    
    aligned_poses_f = f"{out_dir}/{os.path.basename(poses_normalized_f).split('.')[0]}_aligned.npy"
    aligned_sparse_pts_f = f"{out_dir}/{os.path.basename(sparse_pts_normalized_f).split('.')[0]}_aligned.ply"
    aligned_dense_pts_f = f"{out_dir}/{os.path.basename(dense_pts_normalized_f).split('.')[0]}_aligned.ply"
    aligned_mat_f = f"{out_dir}/mat_aligned.npy"

    with open(aligned_pose_f, 'r') as f:
        aligned_info = json.load(f)
        alignedC2Zero123C_T = np.array(aligned_info["que_blw2con_blw_T"])
        alignedC2Zero123C_scale = aligned_info["que_blw2con_blw_scale"][0]
    

    normalized_o2c = np.load(poses_normalized_f)
    normalized_c2o = np.linalg.inv(normalized_o2c)
    

    c2o4x4_relative = get_relative_trans(normalized_c2o, np.eye(4))
    c2o4x4_scale = scale_t(c2o4x4_relative, alignedC2Zero123C_scale)
    aligned_c2o = alignedC2Zero123C_T @ c2o4x4_scale
    aligned_o2c = np.linalg.inv(aligned_c2o)
    np.save(aligned_poses_f, aligned_o2c)

    scale_mat = np.eye(4) * alignedC2Zero123C_scale
    scale_mat[3,3] = 1
    align_mat = alignedC2Zero123C_T @ scale_mat
    convert_point_cloud(sparse_pts_normalized_f,aligned_sparse_pts_f, align_mat)
    convert_point_cloud(dense_pts_normalized_f,aligned_dense_pts_f, align_mat)
    np.save(aligned_mat_f, align_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_pts_normalized_f', type=str, default='', help='a file to normalized sparse points')
    parser.add_argument('--dense_pts_normalized_f', type=str, default='', help='a file to dense sparse points')
    parser.add_argument('--poses_normalized_f', type=str, default='', help='a file to normalized poses from object to camera')
    parser.add_argument('--aligned_pose_f', type=str, default='', help='a file to aligned pose from aligned camera to Zero123 conditon pose')
    parser.add_argument('--out_dir', type=str, default='', help='output results directory')
    args = parser.parse_args()

    main(args)


