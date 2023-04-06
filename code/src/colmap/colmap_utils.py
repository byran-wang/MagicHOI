import os
from glob import glob
from pathlib import Path
import open3d as o3d
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import torch
import trimesh
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
    visualization,
)
from hloc.utils import read_write_model
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import json
from utils_simba.depth import save_depth
import shutil
from utils_simba.depth import depth2xyzmap, xyz2depthmap
from utils_simba.geometry import convert_point_cloud

import src.colmap.colmap_readmodel as read_model
def plot_2d_projection(pts_cano, denormalize_mat, o2w_all, intrinsic, seq_name, fnames):
    
    print("Projecting 3d points to 2d")
    pts_cano_homogeneous = np.hstack([pts_cano, np.ones((pts_cano.shape[0], 1))])
    pts_cano_denorm = np.dot(pts_cano_homogeneous, denormalize_mat.T)

    pts_w = np.array(
        [np.dot(pts_cano_denorm, o2w_all[i].T) for i in range(o2w_all.shape[0])]
    )
    pts_w = pts_w[:, :, :3] / pts_w[:, :, 3:]

    projected_pts = np.dot(pts_w, intrinsic.T)
    projected_2d = projected_pts[:, :, :2] / projected_pts[:, :, 2:]

    for idx in tqdm(range(len(fnames))):
        fname = fnames[idx]
        v2d = projected_2d[idx]
        fname = fname.replace("/images_object/", "/images/")

        out_p = fname.replace("/images/", f"/colmap_2d/")

        # Load image
        img = Image.open(fname)

        # Get the 2D points for the current frame
        points = v2d[:300]
        points_a = points[:150]
        points_b = points[150:]

        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        # Create scatter plot on the image
        plt.figure(figsize=(8, 8))

        plt.scatter(points_a[:, 0], points_a[:, 1], c="red", s=5)
        plt.scatter(points_b[:, 0], points_b[:, 1], c="blue", s=5)
        plt.imshow(img)
        plt.savefig(out_p)
        plt.close()
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    out_p = os.path.join(os.path.dirname(out_p), "keypoints.npy")

    np.save(out_p, projected_2d)
    print("Saving keypoints to", out_p)
    print("----------------------------------------------")

def slerp_o2w(o2w_all, key_frames, num_frames):
    # Assertions to check the input dimensions
    assert o2w_all.ndim == 3, "o2w_all should be a 3D array"
    assert o2w_all.shape[1:] == (4, 4), "Each element of o2w_all should be a 4x4 matrix"
    assert (
        len(key_frames) == o2w_all.shape[0]
    ), "Number of key frames should match the first dimension of o2w_all"
    assert (
        isinstance(num_frames, int) and num_frames > 0
    ), "num_frames should be a positive integer"

    expected_frames = np.arange(num_frames)

    # handle edge cases when key_frames are outside of expected_frames
    # check first keyframe
    if not (key_frames[0] <= expected_frames[0]):
        key_frames = np.concatenate([[expected_frames[0]], key_frames])
        o2w_all = np.concatenate([o2w_all[:1], o2w_all])

    # check last keyframe
    if not (key_frames[-1] >= expected_frames[-1]):
        key_frames = np.concatenate([key_frames, [expected_frames[-1]]])
        o2w_all = np.concatenate([o2w_all, o2w_all[-1:]])

    rots = o2w_all[:, :3, :3]
    key_rots = R.from_matrix(rots)
    key_trans = o2w_all[:, :3, 3]
    slerp = Slerp(key_frames, key_rots)
    interp_rots = slerp(expected_frames).as_matrix()

    # Create an interpolation object for translations, interpolating each dimension separately
    interp_trans_x = np.interp(expected_frames, key_frames, key_trans[:, 0])
    interp_trans_y = np.interp(expected_frames, key_frames, key_trans[:, 1])
    interp_trans_z = np.interp(expected_frames, key_frames, key_trans[:, 2])
    interp_trans = np.vstack([interp_trans_x, interp_trans_y, interp_trans_z]).T

    # Create the interpolated o2w_all matrix
    interp_o2w_all = np.zeros((num_frames, 4, 4))
    interp_o2w_all[:, :3, :3] = interp_rots
    interp_o2w_all[:, :3, 3] = interp_trans
    interp_o2w_all[:, 3, 3] = 1

    return interp_o2w_all


def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, "cameras.bin")
    camdata = read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print("Cameras", len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, "images.bin")
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print("Images #", len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate(
        [poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1
    )

    points3dfile = os.path.join(realdir, "points3D.bin")
    pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate(
        [
            poses[:, 1:2, :],
            poses[:, 0:1, :],
            -poses[:, 2:3, :],
            poses[:, 3:4, :],
            poses[:, 4:5, :],
        ],
        1,
    )

    return poses, pts3d, perm


def export_colmap_results(basedir, poses, pts3d, perm):
    # point cloud export
    pts = np.stack([pts3d[k].xyz for k in pts3d], axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, "sparse_points.ply"))

    # Adjust poses dimensions and apply permutation
    poses = np.moveaxis(poses, -1, 0)
    poses = poses[perm]

    # Save the adjusted poses as a .npy file
    np.save(os.path.join(basedir, "poses.npy"), poses)


def format_poses(seq_name):
    import torch

    poses_hwf = np.load(f"./data/{seq_name}/colmap/poses.npy")

    poses_hwf = torch.FloatTensor(poses_hwf)

    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]

    h, w, f = hwf[0]

    intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
    intrinsic[0, 2] = (w - 1) * 0.5
    intrinsic[1, 2] = (h - 1) * 0.5
    intrinsic = intrinsic[:3, :3]

    num_frames = poses_raw.shape[0]

    convert_mat = torch.zeros([4, 4])
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] = -1.0
    convert_mat[3, 3] = 1.0

    # w2o mat to opencv format
    w2o_all = torch.eye(4)[None, :, :].repeat(num_frames, 1, 1)
    w2o_all[:, :3] = poses_raw
    w2o_all = torch.bmm(w2o_all, convert_mat[None, :, :].repeat(num_frames, 1, 1))
    o2w_all = w2o_all.inverse().numpy()

    # SLERP
    with open(
        f"./data/{seq_name}/colmap/sfm_superpoint+superglue/converged_frames.txt",
        "r",
    ) as file:
        lines = file.readlines()
        integer_lines = [int(line.strip()) for line in lines if line.strip()]
        valid_frames = np.array(integer_lines)
    assert len(valid_frames) == len(o2w_all)
    assert valid_frames.min() > 0  ## assume 1-based in valid frames
    key_frames = valid_frames - 1
    num_frames = len(glob(f"./data/{seq_name}/images/*"))

    sort_idx = np.argsort(key_frames)
    key_frames = key_frames[sort_idx]
    # o2w_all = o2w_all[sort_idx]

    interp_o2w_all = slerp_o2w(o2w_all, key_frames, num_frames)
    o2w_all = interp_o2w_all

    # colmap verts (not normalized)
    mesh = trimesh.load(
        f"./data/{seq_name}/colmap/sparse_points_trim.ply", process=False
    )
    vertices = np.array(mesh.vertices)

    # construct normalization matrix
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)

    # bbox center
    center = (bbox_max + bbox_min) * 0.5

    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    # radius = 1.0
    # from scaled & center to unscaled and not centered (original in COLMAP)
    denormalize_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    denormalize_mat[:3, 3] = center

    # center and scale COLMAP point cloud
    normalize_mat = np.linalg.inv(denormalize_mat)

    # normalize colmap points
    pts_cano = np.ones((vertices.shape[0], 4))
    pts_cano[:, :3] = vertices

    pts_cano = (normalize_mat @ pts_cano.T).T
    pts_cano = pts_cano[:, :3] / pts_cano[:, 3:]
    pc_p = f"./data/{seq_name}/colmap/sparse_points_normalized.obj"
    mesh.vertices = pts_cano
    mesh.export(pc_p)

    norm_mat_p = f"./data/{seq_name}/colmap/normalization_mat.npy"
    intrinsic_p = f"./data/{seq_name}/colmap/intrinsic.npy"
    pose_p = f"./data/{seq_name}/colmap/o2w.npy"

    np.save(norm_mat_p, normalize_mat)
    np.save(intrinsic_p, intrinsic)
    np.save(pose_p, o2w_all)

    print("Saving normalized point cloud to", pc_p)
    print("Saving normalization matrix to", norm_mat_p)
    print("Saving intrinsic matrix to", intrinsic_p)
    print("Saving pose matrix to", pose_p)

def colmap_pose_est(seq_name, num_keypoints):
    image_path = f"./data/{seq_name}/images_object"
    output_path = f"./data/{seq_name}/colmap"
    camera_path = f"./data/{seq_name}/colmap/sfm_superpoint+superglue"
    intrinsic_f = f"./data/{seq_name}/intrinsic/intrins.txt"

    images = Path(image_path)
    outputs = Path(output_path)
    num_images = len(glob(f"{image_path}/*"))
    assert num_keypoints <= num_images, f"{num_keypoints} should be less or equal to {num_images}"

    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"
    features = outputs / "features.h5"
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]
    references = [p.relative_to(images).as_posix() for p in (images).iterdir()]

    retrieval_path = extract_features.main(
        retrieval_conf, images, image_list=references, feature_path=features
    )
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_keypoints)

    feature_path = extract_features.main(feature_conf, images, outputs, vis_features=False)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    model = reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        feature_path,
        match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        intrinsic_f=intrinsic_f,
    )

    visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)

    images_ret = read_model.read_images_binary(sfm_dir / "images.bin")
    file_name = "converged_frames.txt"
    frames_conv = images_ret.keys()
    with open(os.path.join(sfm_dir, file_name), "w") as file:
        # Write each word to a separate line
        for frame_number in frames_conv:
            file.write(str(frame_number) + "\n")

    poses, pts3d, perm = load_colmap_data(camera_path)
    export_colmap_results(outputs, poses, pts3d, perm)

def colmap_pose_est_diff_object(selected_views, data_path, out_path, num_keypoints, mute):
    image_path = f"{data_path}/images"
    intrinsic_f = f"{data_path}/intrinsic/intrins.txt"
    output_path = f"{out_path}"
    camera_path = f"{out_path}/sfm_superpoint+superglue"

    images = Path(image_path)
    outputs = Path(output_path)
    num_images = len(glob(f"{image_path}/*"))
    assert num_keypoints <= num_images, f"{num_keypoints} should be less or equal to {num_images}"

    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"
    features = outputs / "features.h5"
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]
    references_all = [p.relative_to(images).as_posix() for p in (images).iterdir()]
    references = []
    for ref in references_all:
        if int(ref.split(".png")[0]) in selected_views:
            references.append(ref)
    if mute:
        vis_features = False
    else:
        vis_features = True
    retrieval_path = extract_features.main(
        retrieval_conf, images, feature_path=features, image_list=references, vis_features=vis_features
    )
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_keypoints)

    feature_path = extract_features.main(feature_conf, images, outputs, image_list=references, vis_features=vis_features)

    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    model = reconstruction.main(
        sfm_dir,
        images,        
        sfm_pairs,
        feature_path,
        match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        intrinsic_f=intrinsic_f,
        image_list=references,
    )

    visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)

    images_ret = read_model.read_images_binary(sfm_dir / "images.bin")
    file_name = "converged_frames.txt"
    frames_conv = images_ret.keys()
    with open(os.path.join(sfm_dir, file_name), "w") as file:
        # Write each word to a separate line
        for frame_number in frames_conv:
            file.write(str(frame_number) + "\n")

    poses, pts3d, perm = load_colmap_data(camera_path)
    export_colmap_results(outputs, poses, pts3d, perm)


def trim_point_cloud(sp_p, percentile=80, scale_factor=1.5):
    # this function trim point cloud by first computing its median
    # then find its 80 percentile for a threshold and pad with 1.5*thres to create a boundary
    # then we use the boundary to decide which points we include

    out_p = sp_p.replace("/sparse_points.ply", "/sparse_points_trim.ply")

    sp = trimesh.load(sp_p, process=False)
    verts = np.array(sp.vertices)
    center = np.median(verts, axis=0)
    dist = np.linalg.norm(verts - center[None, :], axis=1)

    thres = np.percentile(dist, percentile)
    thres = scale_factor * thres

    verts_trim = verts[dist < thres]

    pc = trimesh.Trimesh(vertices=verts_trim)
    pc.export(out_p)
    
    print("Saved trimmed point cloud to", out_p)
    return pc


def slerp_o2w(o2w_all, key_frames, num_frames):
    # Assertions to check the input dimensions
    assert o2w_all.ndim == 3, "o2w_all should be a 3D array"
    assert o2w_all.shape[1:] == (4, 4), "Each element of o2w_all should be a 4x4 matrix"
    assert (
        len(key_frames) == o2w_all.shape[0]
    ), "Number of key frames should match the first dimension of o2w_all"
    assert (
        isinstance(num_frames, int) and num_frames > 0
    ), "num_frames should be a positive integer"

    expected_frames = np.arange(num_frames)

    start_time = key_frames[0]
    end_time = key_frames[-1]

    start_o2w = o2w_all[:1]
    end_o2w = o2w_all[-1:]

    start_time_query = expected_frames[0]
    end_time_query = expected_frames[-1]

    if start_time_query < start_time:
        o2w_all = np.concatenate((start_o2w, o2w_all), axis=0)
        key_frames = np.concatenate(([start_time_query], key_frames), axis=0)

    if end_time < end_time_query:
        o2w_all = np.concatenate((o2w_all, end_o2w), axis=0)
        key_frames = np.concatenate((key_frames, [end_time_query]), axis=0)

    # interpolate rotation
    rots = o2w_all[:, :3, :3]
    key_rots = R.from_matrix(rots)
    slerp = Slerp(key_frames, key_rots)
    interp_rots = slerp(expected_frames).as_matrix()

    # interpolate translation
    key_trans = o2w_all[:, :3, 3]

    # Create an interpolation object for translations, interpolating each dimension separately
    interp_trans_x = np.interp(expected_frames, key_frames, key_trans[:, 0])
    interp_trans_y = np.interp(expected_frames, key_frames, key_trans[:, 1])
    interp_trans_z = np.interp(expected_frames, key_frames, key_trans[:, 2])
    interp_trans = np.vstack([interp_trans_x, interp_trans_y, interp_trans_z]).T

    # Create the interpolated o2w_all matrix
    interp_o2w_all = np.zeros((num_frames, 4, 4))
    interp_o2w_all[:, :3, :3] = interp_rots
    interp_o2w_all[:, :3, 3] = interp_trans
    interp_o2w_all[:, 3, 3] = 1

    return interp_o2w_all


def read_hwf_poses(hwf_p):
    poses_hwf = np.load(hwf_p)
    poses_hwf = torch.FloatTensor(poses_hwf)

    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]

    h, w, f = hwf[0]

    intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
    intrinsic[0, 2] = (w - 1) * 0.5
    intrinsic[1, 2] = (h - 1) * 0.5
    intrinsic = intrinsic[:3, :3]

    num_frames = poses_raw.shape[0]

    convert_mat = torch.zeros([4, 4])
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] = -1.0
    convert_mat[3, 3] = 1.0

    # w2o mat to opencv format
    w2o_all = torch.eye(4)[None, :, :].repeat(num_frames, 1, 1)
    w2o_all[:, :3] = poses_raw
    w2o_all = torch.bmm(w2o_all, convert_mat[None, :, :].repeat(num_frames, 1, 1))
    o2w_all = w2o_all.inverse().numpy()

    return intrinsic, o2w_all



def canonical_normalization(vertices):
    # construct normalization matrix
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)

    # bbox center
    center = (bbox_max + bbox_min) * 0.5

    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    # radius = 1.0
    # from scaled & center to unscaled and not centered (original in COLMAP)
    denormalize_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    denormalize_mat[:3, 3] = center

    # center and scale COLMAP point cloud
    normalize_mat = np.linalg.inv(denormalize_mat)

    # normalize colmap points
    pts_cano = np.ones((vertices.shape[0], 4))
    pts_cano[:, :3] = vertices

    pts_cano = (normalize_mat @ pts_cano.T).T
    pts_cano = pts_cano[:, :3] / pts_cano[:, 3:]

    return pts_cano, denormalize_mat, normalize_mat

def read_valid_frames(seq_name, reading_path=None):
    # Reading the key frames
    valid_frames = os.listdir(reading_path)
    valid_frames = [int(frame.split(".")[0]) for frame in valid_frames]
    valid_frames = np.array(valid_frames)
    valid_frames = np.sort(valid_frames)
    return valid_frames

def quaternion_to_rotation_matrix(qvec):
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])

def reproject_point(point3D, camera, image):
    f, k, cx, cy = camera.params
    k = 0
    R = quaternion_to_rotation_matrix(image.qvec)
    t = image.tvec
    K = np.array([
        [camera.params[0], 0, camera.params[1]],
        [0, camera.params[0], camera.params[2]],
        [0, 0, 1]
    ])
    # Compute camera projection
    X_world = point3D
    t = image.tvec
    X_cam = R @ X_world + t
    x_cam, y_cam, z_cam = X_cam
    # Normalize to image plane coordinates
    x_norm = x_cam / z_cam
    y_norm = y_cam / z_cam    

    # Apply radial distortion
    r2 = x_norm**2 + y_norm**2
    x_distorted = x_norm * (1 + k * r2)
    y_distorted = y_norm * (1 + k * r2)

    # Convert to pixel coordinates
    u = f * x_distorted + cx
    v = f * y_distorted + cy
    projected_point = np.array([u, v])

    return projected_point

def validate_reprojection(seq_name):
    cameras, images, points3D = read_write_model.read_model(f"./data/{seq_name}/colmap/sfm_superpoint+superglue/")

    for point_id, point3D_data in points3D.items():
        point2D_projs = []
        point2D_kps = []
        reprojection_errors = []
        for image_id, point2D_idx in zip(point3D_data.image_ids, point3D_data.point2D_idxs):
            image = images[image_id]
            camera = cameras[image.camera_id]
            point2D_proj = reproject_point(point3D_data.xyz, camera, image)
            point2D_kp = image.xys[point2D_idx]
            error = np.linalg.norm(point2D_proj - point2D_kp)
            reprojection_errors.append(error)
            point2D_projs.append(point2D_proj)
            point2D_kps.append(point2D_kp)
        average_error = np.mean(reprojection_errors)
        rms_error = np.sqrt(np.mean(np.square(reprojection_errors)))
        breakpoint()    
        print(f"average error: {average_error}, rms error: {rms_error}, point3D_data.error: {point3D_data.error}")

from hloc.utils.read_write_model import read_cameras_binary
def validate_colmap(seq_name, no_vis, do_slerp=True, colmap_path=None, data_path=None):
    # read object poses and intrinsics
    if colmap_path is None:
        hwf_p = f"./data/{seq_name}/colmap/poses.npy"
    else:
        hwf_p = f"{colmap_path}/poses.npy"
    intrinsic, o2w_all = read_hwf_poses(hwf_p)
    cameras = read_cameras_binary(f"{colmap_path}/sfm_superpoint+superglue/cameras.bin")
    params = cameras[list(cameras.keys())[0]].params
    intrinsic[0, 0] = params[0]
    intrinsic[1, 1] = params[1]
    intrinsic[0, 2] = params[2]
    intrinsic[1, 2] = params[3]
    
    # check converged frames in colmap
    valid_frames = read_valid_frames(seq_name, reading_path=f"{colmap_path}/sfm_superpoint+superglue/mvs/images/")
    key_frames = valid_frames - 1
    assert len(valid_frames) == len(o2w_all)

    # SLERP to interpolate failed poses
    sort_idx = np.argsort(key_frames)
    key_frames = key_frames[sort_idx]
    # o2w_all = o2w_all[sort_idx]
    if data_path is None:
        num_frames = len(glob(f"./data/{seq_name}/images_object/*"))
    else:
        num_frames = len(glob(f"{data_path}/images_object/*"))
    if do_slerp:
        interp_o2w_all = slerp_o2w(o2w_all, key_frames, num_frames)
    else:
        interp_o2w_all = o2w_all
    all_frames = np.arange(num_frames)
    missing_frames = [frame for frame in all_frames if frame not in key_frames]
    print("Missing frames", missing_frames)
    print("Number of missing frames", len(missing_frames))

    # remove outlier SfM points
    if colmap_path is None:
        pc_trim = trim_point_cloud(f"./data/{seq_name}/colmap/sparse_points.ply", percentile=80, scale_factor=1.5)
    else:
        pc_trim = trim_point_cloud(f"{colmap_path}/sparse_points.ply", percentile=80, scale_factor=1.5)

    # zero-center and normalize the canonical space
    pts_cano, denormalize_mat, normalize_mat = canonical_normalization(np.array(pc_trim.vertices))
    
    # save processed results
    if colmap_path is None:
        pc_p = f"./data/{seq_name}/colmap/sparse_points_normalized.obj"
        norm_mat_p = f"./data/{seq_name}/colmap/normalization_mat.npy"
        intrinsic_p = f"./data/{seq_name}/colmap/intrinsic.npy"
        pose_p = f"./data/{seq_name}/colmap/o2w.npy"
    else:
        pc_p = f"{colmap_path}/sparse_points_normalized.obj"
        norm_mat_p = f"{colmap_path}/normalization_mat.npy"
        intrinsic_p = f"{colmap_path}/intrinsic.npy"
        pose_p = f"{colmap_path}/o2w.npy"
    pc_trim.vertices = pts_cano
    pc_trim.export(pc_p)
    np.save(norm_mat_p, normalize_mat)
    np.save(intrinsic_p, intrinsic)
    np.save(pose_p, interp_o2w_all)

    # 2d projection
    if not no_vis:
        fnames = []
        for valid_frame in valid_frames:
            fname = f"{data_path}/images_object/{valid_frame:04d}.png"
            fnames.append(fname)
        plot_2d_projection(pts_cano, denormalize_mat, interp_o2w_all, intrinsic, seq_name, fnames)

def list_files_in_directory(directory_path, extension):
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(extension)]

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def center_and_normalize_point_cloud(file_path: str, output_path: str) -> np.ndarray:
    """
    Centers and normalizes a point cloud to fit within a cube [-0.5, 0.5],
    and aligns the longest principal axis with the z-axis.

    Parameters:
    - file_path (str): Path to the input PLY point cloud file.
    - output_path (str): Path to save the transformed PLY point cloud.

    Returns:
    - transformation_matrix (np.ndarray): The 4x4 transformation matrix applied.
    """
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"The point cloud at {file_path} has no points.")
    
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # Check if point cloud has colors
    has_colors = pcd.has_colors()
    if has_colors:
        print("Point cloud has color information.")
    else:
        print("Point cloud does NOT have color information.")

    # Convert point cloud to numpy array for processing
    points = np.asarray(pcd.points)

    # Compute the centroid of the point cloud
    centroid = np.mean(points, axis=0)
    print(f"Centroid of point cloud: {centroid}")

    # Translate the point cloud to center it at the origin
    translated_points = points - centroid
    print("Translated point cloud to center at the origin.")
    
    # Perform PCA to get the principal axes
    U, S, Vt = np.linalg.svd(translated_points, full_matrices=False)
    V = Vt.T
    first_principal_axis = V[:, 0]
    print(f"First principal axis: {first_principal_axis}")
    
    # Compute rotation matrix that aligns the first principal axis with z-axis
    def rotation_matrix_from_vectors(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)

        # Constrain rotation angle within 90 degrees
        angle = np.arccos(np.clip(c, -1.0, 1.0))
        if angle > np.pi / 2:
            b = -b
            v = np.cross(a, b)
            c = np.dot(a, b)

        if c < -0.999999:
            # Vectors are opposite
            ortho = np.array([1, 0, 0]) if abs(a[0]) < abs(a[1]) else np.array([0, 1, 0])
            v = np.cross(a, ortho)
            v = v / np.linalg.norm(v)
            return np.eye(3) - 2 * np.outer(v, v)
        elif c > 0.999999:
            # Vectors are the same
            return np.eye(3)
        else:
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
            return R

    rotation_matrix = rotation_matrix_from_vectors(first_principal_axis, np.array([0, 0, 1]))
    rotation_matrix = np.eye(3)
    print("Computed rotation matrix to align the first principal axis with z-axis.")
    
    # Rotate the point cloud
    rotated_points = translated_points @ rotation_matrix.T
    print("Rotated point cloud to align the longest principal axis with z-axis.")
    
    # Compute the maximum extent (max distance from origin along any axis)
    max_extent = np.max(np.abs(rotated_points))
    print(f"Maximum extent from origin: {max_extent}")

    # Compute the scaling factor to fit within [-0.5, 0.5]
    # Desired maximum extent is 0.5
    scaling_factor = 0.5 / max_extent
    print(f"Scaling factor: {scaling_factor}")

    # Scale the point cloud uniformly
    scaled_points = rotated_points * scaling_factor
    print("Scaled point cloud to fit within [-0.5, 0.5].")

    # Create the transformation matrix
    transaction_4x4 = np.identity(4)
    # rotation
    transaction_4x4[:3, :3] = scaling_factor * rotation_matrix
    # transformation_matrix[:3, :3] = scaling_factor * np.identity(3)
    # transaction
    transaction_4x4[:3, 3] = -scaling_factor * rotation_matrix @ centroid

    rotation_3x3 = rotation_matrix
    # transformation_matrix[:3, 3] = -scaling_factor * centroid
    print(f"Transformation matrix:\n{transaction_4x4}")
    print(f"Rotation matrix:\n{rotation_3x3}")
    print(f"Scaling factor: {scaling_factor}")

    # Apply the transformation to the point cloud
    pcd.points = o3d.utility.Vector3dVector(scaled_points)

    # If colors exist, ensure they are preserved
    if has_colors:
        colors = np.asarray(pcd.colors)
        # Normalize colors if they are in [0, 255]
        if colors.max() > 1.0:
            colors = colors / 255.0
            print("Normalized color values to [0, 1].")
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Preserved color information in the transformed point cloud.")

    # Save the transformed point cloud to a new PLY file
    success = o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    if success:
        print(f"Transformed point cloud saved to {output_path}")
    else:
        raise IOError(f"Failed to save the transformed point cloud to {output_path}")

    return transaction_4x4, rotation_3x3, scaling_factor
    

def convert_to_HO3D_format(out_path):
    out_camera_path = Path(f"{out_path}/HO3D/cameras")
    out_depth_path = Path(f"{out_path}/HO3D/depths")
    model_path = Path(f"{out_path}/sfm_superpoint+superglue/")
    depth_path = Path(f"{out_path}/sfm_superpoint+superglue/mvs/stereo/depth_maps")
    out_camera_path.mkdir(parents=True, exist_ok=True)
    out_depth_path.mkdir(parents=True, exist_ok=True)

    # remove outlier SfM points
    dense_pc_f = f"{out_path}/sfm_superpoint+superglue/mvs/fused.ply"
    sparse_pc_f = f"{out_path}/sparse_points_trim.ply"
    dense_pc_normalized_f = dense_pc_f.split(".ply")[0] + "_normalized.ply"
    sparse_pc_normalized_f = f"{out_path}/sfm_superpoint+superglue/mvs/sparse_points_normalized.ply"
    norm_mat_f = f"{out_path}/sfm_superpoint+superglue/mvs/mat_normalized.npy"
    norm_o2c_f = f"{out_path}/sfm_superpoint+superglue/mvs/o2w_normalized.npy"
    norm_mat, norm_rotate, norm_scale =  center_and_normalize_point_cloud(dense_pc_f, dense_pc_normalized_f)
    convert_point_cloud(sparse_pc_f, sparse_pc_normalized_f, norm_mat)
    np.save(norm_mat_f, norm_mat)
    cameras, images, points3D = read_write_model.read_model(model_path)
    blw2cvc_norms = []
    print(f"out camera path: {out_camera_path}")
    for image_id, image in tqdm(images.items(), total=len(images), desc="normalizing camera"):
        name = image.name.split(".")[0]
        R = image.qvec2rotmat()
        t = image.tvec
        blw2cvc = np.eye(4)
        blw2cvc[:3, :3] = R
        blw2cvc[:3, 3] = t
        cvc2blw = np.linalg.inv(blw2cvc)
        cvc2blw_t = cvc2blw[:4, 3]
        cvc2blw_R = cvc2blw[:3, :3]
        t_norm = norm_mat @ cvc2blw_t        
        R_norm = norm_rotate @ cvc2blw_R
        cvc2blw_norm = np.eye(4)
        cvc2blw_norm[:3, 3] = t_norm[:3]
        cvc2blw_norm[:3, :3] = R_norm
        blw2cvc_norm = np.linalg.inv(cvc2blw_norm)
        blw2cvc_norms.append(blw2cvc_norm)
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = camera.params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        width, height = camera.width, camera.height
        json_f = out_camera_path / f"{name}.json"
        json_data = {
            "blw2cvc": blw2cvc_norm.tolist(),
            # "blw2cvc": blw2cvc.tolist(),
            "K": K.tolist(),
            "width": width,
            "height": height
        }
        with open(json_f, "w") as f:
            json.dump(json_data, f, indent=4)
    np.save(norm_o2c_f, blw2cvc_norms)
    depth_extension = ".png.geometric.bin"
    depth_files = list_files_in_directory(depth_path, depth_extension)
    print(f"saving depth to {out_depth_path}")
    for depth_f in tqdm(depth_files, total=len(depth_files), desc="saving depth"):
        name = depth_f.split(depth_extension)[0]
        depth_map = read_array(depth_path/depth_f)
        depth_map = depth_map * norm_scale
        out_depth_f = out_depth_path / f"{name}.png"
        save_depth(depth_map, str(out_depth_f))

def colmap_mvs(data_path, out_path):
    image_dir = Path(f"{data_path}/rgbas")
    output_path = Path(f"{out_path}/sfm_superpoint+superglue")
    mvs_path = output_path / "mvs"
    if os.path.exists(mvs_path):
        shutil.rmtree(mvs_path)
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    time.sleep(5)
    cmd = ""
    cmd += (
        f"export COLMAP_EXE_PATH=/usr/local/bin/ && "
        f"cd {mvs_path} && "
        f"$COLMAP_EXE_PATH/colmap patch_match_stereo "
        f"--workspace_path . "
        f"--workspace_format COLMAP "
        f"--PatchMatchStereo.max_image_size 2000 "
        f"--PatchMatchStereo.geom_consistency true && "
        f"$COLMAP_EXE_PATH/colmap stereo_fusion "
        f"--workspace_path . "
        f"--workspace_format COLMAP "
        f"--input_type geometric "
        f"--output_path ./fused.ply"
    )
    print(f"cmd: {cmd}")
    os.system(cmd)

def colmap_mvs_1(seq_name):
    image_path = f"./data/{seq_name}/images_object/"
    model_path = f"./data/{seq_name}/colmap/sfm_superpoint+superglue"
    mvs_path = f"./data/{seq_name}/colmap/mvs"
    if os.path.exists(mvs_path):
        shutil.rmtree(mvs_path)
    pycolmap.undistort_images(mvs_path, model_path, image_path)
    cmd = ""
    cmd += f"export COLMAP_EXE_PATH=/usr/local/bin/ && cd {mvs_path} && bash ./run-colmap-geometric.sh "
    print(f"cmd: {cmd}")
    os.system(cmd)    
