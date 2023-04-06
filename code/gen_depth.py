import numpy as np
import torch
import math
import torch.nn.functional as F
import os
import trimesh
# from Utils import *
import nvdiffrast.torch as dr
import json
from pathlib import Path
import sys
sys.path.append("third_party/utils_simba")
from utils_simba.geometry import save_point_cloud_to_ply
from utils_simba.img import show_img
from utils_simba.depth import save_depth
from utils_simba.render import nvdiffrast_render, make_mesh_tensors


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--seqence_name", required=False, help="sequence name")
parser.add_argument("--in_dir", required=False, help="input directory")
parser.add_argument("--out_dir", required=False, help="output directory")
parser.add_argument("--debug", action="store_true")


args, extras = parser.parse_known_args()

def get_sphere_camera(elevation_deg = 30, azimuth_deg = -30, camera_distance = 2, fovy_deg = 41.15, size = [256, 256]):
    elevation = torch.tensor(elevation_deg) * math.pi / 180
    azimuth = torch.tensor(azimuth_deg) * math.pi / 180
    camera_distance = torch.tensor(camera_distance)
    camera_position = torch.stack(
        [
            camera_distance * torch.cos(elevation) * torch.cos(azimuth),
            camera_distance * torch.cos(elevation) * torch.sin(azimuth),
            camera_distance * torch.sin(elevation),
        ],
        dim=-1,
    )

    center = torch.zeros_like(camera_position)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)

    lookat = F.normalize(center - camera_position, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_position[:, None]],
        dim=-1,
    )
    glc2w4x4 = torch.cat(
        [c2w, torch.zeros_like(c2w[:1])], dim=0
    )
    glc2w4x4[3, 3] = 1.0
    glc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cvc2w4x4 = glc2w4x4 @ torch.tensor(glc2cvc).float()

    fovy = torch.deg2rad(torch.FloatTensor([fovy_deg]))
    f = 0.5 * size[1] / torch.tan(0.5 * fovy)
    K = torch.tensor(
        [
            [f, 0, int(0.5 * (size[0] - 1))],
            [0, f, int(0.5 * (size[1] - 1))],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return np.array(cvc2w4x4), np.array(K)

def depth_to_point_cloud(depth_map, K):
    """
    Args:
        depth_map (numpy.ndarray): A 2D array (height, width) containing depth values.
        K (numpy.ndarray): The 3x3 intrinsic camera matrix.
        
    Returns:
        numpy.ndarray: A (N, 3) array where N is the number of valid points in the point cloud, and each row is a 3D point (X, Y, Z).
    """
    
    # Get the image height and width
    height, width = depth_map.shape

    # Generate a grid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten u, v, and depth_map for easier manipulation
    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth_map.flatten()

    # Filter out invalid (zero or very close to zero) depth values
    valid = depth_flat > -1000
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    depth_valid = depth_flat[valid]

    # Create homogeneous pixel coordinates
    pixel_coords = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)

    # Compute the inverse of the intrinsic matrix K
    K_inv = np.linalg.inv(K)

    # Unproject the pixel coordinates to camera space
    # point_camera = depth * K_inv * pixel_coords
    points_3d = depth_valid * (K_inv @ pixel_coords)

    # Transpose the result to get an (N, 3) point cloud
    points_3d = points_3d.T
    
    return points_3d


if __name__ == "__main__":    
    seq_name = args.seqence_name
    mesh_path = f"{args.in_dir}/model.obj"
    
    outputs = Path(f"{args.out_dir}")    
    cmd = f"rm -rf {outputs} && mkdir -p {outputs}"

    os.system(cmd)

    mesh = trimesh.load(mesh_path)

    mesh_tensors = make_mesh_tensors(mesh)

    camera_file = f"{args.in_dir}/camera.json"
    camera = json.load(open(camera_file, 'r'))
    elevation_deg, azimuth_deg, camera_distance, fovy_deg = float(camera['elevation_deg']), float(camera['azimuth_deg']), float(camera['distance']), float(camera['fovy_deg'])
    H = 256
    W = 256
    image_size = [H, W]
    cvc2w4x4, K =  get_sphere_camera(elevation_deg = elevation_deg, azimuth_deg = azimuth_deg, camera_distance = camera_distance, fovy_deg = fovy_deg, size = image_size)
    w2cvc4x4 = np.linalg.inv(cvc2w4x4)
    w2cvc4x4 = torch.tensor(w2cvc4x4)[None].to('cuda')

    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=w2cvc4x4, context='cuda', get_normal=False, 
                                                glctx=dr.RasterizeCudaContext(), mesh_tensors=mesh_tensors, output_size=image_size, bbox2d=None, use_light=True)

    # 

    rgb_r = rgb_r[0].cpu().numpy()
    depth_r = depth_r[0].cpu().numpy()
    normal_r = normal_r[0].cpu().numpy()
    if args.debug:
        show_img(rgb_r)

    R = cvc2w4x4[:3, :3]
    t = cvc2w4x4[:3, 3]

    point_cloud = depth_to_point_cloud(depth_r, K)
    points_in_camera_space = (R @ point_cloud.T + t[:, None]).T


    camera['blw2cvc'] = np.linalg.inv(cvc2w4x4).tolist()
    camera['K'] = K.tolist()

    pcs_in_cam_f = f"{outputs}/pcs_in_cam.ply"
    pcs_in_world_f = f"{outputs}/pcs_in_world.ply"
    depth_f = f"{outputs}/depth.png"
    out_cam_file = f"{outputs}/camera.json"
    save_point_cloud_to_ply(point_cloud, pcs_in_cam_f)
    save_point_cloud_to_ply(points_in_camera_space, pcs_in_world_f)
    save_depth(depth_r, depth_f)

    with open(out_cam_file, 'w') as f:
        json.dump(camera, f, indent=4)
        print(f"Camera saved to {out_cam_file}")
    

