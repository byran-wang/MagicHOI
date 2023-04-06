import numpy as np
from scipy.optimize import least_squares
import argparse
from pathlib import Path
import os
import open3d as o3d
import json
import sys
sys.path.append("third_party/utils_simba")
from utils_simba.geometry import transform_points, project_points, rodrigues_to_rotation_matrix, transform4x4_to_rodrigues_and_translation, rodrigues_and_translation_to_transform4x4
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def rotation_matrix_from_vector(theta):
    """
    Convert a rotation vector (axis-angle representation) to a rotation matrix.
    """
    angle = np.linalg.norm(theta)
    if angle == 0:
        return np.eye(3)
    axis = theta / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def align_point_clouds_huber(P, Q, delta, s0=1.0, theta0=np.zeros(3), t0=np.zeros(3)):
    """
    Align source point cloud P to target point cloud Q using robust optimization with Huber loss.
    """
    def residuals(params, P, Q):
        s = params[0]
        theta = params[1:4]
        t = params[4:7]
        
        # Convert rotation vector to rotation matrix
        R = rotation_matrix_from_vector(theta)
        
        # Transform source points
        transformed_P = s * (R @ P.T).T + t
        
        # Compute residuals (Euclidean distances)
        res = transformed_P - Q
        res_norms = np.linalg.norm(res, axis=1)
        return res_norms
    
    # Initial parameter guess
    params0 = np.hstack([s0, theta0, t0])
    
    # Set bounds for optimization parameters
    lower_bounds = [0] + [-np.inf]*6  # s > 0, other parameters unbounded
    upper_bounds = [np.inf]*7  # No upper bounds
    
    # Perform optimization with Huber loss and bounds
    result = least_squares(
        residuals,
        params0,
        args=(P, Q),
        loss='huber',
        f_scale=delta,
        jac='2-point',
        method='trf',  # Trust Region Reflective algorithm
        bounds=(lower_bounds, upper_bounds)
    )
    
    # Extract optimized parameters
    s_opt = result.x[0]
    theta_opt = result.x[1:4]
    t_opt = result.x[4:7]
    R_opt = rotation_matrix_from_vector(theta_opt)
    
    # Transform the source point cloud
    P_aligned = s_opt * (R_opt @ P.T).T + t_opt
    
    return s_opt, R_opt, t_opt, P_aligned


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def Rt2T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def create_rotation_matrix(axis, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle_rad), -np.sin(angle_rad)],
                      [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                      [np.sin(angle_rad), np.cos(angle_rad), 0],
                      [0, 0, 1]])
    else:
        R = np.eye(3)
    return R

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_index", required=True, help="query index")
    parser.add_argument("--seqence_name", required=True, help="sequence name")
    parser.add_argument("--data_dir", required=False, help="data directory")
    parser.add_argument("--cond_camera_f", required=False, help="condition camera file")
    parser.add_argument("--cond_img_f", required=False, help="condition image file")
    parser.add_argument("--out_dir", required=False, help="output directory")
    parser.add_argument("--mute", action="store_true")

    args = parser.parse_args()
    return args

def load_point_cloud(data_dir, show_corres=False):
    A_corr_f = f'{data_dir}/query_corres.ply'
    A_raw_f = f'{data_dir}/query_raw_in_world.ply'
    B_corr_f = f'{data_dir}/cond_corres.ply'
    B_raw_f = f'{data_dir}/cond_raw.ply'

    # Check if files exist
    for f in [A_corr_f, A_raw_f, B_corr_f, B_raw_f]:
        if not os.path.exists(f):
            print(f"File {f} does not exist")
            exit(1)

    # Load point clouds
    A_corr_o3d = o3d.io.read_point_cloud(A_corr_f)
    B_corr_o3d = o3d.io.read_point_cloud(B_corr_f)
    A_raw_o3d = o3d.io.read_point_cloud(A_raw_f)
    B_raw_o3d = o3d.io.read_point_cloud(B_raw_f)

    A_corr = pcd2xyz(A_corr_o3d)
    A_raw = pcd2xyz(A_raw_o3d)
    B_corr = pcd2xyz(B_corr_o3d)
    B_raw = pcd2xyz(B_raw_o3d)
    num_corrs = A_corr.shape[1]
    print(f'Read {num_corrs} correspondences from {A_corr_f} and {B_corr_f}')

    if show_corres:
        points = np.concatenate((A_corr.T, B_corr.T), axis=0)
        lines = [[i, i + num_corrs] for i in range(num_corrs)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(A_corr_o3d.colors)
        o3d.visualization.draw_geometries([A_corr_o3d, B_corr_o3d, line_set])    

    return A_corr_o3d, B_corr_o3d, A_raw_o3d, B_raw_o3d, A_corr, A_raw, B_corr, B_raw

def align_ICP(P_origin, Q):
    # P_origin is the source point cloud
    # Q is the target point cloud
    delta = 0.05  # Adjust based on expected inlier noise level

    # Define rotation angles and axes
    angles = [-90, 90]  # Rotation angles in degrees
    axes = ['x', 'y', 'z']  # Axes to rotate around
    best_num_inliers = 0
    best_result = None

    # Include identity rotation
    # To initialize the optimization with pre-rotation
    rotation_combinations = [(None, 0)] + [(axis, angle) for axis in axes for angle in angles]

    for axis, angle in rotation_combinations:
        if axis is not None:
            R_pre = create_rotation_matrix(axis, angle)
            P_rotated = (R_pre @ P_origin.T).T  # Rotate the source point cloud
            print(f"Rotating around {axis}-axis by {angle} degrees")
        else:
            R_pre = np.eye(3)
            P_rotated = P_origin.copy()
            print("No pre-rotation applied")

        # Align point clouds using Huber loss
        s_opt, R_opt, t_opt, P_aligned = align_point_clouds_huber(P_rotated, Q, delta)
        
        # Compute errors and inliers
        errors = np.linalg.norm(P_aligned - Q, axis=1)
        inlier_mask = errors < delta
        num_inliers = np.sum(inlier_mask)
        mean_error = np.mean(errors)
        print(f"Number of inliers: {num_inliers}, Mean error: {mean_error}")

        # Update the best result based on the number of inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_result = {
                's_opt': s_opt,
                'R_opt': R_opt @ R_pre,
                't_opt': t_opt,
                'P_aligned': P_aligned,
                'errors': errors,
                'inlier_mask': inlier_mask,
                'rotation': (axis, angle)
            }

    if best_result is not None:
        # Extract the best alignment result
        s_opt = best_result['s_opt']
        R_opt = best_result['R_opt']
        t_opt = best_result['t_opt']
        P_aligned = best_result['P_aligned']
        errors = best_result['errors']
        inlier_mask = best_result['inlier_mask']
        rotation = best_result['rotation']
        print(f"\nBest alignment achieved with rotation around {rotation[0]}-axis by {rotation[1]} degrees")
    else:
        print("No alignment found")
        exit(1)

    # Evaluate alignment
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    num_inliers = np.sum(inlier_mask)
    print("\nFinal Alignment Results:")
    print("Estimated Scale (s_opt):", s_opt)
    print("\nEstimated Rotation Matrix (R_opt):\n", R_opt)
    print("\nEstimated Translation (t_opt):", t_opt)
    print("\nMean Alignment Error:", mean_error)
    print("Median Alignment Error:", median_error)
    print("\nNumber of Inliers:", num_inliers, "out of", P_origin.shape[0])        
    return s_opt, R_opt, t_opt, P_aligned.T


def transform_pc(pc, s_opt, R_opt, t_opt):
    # pc is a numpy 3xN array
    # s_opt is a scalar
    # R_opt is a 3x3 numpy array
    # t_opt is a 3x1 numpy array
    pc_s = pc * s_opt
    pc_R = R_opt @ pc_s
    pc_t = pc_R + t_opt.reshape(3, 1)
    return pc_t

def save_aligned_pcs(A_corr_o3d, B_corr_o3d, A_raw_o3d, A_corr_aligned, A_raw_aligned, out_dir, show_alignment):
    # Prepare point clouds for visualization
    A_corr_aligned_o3d = o3d.geometry.PointCloud()
    A_corr_aligned_o3d.points = o3d.utility.Vector3dVector(A_corr_aligned.T)
    A_corr_aligned_o3d.colors = A_corr_o3d.colors

    A_raw_aligned_o3d = o3d.geometry.PointCloud()
    A_raw_aligned_o3d.points = o3d.utility.Vector3dVector(A_raw_aligned.T)
    A_raw_aligned_o3d.colors = A_raw_o3d.colors

    if show_alignment:
        o3d.visualization.draw_geometries([A_corr_aligned_o3d, B_corr_o3d])

    # Save aligned point clouds
    A_corr_aligned_f = f'{out_dir}/query_corr_align.ply'
    A_raw_aligned_f = f'{out_dir}/query_raw_align.ply'
    o3d.io.write_point_cloud(A_corr_aligned_f, A_corr_aligned_o3d)
    print(f"Aligned correspondence point cloud saved to {A_corr_aligned_f}")
    o3d.io.write_point_cloud(A_raw_aligned_f, A_raw_aligned_o3d)
    print(f"Aligned raw point cloud saved to {A_raw_aligned_f}")

def save_aligned_transform(s_opt, R_opt, t_opt, out_cam_file):
    # Save transformation parameters
    T_opt = Rt2T(R_opt, t_opt)
    camera = {
        'que_blw2con_blw_T': T_opt.tolist(),
        'que_blw2con_blw_scale': [s_opt]
    }
    with open(out_cam_file, 'w') as file:
        json.dump(camera, file, indent=4)
    print(f"Predicted pose saved to {out_cam_file}")    
    
def load_2D_points(data_dir, camera_f, image_f, show_2D_points):
    """
    Load 2D keypoints, camera intrinsics and image for visualization.

    Args:
        data_dir (str): Directory containing the keypoint data
        camera_f (str): Path to camera intrinsics JSON file
        image_f (str): Path to condition image file 
        show_2D_points (bool): Whether to visualize the 2D keypoints

    Returns:
        image_points (np.ndarray): 2xN array of 2D keypoint coordinates
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix
    """

    image_points = np.load(f'{data_dir}/cond_kp.npy')

    condition_camera = json.load(open(camera_f))
    intrinsic = np.array(condition_camera["K"])
    w2c = np.array(condition_camera["blw2cvc"])
    condition_image = cv2.imread(image_f, cv2.IMREAD_UNCHANGED)


    
    # show the 2D points
    if show_2D_points:
        plt.imshow(condition_image)
        plt.scatter(image_points[:, 0], image_points[:, 1], c='red', marker='x')
        plt.show()


    return image_points.T, intrinsic, w2c

def save_projected_points(points_3D, points_2D, intrinsic, w2c, condition_image_f, out_f):
    # show the projected 3D points and 2D points in the image
    condition_image = cv2.imread(condition_image_f, cv2.IMREAD_UNCHANGED)
    condition_image = cv2.cvtColor(condition_image, cv2.COLOR_BGR2RGB)
    points_3D_c = transform_points(points_3D[None], w2c[None]).squeeze()
    points_3D_proj = project_points(points_3D_c[None], intrinsic[None]).squeeze()
    # get the residual of projected points and 2D points
    residual = np.abs(points_3D_proj[:, 0:2] - points_2D).mean()
    plt.imshow(condition_image)
    plt.scatter(points_2D[:, 0], points_2D[:, 1], c='red', marker='x')
    plt.scatter(points_3D_proj[:, 0], points_3D_proj[:, 1], c='blue', marker='*')
    plt.title(f'Projected Points Visualization\nRed: 2D keypoints, Blue: Projected 3D points\nResidual: {residual:.2f}')
    plt.savefig(out_f)
    plt.close()

def project_points_from_rodrigues(points_3D, rvec, tvec, intrinsic, w2c):
    """
    Project 3D points into 2D using the given rotation (Rodrigues),
    translation, and camera intrinsic matrix.
    
    points_3D: (3, N) numpy array
    rvec     : (3,)  Rodrigues rotation vector
    tvec     : (3,)  translation vector
    intrinsic: (3, 3) camera intrinsic matrix
    
    Returns: (2, N) numpy array of projected 2D points
    """
    R = rodrigues_to_rotation_matrix(rvec)
    # Transform 3D points: X_cam = R * X_world + t
    points_w = R @ points_3D + tvec.reshape(3, 1)

    points_c = transform_points(points_w.T[None], w2c[None]).squeeze().T
    # Avoid division by zero
    z = points_c[2, :]
    z[z == 0] = 1e-12
    
    # Project onto image plane
    # [u, v, 1]^T = K * [x/z, y/z, 1]^T
    uv_hom = intrinsic @ (points_c / z)
    uv = uv_hom[:2, :]
    
    return uv

def reprojection_residual(params, points_3D, points_2D, intrinsic, w2c):
    """
    Compute the reprojection error residuals for all points.
    
    params   : [rvec(3), tvec(3)]
    points_3D: (3, N)
    points_2D: (2, N)
    intrinsic: (3, 3)
    w2c: (4, 4)
    Returns: flattened 2D reprojection errors (size 2N,)
    """
    rvec = params[0:3]
    tvec = params[3:6]
    
    projected_2D = project_points_from_rodrigues(points_3D, rvec, tvec, intrinsic, w2c)
    
    # Residual: difference between observed points_2D and projected points
    residuals = np.abs(points_2D - projected_2D).ravel()
    # print(f"Residuals: {residuals.mean()}")
    return residuals

def align_PnP(points_3D, points_2D, intrinsic, w2c):
    """
    Optimize a 4x4 transformation matrix (homogeneous representation of
    rotation + translation) that maps world coordinates to camera coordinates,
    minimizing reprojection error.

    Args:
        points_3D : (3, N) 3D points
        points_2D : (2, N) 2D points
        intrinsic : (3, 3) camera intrinsics
        initial_w2c : (4, 4) initial transformation matrix from world to camera
    Returns:
        T_opt: (4,4) optimized transformation matrix
    """
    # rvec, tvec = transform4x4_to_rodrigues_and_translation(w2c)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    initial_guess = np.concatenate([rvec, tvec])
    # Perform non-linear least squares
    # Perform non-linear least squares optimization
    result = least_squares(
        fun   = reprojection_residual,
        x0    = initial_guess,
        args  = (points_3D, points_2D, intrinsic, w2c),
        method= 'lm'  # Levenberg-Marquardt
    )
    # breakpoint()
    # reprojection_residual(result.x, points_3D, points_2D, intrinsic)
    optimized_params = result.x
    rvec_optimized = optimized_params[0:3]
    tvec_optimized = optimized_params[3:6]
    R_optimized = rodrigues_to_rotation_matrix(rvec_optimized)
    return R_optimized, tvec_optimized

def combined_residual(params, P, Q, points_2D, K, w2c, w_3d=1.0, w_2d=1.0):
    """
    params: [s, rx, ry, rz, tx, ty, tz]
      - s   : scalar scale factor
      - rvec: (3,) rotation vector (Rodrigues)
      - t   : (3,) translation
    
    P: (N,3) source  3D points
    Q: (N,3) target  3D points
    points_2D: (N,2) 2D correspondences for P (after transform)
    K: (3,3) camera intrinsic matrix
    w2c: (4,4) world to camera transformation matrix
    w_3d, w_2d: weighting factors for 3D vs. 2D cost
    
    Returns a 1D vector of residuals for the optimizer:
        [res_3d, res_2d] all concatenated
    """
    # Extract parameters
    s = params[0]
    rvec = params[1:4]
    tvec = params[4:7]
    
    # Convert to a rotation matrix
    R = rotation_matrix_from_vector(rvec)
    
    # Transform P into "aligned" space:  P' = s * R*P + t
    #   P has shape (N,3) so do (R @ P.T).T for matrix multiply
    P_transformed = s * (R @ P.T).T + tvec  # shape (N,3)
    
    # 1) 3D alignment error: difference vs Q
    #    shape (N,3), flatten to (3N,)
    res_3d = (P_transformed - Q).ravel() * w_3d
    
    # 2) 2D reprojection error: difference vs points_2D
    #    project P_transformed with intrinsics
    P_transformed_c = transform_points(P_transformed[None], w2c[None]).squeeze()
    uv_projected = project_points(P_transformed_c[None], K[None]).squeeze()[:, 0:2]  # shape (N,2)
    res_2d = (points_2D - uv_projected).ravel() * w_2d
    
    print(f"Res 3D: {res_3d.mean()}, Res 2D: {res_2d.mean()}")
    # Combine them into a single 1D residual vector
    return np.concatenate([res_3d, res_2d])

def align_3D_and_2D_with_huber(
    P,           # (N,3) source 3D points
    Q,           # (N,3) target 3D points
    points_2D,   # (N,2) 2D correspondences
    K,           # (3,3) camera intrinsics
    w2c,         # (4,4) world to camera transformation matrix
    delta=1.0,   # Huber parameter f_scale
    s0=1.0,      # initial guess for scale
    rvec0=np.zeros(3),  # initial guess for rotation
    tvec0=np.zeros(3),  # initial guess for translation
    w_3d=1.0,    # weighting factor for 3D residual
    w_2d=1.0     # weighting factor for 2D residual
):
    """
    Solve a single optimization that aligns P->Q in 3D while also fitting
    the 2D reprojection of P->points_2D. Uses a robust Huber loss.
    
    Returns:
        s_opt, R_opt, t_opt, P_aligned
    """
    # Initial parameter guess: [scale, rvec(3), t(3)]
    params0 = np.hstack([s0, rvec0, tvec0])
    
    # Example: force scale > 0 by setting a lower bound = 0
    lower_bounds = [0] + [-np.inf]*6
    upper_bounds = [np.inf]*7
    
    # Call SciPy least_squares with robust (Huber) loss
    result = least_squares(
        fun=combined_residual,
        x0=params0,
        args=(P, Q, points_2D, K, w2c, w_3d, w_2d),
        method='trf',      # must be 'trf' or 'dogbox' for bounded problems
        loss='huber',      # robust Huber loss
        f_scale=delta,     # threshold parameter for Huber
        bounds=(lower_bounds, upper_bounds)
    )
    
    # Extract optimized parameters
    s_opt = result.x[0]
    rvec_opt = result.x[1:4]
    t_opt = result.x[4:7]
    R_opt = rotation_matrix_from_vector(rvec_opt)
    print(f"Residual: {result.cost}")
    # Transform P with the final solution
    P_aligned = s_opt * (R_opt @ P.T).T + t_opt
    
    return s_opt, R_opt, t_opt, P_aligned.T

def main():
    args = parse_args()
    
    # Visualization flags
    show_corres = False
    show_alignment = False
    show_2D_points = True
    if args.mute:
        show_corres = False
        show_alignment = False
        show_2D_points = False
    seq_name = args.seqence_name
    query_index = args.query_index
    data_dir = args.data_dir
    cond_camera_f = args.cond_camera_f
    cond_img_f = args.cond_img_f
    outputs = Path(f"{args.out_dir}")

    cmd = f"rm -rf {outputs} && mkdir -p {outputs}"
    os.system(cmd)

    A_corr_o3d, B_corr_o3d, A_raw_o3d, B_raw_o3d, A_corr, A_raw, B_corr, B_raw = load_point_cloud(data_dir, show_corres)

    B_2d_points, K, w2c = load_2D_points(data_dir, cond_camera_f, cond_img_f, show_2D_points)

    # s_opt_ICP, R_opt_ICP, t_opt_ICP, A_corr_align_ICP = align_ICP(A_corr.T, B_corr.T)
    # Transform the raw source point cloud
    # A_raw_align_ICP = transform_pc(A_raw, s_opt_ICP, R_opt_ICP, t_opt_ICP)
    
    
    save_projected_points(A_corr.T, B_2d_points.T, K, w2c,cond_img_f, outputs / f'icp_projected.png')
    # R_opt_PnP, t_opt_PnP = align_PnP(A_corr_align_ICP, B_2d_points, K, w2c); s_opt_PnP = 1.0
    s_opt_PnP, R_opt_PnP, t_opt_PnP, A_corr_align_PnP = align_3D_and_2D_with_huber(
        A_corr.T, B_corr.T, B_2d_points.T, K, w2c)

    A_raw_align_PnP = transform_pc(A_raw, s_opt_PnP, R_opt_PnP, t_opt_PnP)
    T_opt_PnP = Rt2T(R_opt_PnP, t_opt_PnP)
    # show_projected_points(A_corr_align_ICP.T, B_2d_points.T, K, T_opt_PnP, cond_img_f)
    save_projected_points(A_corr_align_PnP.T, B_2d_points.T, K, w2c, cond_img_f, outputs / f'pnp_projected.png')

    
    save_aligned_pcs(A_corr_o3d, B_corr_o3d, A_raw_o3d, A_corr_align_PnP, A_raw_align_PnP, outputs, show_alignment)
    # T_opt_ICP = Rt2T(R_opt_ICP, t_opt_ICP)
    s_opt = s_opt_PnP
    T_opt = T_opt_PnP
    R_opt = T_opt[:3, :3]
    t_opt = T_opt[:3, 3]
    # save the transformation parameters
    save_aligned_transform(s_opt, R_opt, t_opt, outputs / f'{query_index}.json')



if __name__ == "__main__":
    main()
