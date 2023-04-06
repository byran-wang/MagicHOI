import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import tqdm
import tqdm.notebook
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Make tqdm notebook-friendly
tqdm.tqdm = tqdm.notebook.tqdm

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.utils.viz import add_text, cm_RdGn, plot_keypoints, plot_matches
from hloc.utils.io import get_keypoints, get_matches

import sys
sys.path.append("third_party/utils_simba")
from utils_simba.geometry import read_point_cloud_from_ply, save_point_cloud_to_ply
from utils_simba.depth import get_depth, depth2xyzmap

# Visualization functions

def plot_features(
        image_dir: Path,
        image_name: str,        
        keypoints: np.ndarray,
        color = (0, 0, 1),
        dpi=75,                   
    ) -> None:
    """Plot detected features on an image."""
    color = [color] * len(keypoints)
    text = f"feature number: {len(keypoints)}"
    plot_images([read_image(image_dir / image_name)], dpi=dpi)
    plot_keypoints([keypoints], colors=[color], ps=4)
    add_text(0, text)
    add_text(0, image_name, pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")

def plot_repjct(
        image_dir: Path,
        image_name: str,        
        kpts: np.ndarray,
        prjpts: np.ndarray,
        inliers: np.ndarray,
        kpts_color = (0, 1, 0),
        prjpts_color = (1, 0, 0),
        inliers_color = (0, 0, 1),
        dpi=75,
        ps=6,                   
    ) -> None:
    """Plot reprojection points on an image."""
    kpts_colors = [kpts_color] * len(kpts)
    prjpts_colors = [prjpts_color] * len(prjpts)
    inliers_colors = [inliers_color] * len(inliers)

    text = f"kpts: {len(kpts)}/{kpts_color}, prjpts: {len(prjpts)}/{prjpts_color}, inliers: {len(inliers)}/{inliers_color}"
    plot_images([read_image(image_dir / image_name)], dpi=dpi)
    plot_keypoints([kpts], colors=[kpts_colors], ps=ps)
    plot_keypoints([prjpts], colors=[prjpts_colors], ps=ps)
    plot_keypoints([inliers], colors=[inliers_colors], ps=ps)
    add_text(0, text)
    add_text(0, image_name, pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")

def plot_and_save_matches(
        in_dir: Path, 
        out_dir: Path,
        query_img_f: str, 
        cond_img_f: str, 
        kp_match_query: np.ndarray, 
        kp_match_cond: np.ndarray, 
        matches_count: int, 
        total_keypoints: int,
        show_feature_matches: bool
    ) -> None:
    """Plot and save feature matches between two images."""
    plot_images([read_image(in_dir / query_img_f), read_image(in_dir / cond_img_f)], dpi=75)
    plot_matches(kp_match_query, kp_match_cond, color=None, lw=3, a=0.1)
    text = f"inliers: {matches_count}/{total_keypoints}"
    add_text(0, text)
    opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")
    add_text(0, query_img_f, **opts)
    add_text(1, cond_img_f, **opts)
    plt.savefig(out_dir / "matches.png")
    if show_feature_matches: 
        plt.show()
    else:
        plt.close()

def get_camera_info(camera_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera information from a JSON file."""
    camera = json.load(open(camera_file, 'r'))
    cvc2blw = np.linalg.inv(camera['blw2cvc'])
    K = np.array(camera['K']).reshape(3, 3)
    return cvc2blw, K

def process_correspondences(
        pts3D: np.ndarray, 
        color: np.ndarray, 
        keypoints: np.ndarray, 
        indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract 3D points, colors, and keypoints at match indices."""
    pts3D_matches = pts3D[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
    color_matches = color[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
    return pts3D_matches, color_matches, keypoints

def filter_valid_matches(
        cond_pts3D_matches: np.ndarray, 
        query_pts3D_matches: np.ndarray, 
        query_color_matches: np.ndarray,
        cond_color_matches: np.ndarray,
        kp_match_query: np.ndarray,
        kp_match_cond: np.ndarray,
        min_depth: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter matches based on depth validity."""
    valid_indices = []
    
    for i, (cond_pts3D_match, query_pts3D_match) in enumerate(zip(cond_pts3D_matches, query_pts3D_matches)):
        if cond_pts3D_match[2] <= min_depth or query_pts3D_match[2] <= min_depth:
            continue
        valid_indices.append(i)
    
    cond_pts3D_val_matches = cond_pts3D_matches[valid_indices]
    query_pts3D_val_matches = query_pts3D_matches[valid_indices]
    query_color_val_matches = query_color_matches[valid_indices]
    cond_color_val_matches = cond_color_matches[valid_indices]
    kp_match_val_query = kp_match_query[valid_indices]
    kp_match_val_cond = kp_match_cond[valid_indices]
    
    return (cond_pts3D_val_matches, query_pts3D_val_matches, query_color_val_matches, 
            cond_color_val_matches, kp_match_val_query, kp_match_val_cond)

def transform_to_world(
        pts3D: np.ndarray, 
        cvc2blw: np.ndarray
    ) -> np.ndarray:
    """Transform 3D points from camera to world coordinates."""
    pts3D_in_world = cvc2blw[:3, :3] @ pts3D.reshape(-1, 3).T + cvc2blw[:3, 3:4]
    return pts3D_in_world.T

def save_results(
        out_dir: Path,
        cond_pts3D_val_matches: np.ndarray,
        cond_pts3D: np.ndarray,
        cond_color_val_matches: np.ndarray,
        cond_color: np.ndarray,
        kp_match_val_cond: np.ndarray,
        query_pts3D_val_matches: np.ndarray,
        query_pts3D: np.ndarray,
        query_pts3D_in_world: np.ndarray,
        query_color_val_matches: np.ndarray,
        query_color: np.ndarray,
        kp_match_val_query: np.ndarray
    ) -> None:
    """Save point clouds and keypoints to disk."""
    # Save condition data
    cond_match_name = "cond_corres.ply"
    cond_raw_name = "cond_raw.ply"
    save_point_cloud_to_ply(cond_pts3D_val_matches, out_dir/cond_match_name, colors=cond_color_val_matches)
    save_point_cloud_to_ply(cond_pts3D.reshape(-1, 3), out_dir/cond_raw_name, colors=cond_color.reshape(-1, 3))
    cond_kp_name = "cond_kp.npy"
    np.save(out_dir/cond_kp_name, kp_match_val_cond)
    print(f"Condition keypoints saved to {out_dir/cond_kp_name}")

    # Save query data
    query_match_name = "query_corres.ply"
    query_raw_name = "query_raw_in_cam.ply"
    query_raw_in_world_name = "query_raw_in_world.ply"
    save_point_cloud_to_ply(query_pts3D_val_matches, out_dir/query_match_name, colors=query_color_val_matches)
    save_point_cloud_to_ply(query_pts3D.reshape(-1, 3), out_dir/query_raw_name, colors=query_color.reshape(-1, 3))
    save_point_cloud_to_ply(query_pts3D_in_world.reshape(-1, 3), out_dir/query_raw_in_world_name, colors=query_color.reshape(-1, 3))
    query_kp_name = "query_kp.npy"
    np.save(out_dir/query_kp_name, kp_match_val_query)
    print(f"Query keypoints saved to {out_dir/query_kp_name}")

def extract_and_match_features(
        in_dir: Path, 
        out_dir: Path,
        cond_img_f: str,
        query_img_f: str,
        score_threshold: float,
        show_flags: Dict[str, bool]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Extract and match features between condition and query images."""
    features_path = out_dir / "features.h5"
    matches_path = out_dir / "matches.h5"
    loc_pairs_path = out_dir / "pairs-loc.txt"
    
    feature_conf = extract_features.confs["superpoint_inloc"]
    matcher_conf = match_features.confs["superglue"]
    
    # Extract features for condition image
    cond_view_feature = []
    extract_features.main(
        feature_conf, in_dir, image_list=[cond_img_f], feature_path=features_path, features_out=cond_view_feature
    )
    
    # Plot condition features
    fig = viz_3d.init_figure()
    plot_features(in_dir, cond_img_f, cond_view_feature[0]['keypoints'])
    plt.savefig(out_dir / "cond_view_features.png")
    if show_flags['cond_view_features']:
        plt.show()
    else:
        plt.close()
    
    # Extract features for query image
    query_view_feature = []
    extract_features.main(
        feature_conf, in_dir, image_list=[query_img_f], feature_path=features_path, 
        overwrite=True, features_out=query_view_feature
    )
    
    # Plot query features
    plot_features(in_dir, query_img_f, query_view_feature[0]['keypoints'])
    plt.savefig(out_dir / "query_view_features.png")
    if show_flags['query_view_features']:
        plt.show()
    else:
        plt.close()
    
    # Match features
    pairs_from_exhaustive.main(loc_pairs_path, image_list=[query_img_f], ref_list=[cond_img_f])
    matches_output = []
    match_features.main(
        matcher_conf, loc_pairs_path, features=features_path, matches=matches_path, 
        overwrite=True, matches_output=matches_output
    )
    
    # Get matches and filter by score
    matches, scores = get_matches(matches_path, query_img_f, cond_img_f)
    valid_matches = matches[scores > score_threshold]
    
    # Get matched keypoints
    kp_match_query = (query_view_feature[0]['keypoints'][valid_matches[:,0]]).astype(np.float32)
    kp_match_cond = (cond_view_feature[0]['keypoints'][valid_matches[:,1]]).astype(np.float32)
    
    # Plot matches
    plot_and_save_matches(
        in_dir, out_dir, query_img_f, cond_img_f, 
        kp_match_query, kp_match_cond,
        len(valid_matches), len(query_view_feature[0]['keypoints']),
        show_flags['feature_matches']
    )
    
    return cond_view_feature, query_view_feature, kp_match_query, kp_match_cond

def verify_inputs(args):
    """Verify that all input files exist."""
    required_files = [
        (args.cond_img_f, "condition image"),
        (args.cond_depth_f, "condition depth"),
        (args.cond_camera_f, "condition camera"),
        (args.query_img_f, "query image"),
        (args.query_depth_f, "query depth"),
        (args.query_camera_f, "query camera")
    ]
    
    for file_path, file_type in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} file {file_path} does not exist")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqence_name", required=False, help="sequence name")
    parser.add_argument("--cond_img_f", required=False, help="condition image file")
    parser.add_argument("--cond_depth_f", required=False, help="condition depth file")
    parser.add_argument("--cond_camera_f", required=False, help="condition camera file")
    parser.add_argument("--query_img_f", required=False, help="query image file")
    parser.add_argument("--query_depth_f", required=False, help="query depth file")
    parser.add_argument("--query_camera_f", required=False, help="query camera file")
    parser.add_argument("--in_dir", required=False, help="data directory")
    parser.add_argument("--out_dir", required=False, help="output directory")
    parser.add_argument("--score_threshold", required=False, type=float, default=0.3, help="score threshold")
    parser.add_argument("--mute", action="store_true", help="Disable all visualizations")
    parser.add_argument("--convert_to_world", action="store_true", default=True, 
                       help="Convert points to world coordinates")
    
    return parser.parse_known_args()

def main():
    print("+++++++++ begin ++++++++++")
    
    # Parse arguments
    args, extras = parse_arguments()
    
    # Setup paths
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    
    # Verify inputs
    verify_inputs(args)
    
    # Create output directory
    cmd = f"rm -rf {out_dir} && mkdir -p {out_dir}"
    os.system(cmd)
    
    # Set visualization flags
    show_flags = {
        'cond_view_features': not args.mute,
        'query_view_features': not args.mute,
        'feature_matches': not args.mute
    }
    
    # Extract and match features
    _, _, kp_match_query, kp_match_cond = extract_and_match_features(
        in_dir, out_dir, args.cond_img_f, args.query_img_f, args.score_threshold, show_flags
    )
    
    # Get camera parameters
    cond_cvc2blw, cond_K = get_camera_info(args.cond_camera_f)
    query_cvc2blw, query_K = get_camera_info(args.query_camera_f)
    
    # Load depth and color images
    cond_depth = get_depth(str(args.cond_depth_f))
    cond_color = cv2.cvtColor(cv2.imread(args.cond_img_f), cv2.COLOR_BGR2RGB)
    query_depth = get_depth(args.query_depth_f)
    query_color = cv2.cvtColor(cv2.imread(args.query_img_f), cv2.COLOR_BGR2RGB)
    
    # Get 3D points
    cond_pts3D = depth2xyzmap(cond_depth, cond_K)
    query_pts3D = depth2xyzmap(query_depth, query_K)
    
    # Get 3D points for matches
    cond_pts3D_matches, cond_color_matches, _ = process_correspondences(
        cond_pts3D, cond_color, kp_match_cond, kp_match_cond
    )
    query_pts3D_matches, query_color_matches, _ = process_correspondences(
        query_pts3D, query_color, kp_match_query, kp_match_query
    )
    
    # Filter valid matches
    (cond_pts3D_val_matches, query_pts3D_val_matches, query_color_val_matches,
     cond_color_val_matches, kp_match_val_query, kp_match_val_cond) = filter_valid_matches(
        cond_pts3D_matches, query_pts3D_matches, query_color_matches,
        cond_color_matches, kp_match_query, kp_match_cond
    )
    
    # Transform to world coordinates if needed
    if args.convert_to_world:
        query_pts3D_val_matches = transform_to_world(query_pts3D_val_matches, query_cvc2blw)
        query_pts3D_in_world = transform_to_world(query_pts3D, query_cvc2blw)
        cond_pts3D_val_matches = transform_to_world(cond_pts3D_val_matches, cond_cvc2blw)
        cond_pts3D = transform_to_world(cond_pts3D, cond_cvc2blw)
    else:
        query_pts3D_in_world = query_pts3D.reshape(-1, 3)
    
    # Save results
    save_results(
        out_dir,
        cond_pts3D_val_matches, cond_pts3D, cond_color_val_matches, cond_color,
        kp_match_val_cond, query_pts3D_val_matches, query_pts3D, query_pts3D_in_world,
        query_color_val_matches, query_color, kp_match_val_query
    )

if __name__ == "__main__":
    main()


