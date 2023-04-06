import argparse
import sys
import ast
sys.path = ["."] + sys.path
from src.colmap.colmap_utils import colmap_pose_est, colmap_pose_est_diff_object
from src.colmap.colmap_utils import colmap_mvs, convert_to_HO3D_format
from src.colmap.colmap_utils import validate_colmap
from src.colmap.colmap_utils import format_poses
from src.colmap.colmap_utils import validate_reprojection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name")
    parser.add_argument("--out_path", type=str, help="out path")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument(
        "--num_pairs",
        type=int,
        help="number of the frames that the model is searching for connections",
    )
    parser.add_argument("--no_vis", default=False, action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    seq_name = args.seq_name
    out_path = args.out_path


    print("Processing sequence", seq_name)
    convert_to_HO3D_format(out_path)    
