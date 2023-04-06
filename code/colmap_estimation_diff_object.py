import argparse
import sys
import ast
sys.path = ["."] + sys.path
from src.colmap.colmap_utils import colmap_pose_est_diff_object
from src.colmap.colmap_utils import colmap_mvs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name")
    parser.add_argument("--selected_views", type=str, help="selected views")
    parser.add_argument("--data_path", type=str, help="data path")
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
    num_pairs = args.num_pairs
    no_vis = args.no_vis
    data_path = args.data_path
    out_path = args.out_path
    selected_views = args.selected_views
    mute = args.mute
    selected_views = ast.literal_eval(selected_views)

    print("Processing sequence", seq_name)
    colmap_pose_est_diff_object(selected_views, data_path, out_path, num_pairs, mute)
    colmap_mvs(data_path, out_path)
