import sys

sys.path = ["../"] + sys.path
from common.viewer import HOLDViewer
import os.path as op
import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.mesh import transform_vertices


class DataViewer(HOLDViewer):
    def __init__(
        self,
        render_types=["rgb", "meshes", "video"], # options: ["rgb", "meshes", "video", "depth", "mask"]
        interactive=True,
        size=(2024, 2024),
    ):
        super().__init__(render_types, interactive, size)


def fetch_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_p", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--gt_ho3d", action="store_true")
    parser.add_argument("--gt_arctic", action="store_true")
    parser.add_argument("--ours", action="store_true")
    parser.add_argument("--ours_left", action="store_true")
    parser.add_argument("--ours_implicit", action="store_true")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--obj_mesh_f", type=str, default=None)
    parser.add_argument("--obj_pose_f", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out_folder", type=str, default="render_out")
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def merge_batch(batch_list):
    if len(batch_list) == 1:
        return batch_list[0][0], batch_list[0][1]
    meshes = {}
    for batch in batch_list:
        meshes.update(batch[0])
    batch_list = batch_list[:-1]
    data_gt = batch_list[-1][1]  # from gt data
    return meshes, data_gt


def main():
    args = fetch_parser()
    interactive = True
    if args.headless:
        interactive = False
    viewer = DataViewer(interactive=interactive, size=(2024, 2024))
    batch = None
    right_batch = None
    gt_batch = None
    left_batch = None
    if args.ours:
        import src.utils.io.ours as ours
        right_batch = ours.load_viewer_data_magicHOI(args.obj_mesh_f, args.ckpt_p, args.data_root, args.obj_pose_f, is_right=True)
        if args.ours_left:
            left_batch = ours.load_viewer_data_magicHOI(args.obj_mesh_f, args.ckpt_p, args.data_root, args.obj_pose_f, is_right=False)

    if args.gt_ho3d:
        import src.utils.io.gt as gt
        batch = gt.load_viewer_data(args)
        gt_batch = batch

    if args.ours and args.gt_ho3d:
        ours_K = right_batch[1]["K"]
        gt_K = gt_batch[1]["K"]
        ours_meshes = right_batch[0]
        # Transform object vertices
        ours_meshes["object"].vertices = transform_vertices(ours_meshes["object"].vertices, ours_K, gt_K)        
        # Transform hand vertices
        ours_meshes["right"].vertices = transform_vertices(ours_meshes["right"].vertices, ours_K, gt_K)

        right_batch = (ours_meshes, right_batch[1])
    
    batch_list = []
    if gt_batch is not None:
        batch_list.append(gt_batch) # gt camera would be used for visualization
    if right_batch is not None:
        batch_list.append(right_batch)
    if left_batch is not None:
        batch_list.append(left_batch)
        
    if args.gt_arctic:
        import src.utils.io.gt_arctic as gt_arctic

        batch = gt_arctic.load_viewer_data(args)
        batch_list.append(batch)

    # if len(batch_list) > 1:
    batch = merge_batch(batch_list)
    viewer.render_seq(batch, out_folder=args.out_folder)


if __name__ == "__main__":
    main()
