from glob import glob

import common.comet_utils as comet_utils
from common.exp_manager import ExpManager
from omegaconf import OmegaConf
import numpy as np


def parser_args():
    import argparse

    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./confs/general.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--shape_init", type=str, default="75268d864")
    parser.add_argument("--mute", help="No logging", action="store_true")
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--num_sample", type=int, default=128)
    parser.add_argument("--cluster_node", type=str, default="")
    parser.add_argument("--gpu_arch", type=str, default="ampere")
    parser.add_argument("--bid", type=int, default=151)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--memory", type=int, default=55000)
    parser.add_argument("--exp_key", type=str, default="")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--freeze_pose", action="store_true", help="no optimize pose")
    parser.add_argument("--barf_s", type=int, default=1000)

    parser.add_argument("--barf_e", type=int, default=10000)
    parser.add_argument("--no_barf", action="store_true", help="no barf")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--no_meshing", action="store_true")
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--render_downsample", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument(
        "-f",
        "--fast",
        dest="fast_dev_run",
        help="single batch for development",
        action="store_true",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--infer_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--load_pose",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--eval_every_epoch", type=int, default=6, help="Eval every k epochs"
    )
    parser.add_argument("--tempo_len", type=int, default=2000)
    parser.add_argument("--gpu_min_mem", type=int, default=32000)
    parser.add_argument("--dump_eval_meshes", action="store_true")
    args = parser.parse_args()
    args = edict(vars(args))
    opt = edict(OmegaConf.load(args.config))
    manager = ExpManager(args)
    cmd = " ".join(sys.argv)
    args.cmd = cmd
    args.project = "blaze"

    data = np.load(f"./data/{args.case}/build/data.npy", allow_pickle=True).item()
    opt.model.scene_bounding_sphere = data["scene_bounding_sphere"]

    # args.tempo_len = 2000
    if args.fast_dev_run:
        args.num_workers = 0
        args.eval_every_epoch = 1
        args.num_sample = 8
        args.tempo_len = 50
        args.log_every = 1

    args.total_step = int(
        args.num_epoch * args.tempo_len / opt.dataset.train.batch_size
    )

    cluster_cmd = args.cmd.replace("--cluster", "")
    experiment, args = comet_utils.init_experiment(
        args, api_key="gkEjEq3RpV8xNYREIdqxBJ3aw", workspace="zc-alexfan"
    )
    comet_utils.save_args(args, save_keys=["comet_key", "git_commit", "git_branch"])
    manager.run_experiment(use_cluster=args.cluster, script=cluster_cmd, num_exp=1)

    if experiment is not None:
        comet_utils.log_exp_meta(args)

    img_paths = sorted(glob(f"./data/{args.case}/build/image/*.png"))
    assert len(img_paths) > 0, "No images found"
    args.n_images = len(img_paths)
    return args, opt
