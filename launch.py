import argparse
import contextlib
import importlib
import logging
import os
import sys
import time
import traceback


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def load_custom_module(module_path):
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
        else:
            module_spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(module_path, "__init__.py")
            )

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(traceback.format_exc())
        print(f"Cannot import {module_path} module for custom nodes:", e)
        return False


def load_custom_modules():
    node_paths = ["custom"]
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if (
                os.path.isfile(module_path)
                and os.path.splitext(module_path)[1] != ".py"
            ):
                continue
            if module_path.endswith("_disabled"):
                continue
            time_before = time.perf_counter()
            success = load_custom_module(module_path)
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, success)
            )

    if len(node_import_times) > 0:
        print("\nImport times for custom modules:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

import re
import numpy as np
import sys
sys.path.append("third_party/utils_simba")
from utils_simba.data_read import ReadHO3DFoundationPose
from utils_simba.pose import cartesian_to_spherical
import pickle
import math
import cv2
from glob import glob

def get_inpaint_view_info(cfg):
    cond_rgba_f_ind = int(re.findall(r'\d+', os.path.basename(cfg.data.image_path))[0])
    cam_infos = ReadHO3DFoundationPose(cfg.data.real_image_foundation_pose)
    
    for cam_info in cam_infos:
        if cam_info.uid == cond_rgba_f_ind:
            cond_c2w4x4 = cam_info.c2w4x4
            cond_fovy = cam_info.fov_y
            break
    return cond_c2w4x4, cond_fovy

def update_cfg(cfg): 
    if cfg.data.data_type == "real_image_foundation_pose":
        cfg.data.image_path = cfg.data.real_image_foundation_pose.inpaint_f
        if cfg.data.cond_pose_from_selected_cam:
            cond_c2w4x4, cond_fovy  = get_inpaint_view_info(cfg)
            inpaint_shpere= cartesian_to_spherical(cond_c2w4x4[:3,3][None])
            cfg.data.default_elevation_deg = inpaint_shpere['elevation']
            cfg.data.default_azimuth_deg = inpaint_shpere['azimuth']
            cfg.data.default_fovy_deg = cond_fovy * 180 / np.pi
        # else:
        #     with open(cfg.data.real_image_foundation_pose.inpaint_pose, 'r+') as f:
        #         import json
        #         cond_pose = json.load(f)
        #         if cfg.data.user_elevation_deg > -1000:
        #             cond_pose['elevation_deg'] = cfg.data.user_elevation_deg
        #         cfg.data.default_elevation_deg = cond_pose['elevation_deg']
        #         cfg.data.default_azimuth_deg = cond_pose['azimuth_deg']
        #         cfg.data.default_fovy_deg = cond_pose['fovy_deg']
        #         cfg.data.default_camera_distance = cond_pose['distance']
        #         f.seek(0)
        #         json.dump(cond_pose, f, indent=4)
        #         f.truncate()
    azim_delta = 90
    cfg.system.guidance.cond_image_path = cfg.data.image_path
    cfg.data.random_camera.eval_elevation_deg = cfg.data.default_elevation_deg
    cfg.system.guidance.cond_elevation_deg = cfg.data.default_elevation_deg
    cfg.system.guidance.cond_azimuth_deg = cfg.data.default_azimuth_deg
    cfg.system.guidance.cond_camera_distance = cfg.data.default_camera_distance
    cfg.data.random_camera.fovy_range = [cfg.data.default_fovy_deg, cfg.data.default_fovy_deg]
    cfg.data.random_camera.eval_fovy_deg = cfg.data.default_fovy_deg
    cfg.data.random_camera.elevation_range = [-10, 80]
    cfg.data.random_camera.default_elevation_deg = cfg.data.default_elevation_deg
    cfg.data.random_camera.default_azimuth_deg = cfg.data.default_azimuth_deg
    cfg.data.random_camera.default_camera_distance = cfg.data.default_camera_distance
    if cfg.data.align.do_3d_guidance_only and not cfg.system.freq.do_ref_only:
        cfg.system.freq.ref_only_steps = -1
        cfg.data.random_camera.azimuth_range = [-180, 180]
        # cfg.data.random_camera.azimuth_range = [cfg.data.default_azimuth_deg-azim_delta, cfg.data.default_azimuth_deg+azim_delta]
        cfg.data.random_camera.camera_distance_range = [cfg.data.default_camera_distance, cfg.data.default_camera_distance]
    else:
        cfg.data.random_camera.azimuth_range = [-180, 180]
        # cfg.data.random_camera.azimuth_range = [cfg.data.default_azimuth_deg-azim_delta, cfg.data.default_azimuth_deg+azim_delta
        cfg.data.random_camera.camera_distance_range = [cfg.data.default_camera_distance, cfg.data.default_camera_distance]
    print(f"cond_elevation_deg: {cfg.system.guidance.cond_elevation_deg}, cond_azimuth_deg: {cfg.system.guidance.cond_azimuth_deg}, cond_camera_distance: {cfg.system.guidance.cond_camera_distance}, default_fovy_deg: {cfg.data.default_fovy_deg}")

def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # load_custom_modules()

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)
    update_cfg(cfg)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            # CodeSnapshotCallback(
            #     os.path.join(cfg.trial_dir, "code"), use_version=False
            # ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
