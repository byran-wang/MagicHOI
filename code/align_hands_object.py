import os

import torch
import pytorch_lightning as pl
import os.path as op
import numpy as np
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from src.alignment.data import read_data
from src.alignment.pl_module.ray_hit import RayHit
from common.xdict import xdict
from src.alignment.data import FakeDataset
WANDB_ENABLED = False

class SaveCheckpointBeforeOptimization(Callback):
    def __init__(self, dirpath=""):
        self.save_dir = dirpath
        
    def on_train_start(self, trainer, pl_module):
        # Save the checkpoint before the very first optimization
        if args.mode == "before":
            trainer.save_checkpoint(f"{self.save_dir}/last.ckpt")
            print(f"Saved {self.save_dir}/last.ckpt")
            trainer.should_stop = True

def main(args):
    device = "cuda"
    if args.colmap_path == "":
        k_path_colmap = op.join(f"./data/{args.seq_name}/processed/colmap/intrinsic.npy")
    else:
        k_path_colmap = op.join(f"{args.colmap_path}/intrinsic.npy")
    # ho3d_seq = args.seq_name.split("_")[1]
    # k_path_ho3d = f"./assets/datasets/HO3D_v3/{ho3d_seq}.pt"
    # if op.exists(k_path_ho3d):
    #     K = torch.load(k_path_ho3d)["K"][0]
    # else:
    K = torch.FloatTensor(np.load(k_path_colmap))
    data = read_data(args.seq_name, K, args, colmap_path=args.colmap_path).to(device)
    
    out_p = op.join(f"{args.out_dir}/hold_fit.aligned.npy")

    mano_path = f"{args.out_dir}/mano_fit_ckpt/{args.mode}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=op.join(mano_path),
        save_last=True,
    )
    checkpoint_boefore_opti_callback = SaveCheckpointBeforeOptimization(
        dirpath=op.join(mano_path),
    )    
    os.makedirs(op.dirname(out_p), exist_ok=True)
    # Initialize WandbLogger
    if WANDB_ENABLED:
        wandb_logger = WandbLogger(project='MagicHOI', name='align_hands_object')
    else:
        wandb_logger = False

    conf = load_conf(args)
    if args.is_arctic:
        from src.alignment.pl_module.arctic import ARCTICModule as PLModule
        print("Using ARCTIC module..")
    elif len(data['entities']) == 3:
        from src.alignment.pl_module.h2o import H2OModule as PLModule
        print("Using H2O module..")
    else:
        from src.alignment.pl_module.ho import HOModule as PLModule
        print("Using HO module..")
    pl_model = PLModule(data, args, conf)
    trainer = pl.Trainer(
        accelerator="gpu",
        gradient_clip_val=0.5,
        max_epochs=1,
        logger=wandb_logger,        # Pass in the wandb logger
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, checkpoint_boefore_opti_callback],
    )

    dataset = FakeDataset(conf.num_iters)
    trainset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if args.mode == 'h' or args.mode == 'before':
        load_ckpt = None
    elif args.mode == 'o' or args.mode == 'r':
        load_ckpt = f"{args.out_dir}/mano_fit_ckpt/h/last.ckpt"
    elif args.mode == 'ho':
        load_ckpt = f"{args.out_dir}/mano_fit_ckpt/o/last.ckpt"
    else:
        assert False, f"Invalid args.mode {args.mode}"

    if load_ckpt is not None:
        sd = torch.load(load_ckpt)["state_dict"]
        pl_model.load_state_dict(sd)
        print(f"Loaded hand model from {load_ckpt}")
    if args.mode == 'r':
        preds = xdict()
        for key in pl_model.entities.keys():
            preds.merge(pl_model.models[key]().prefix(key + '.'))   
        ray_hit = RayHit(save_dir=f"{args.out_dir}/mano_fit_ckpt/r/")
        ray_hit.fwd(preds, data['meta'])
        ray_hit.save_data()
        return

    trainer.fit(pl_model, trainset)
    pl_model.to("cuda")
    out = xdict()
    for key in pl_model.models.keys():
        out[key] = pl_model.models[key]()
    if 'sdf' in out['object']:
        del out['object']['sdf']
    out = out.to("cpu").to_np()
    np.save(out_p, out)
    print(f"Saved to {out_p}")


def load_conf(args):
    conf_generic = OmegaConf.load(args.config)
    conf_path = f"./confs/{args.seq_name}.yaml"
    if op.exists(conf_path):
        conf_curr = OmegaConf.load(conf_path)
        config = OmegaConf.merge(conf_generic, conf_curr)
    else:
        config = conf_generic
    config = edict(OmegaConf.to_container(config, resolve=True))

    if args.mode == "h":
        conf = config["optim_configs"]["hand_optim"]
    elif args.mode == "r":
        conf = config["optim_configs"]["ray_hit"]        
    elif args.mode == "o":
        conf = config["optim_configs"]["object_optim"]
    elif args.mode == "ho":
        conf = config["optim_configs"]["hand_object_optim"]
    elif args.mode == "before":
        conf = config["optim_configs"]["save_before_optimization"]        
    else:
        raise NotImplementedError

    conf.update(config.weights)
    return conf


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--colmap_k", action="store_true")
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument('--config', type=str, default='code/confs/generic.yaml')
    parser.add_argument('--is_arctic', action='store_true')
    parser.add_argument("--colmap_path", type=str, default="")
    parser.add_argument("--object_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()
    args = edict(vars(args))
    dirs = os.listdir((args.object_dir + "/save"))
    mesh_dir = next((item for item in dirs if 'export' in item), None)
    args["object_mesh_f"] = (args.object_dir + "/save/" + mesh_dir + "/model.obj")
    args["object_ckpt_f"] = (args.object_dir + "/ckpts/last.ckpt")
    args["object_cfg_f"] = (args.object_dir + "/configs/parsed.yaml")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
