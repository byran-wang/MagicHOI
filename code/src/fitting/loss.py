import sys
import numpy as np
import torch

# Standard Libraries

# Third-party Libraries
import numpy as np
import torch


import pickle as pkl

from pytorch3d.ops import knn_points

# from src.fitting.generic import Generic
from src.fitting.utils import (
    l1_loss,
)

with open("./code/body_models/contact_zones.pkl", "rb") as f:
    contact_zones = pkl.load(f)
contact_zones = contact_zones["contact_zones"]
contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])


def loss_fn_h(out, targets, flag):
    v3d_h_c = out[f"{flag}.v3d_c"]
    v3d_o_c = out["object.v3d_c"]
    v3d_tips = v3d_h_c[:, contact_idx]

    # contact
    loss_fine_ho = knn_points(v3d_tips, v3d_o_c, K=1, return_nn=False)[0].mean()

    mask_h = out[f"{flag}.mask"]
    mask_o = out["object.mask"]

    valid_pix = 1 - targets[flag]
    err = l1_loss(mask_o, targets["object"]) * valid_pix
    loss_mask_o = torch.sum(err) / torch.sum(valid_pix)

    valid_pix = 1 - targets["object"]
    err = l1_loss(mask_h, targets[flag]) * valid_pix
    loss_mask_h = torch.sum(err) / torch.sum(valid_pix)

    loss_dict = {}
    loss_dict["mask_o"] = loss_mask_o * 0
    loss_dict["mask_h"] = loss_mask_h * 1000
    loss_dict["fine_ho"] = loss_fine_ho * 1000.0

    loss = sum(loss_dict.values())
    loss_dict["loss"] = loss
    return loss_dict


def loss_fn_rh(out, targets):
    return loss_fn_h(out, targets, "right")


def loss_fn_lh(out, targets):
    return loss_fn_h(out, targets, "left")


def loss_fn_ih(out, targets):
    loss_dict_r = loss_fn_h(out, targets, "right")
    loss_dict_r.pop("loss")
    loss_dict_l = loss_fn_h(out, targets, "left")
    loss_dict_l.pop("loss")
    loss_dict = {k: (v + loss_dict_l[k]) / 2 for k, v in loss_dict_r.items()}
    loss = sum(loss_dict.values())
    loss_dict["loss"] = loss
    return loss_dict
