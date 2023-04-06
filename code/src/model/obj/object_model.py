import sys

import torch
import torch.nn as nn
import numpy as np

sys.path = [".."] + sys.path
from common.rot import axis_angle_to_matrix


class ObjectModel(nn.Module):
    def __init__(self, seq_name, template=None):
        super(ObjectModel, self).__init__()

        data = np.load(f"./data/{seq_name}/build/data.npy", allow_pickle=True).item()[
            "entities"
        ]["object"]
        if template is None:
            v3d_cano = torch.FloatTensor(data["pts.cano"])
        else:
            v3d_cano = torch.FloatTensor(template.vertices)
        self.register_buffer(
            "obj_scale", torch.FloatTensor(np.array([data["obj_scale"]]))
        )
        self.register_buffer("v3d_cano", v3d_cano)
        self.register_buffer("norm_mat", torch.FloatTensor(data["norm_mat"]))
        self.register_buffer("denorm_mat", torch.inverse(self.norm_mat))

    def forward(self, rot, trans, scene_scale=None):
        device = self.v3d_cano.device

        batch_size = rot.shape[0]
        # rot = torch.zeros_like(rot)
        # trans = torch.zeros_like(trans)
        # rot.data.zero_()
        # trans.data.zero_()
        rot_mat = axis_angle_to_matrix(rot).view(batch_size, 3, 3)

        # object to camera
        batch_size = rot_mat.shape[0]
        tf_mats = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        tf_mats[:, :3, :3] = rot_mat
        tf_mats[:, :3, 3] = trans.view(batch_size, 3)

        v3d_cano_pad = torch.cat(
            [self.v3d_cano, torch.ones(self.v3d_cano.shape[0], 1, device=device)],
            dim=1,
        )
        v3d_cano_pad = v3d_cano_pad[None, :, :].repeat(batch_size, 1, 1)

        vertices = torch.bmm(tf_mats, v3d_cano_pad.permute(0, 2, 1)).permute(0, 2, 1)
        out = {}
        out["vertices"] = vertices[:, :, :3] # in camera space
        out["T"] = tf_mats # from object space to camera space
        return out
