import torch
import torch.nn as nn
import smplx
from common.xdict import xdict
from common.transforms import project2d_batch
from src.alignment.loss_terms import gmof
from utils_simba.geometry import transform_points

import pickle as pkl
import numpy as np

import cv2
MANO_DIR_L = "./code/body_models/MANO_LEFT.pkl"
MANO_DIR_R = "./code/body_models/MANO_RIGHT.pkl"


class MANOParameters(nn.Module):
    def __init__(self, data, meta, is_right):
        super().__init__()

        K = meta["K"]
        num_frames = len(data["global_orient"])
        transl = torch.zeros(num_frames, 3)
        transl[:, 2] = 1.0  # init in front of camera (opencv)        

        # register parameters
        betas = nn.Parameter(data["betas"].mean(dim=0))
        global_orient = nn.Parameter(data["global_orient"])
        pose = nn.Parameter(data["hand_pose"])
        transl = nn.Parameter(transl)
        self.register_parameter("hand_beta", betas)
        self.register_parameter("hand_rot", global_orient)
        self.register_parameter("hand_pose", pose)
        self.register_parameter("hand_transl", transl)
        self.register_parameter("hand_scale", nn.Parameter(torch.tensor(1.0)))

        MANO_DIR = MANO_DIR_R if is_right else MANO_DIR_L
        self.mano_layer = smplx.create(
            model_path=MANO_DIR, model_type="mano", use_pca=False, is_rhand=is_right
        )
        self.f3d = torch.from_numpy(self.mano_layer.faces.astype(np.int32)).to("cuda")

        self.K = meta["K"]
        self.o2c = torch.tensor(meta["o2c"]).to("cuda")

        with open("./code/body_models/contact_zones.pkl", "rb") as f:
            contact_zones = pkl.load(f)
            contact_zones = contact_zones["contact_zones"]
            contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])
            self.contact_idx = contact_idx[19:]
            self.contact_idx_index_finger = contact_idx[19:47]
            self.contact_idx_middle_finger = contact_idx[47:66]
            self.contact_idx_ring_finger = contact_idx[66:73]
            self.contact_idx_little_finger = contact_idx[73:98]
            self.contact_idx_thumb_finger = contact_idx[98:]

            self.faces_idx_index_finger = range(381, 513)
            self.faces_idx_middle_finger = range(613, 745)
            self.faces_idx_ring_finger = range(849, 981)
            self.faces_idx_little_finger = range(1081, 1213)
            self.faces_idx_thumb = range(1251, 1383)
        self.to("cuda")        

    def to(self, device):
        self.mano_layer.to(device)
        return super().to(device)

    def forward(self, debug=False):
        num_frames = len(self.hand_rot)
        betas = self.hand_beta[None, :].repeat(num_frames, 1)
        output = self.mano_layer(
            betas=betas,
            global_orient=self.hand_rot,
            hand_pose=self.hand_pose,
            transl=torch.zeros_like(self.hand_transl),
        )
        # canonical space
        j3d_can = output.joints
        v3d_can = output.vertices

        # transform to camera space
        h2c_mat = torch.tile(torch.eye(4), (self.hand_rot.shape[0], 1, 1)).to(self.hand_rot.device)
        h2c_mat[:, :3, 3] = self.hand_transl
        h2c_mat = h2c_mat * self.hand_scale # scale the hand
        h2c_mat[:, 3, 3] = 1
        h2o_mat = torch.inverse(self.o2c) @ h2c_mat

        j3d_cam = transform_points(j3d_can, h2c_mat)
        v3d_cam = transform_points(v3d_can, h2c_mat)
        j3d_obj = transform_points(j3d_can, h2o_mat)
        v3d_obj = transform_points(v3d_can, h2o_mat)
        



        # j3d_ra = j3d - j3d[:, 0:1, :]

        K = self.K[None, :, :].repeat(num_frames, 1, 1).to(v3d_cam.device)

        out = xdict()
        v2ds = project2d_batch(K, v3d_cam)
        j2ds = project2d_batch(K, j3d_cam)
        

        out["j3d_cam"] = j3d_cam
        out["v3d_cam"] = v3d_cam
        out["j3d_obj"] = j3d_obj
        out["v3d_obj"] = v3d_obj
        out["f3d"] = self.f3d
        out["v2d"] = v2ds
        out["j2d"] = j2ds
        out["hand_rot"] = self.hand_rot
        out["hand_pose"] = self.hand_pose
        out["hand_beta"] = self.hand_beta
        out["hand_transl"] = self.hand_transl
        out["hand_scale"] = self.hand_scale
        # out["im_paths"] = self.im_paths
        # out["mask_paths"] = self.mask_paths
        # out["K"] = self.K
        out["contact_idx"] = self.contact_idx
        out["contact_idx_index_finger"] = self.contact_idx_index_finger
        out["contact_idx_middle_finger"] = self.contact_idx_middle_finger
        out["contact_idx_ring_finger"] = self.contact_idx_ring_finger
        out["contact_idx_little_finger"] = self.contact_idx_little_finger
        out["contact_idx_thumb_finger"] = self.contact_idx_thumb_finger

        out["faces_idx_index_finger"] = self.faces_idx_index_finger
        out["faces_idx_middle_finger"] = self.faces_idx_middle_finger
        out["faces_idx_ring_finger"] = self.faces_idx_ring_finger
        out["faces_idx_little_finger"] = self.faces_idx_little_finger
        out["faces_idx_thumb"] = self.faces_idx_thumb
        return out
