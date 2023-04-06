import torch.nn as nn
import torch

import torch
import torch.nn as nn
from src.alignment.loss_terms import gmof
import pickle as pkl
import numpy as np
from pytorch3d.ops import knn_points
import open3d as o3d

import torch.optim as optim
import pytorch_lightning as pl
import cv2
from PIL import Image
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from utils_simba.nn_tool import get_learnable_parameters

import sys
import common.torch_utils as torch_utils
from common.xdict import xdict
from common.transforms import project2d_batch
from src.alignment.pl_module.ray_hit import RayHit




mse_loss = nn.MSELoss(reduction="none")
l1_loss = nn.L1Loss(reduction="none")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_2d_keypoints(img_path, points):
    # Load the image
    img = mpimg.imread(img_path)
    
    # Define finger segments by their corresponding keypoint indices
    # (These indices may vary depending on the dataset conventions)
    wrist_idx = [0]
    thumb_idx = [13, 14, 15, 16]
    index_idx = [1, 2, 3, 17]
    middle_idx = [4, 5, 6, 18]
    ring_idx = [10, 11, 12, 19]
    little_idx = [7, 8, 9, 20]
    
    plt.figure(figsize=(6.4, 4.8), dpi=100)
    plt.imshow(img)
    
    # Plot each group with a different color and label
    # Wrist
    plt.scatter(points[wrist_idx, 0], points[wrist_idx, 1], c='black', s=20, label='Wrist')
    
    # Thumb
    plt.scatter(points[thumb_idx, 0], points[thumb_idx, 1], c='red', s=20, label='Thumb')
    
    # Index Finger
    plt.scatter(points[index_idx, 0], points[index_idx, 1], c='green', s=20, label='Index')
    
    # Middle Finger
    plt.scatter(points[middle_idx, 0], points[middle_idx, 1], c='blue', s=20, label='Middle')
    
    # Ring Finger
    plt.scatter(points[ring_idx, 0], points[ring_idx, 1], c='magenta', s=20, label='Ring')
    
    # Little Finger
    plt.scatter(points[little_idx, 0], points[little_idx, 1], c='orange', s=20, label='Little')
    
    plt.title("2D Keypoints on Image")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    # If you want the y-axis to match typical image coordinates:
    # top-left corner as (0,0) and y increasing downwards:
    # plt.gca().invert_yaxis()
    
    plt.show()

def loss_fn_h_j2d(preds, targets, conf, valid_frames=None):
    # op2d
    loss = 0.0
    device = preds["right.j2d"].device
    targets_j2d = targets["right.j2d.gt"].to(device)
    if valid_frames is None:
        is_valid = (~torch.isnan(targets_j2d[:, 0, 0]))
    else:
        is_valid = (~torch.isnan(targets_j2d[:, 0, 0])) & valid_frames
    
    loss_2d = (
        gmof(preds["right.j2d"][is_valid] - targets_j2d[is_valid, :, :2], sigma=conf.j2d_sigma).sum(dim=-1)
    )
    
    loss_2d = loss_2d.mean() * conf.j2d
    loss += loss_2d
    
    if "left" in preds:
        # if "left" in preds and "left" in targets:
        targets_j2d = targets["left.j2d.gt"].to(device)

        if valid_frames is None:
            is_valid = (~torch.isnan(targets_j2d[:, 0, 0]))
        else:
            is_valid = (~torch.isnan(targets_j2d[:, 0, 0])) & valid_frames

        loss_2d = gmof(
            preds["left.j2d"][is_valid] - targets_j2d[is_valid, :, :2], sigma=conf.j2d_sigma
        ).sum(dim=-1)

        loss_2d = loss_2d.mean() * conf.j2d
        loss += loss_2d

    return loss

def loss_fn_center(preds, targets, conf):
    hand_3d = preds["right.v3d_cam"]
    obj_3d = preds["object.j3d_cam"]

    # coarse contact
    centroid_h = hand_3d.mean(dim=1)
    centroid_o = obj_3d.mean(dim=1)
    loss = l1_loss(centroid_h, centroid_o).mean() * conf.center

    return loss

def loss_fn_smooth_verts(preds, targets, conf):
    v3d_pred = preds["right.v3d_obj"]
    diff_v3d = v3d_pred[1:] - v3d_pred[:-1]

    loss_smooth = mse_loss(diff_v3d, torch.zeros_like(diff_v3d).detach()).mean() * conf.verts_smooth
    return loss_smooth

def loss_fn_smooth_pose(preds, targets, conf):
    pose_pred = preds["right.hand_pose"]
    diff_pose = pose_pred[1:] - pose_pred[:-1]

    loss_smooth = mse_loss(diff_pose, torch.zeros_like(diff_pose).detach()).mean() * conf.pose_smooth
    return loss_smooth

def loss_fn_reg(preds, targets, conf):
    pose_pred = preds["right.hand_pose"]
    pose_gt = targets["right.hand_pose"]

    loss = l1_loss(pose_pred, pose_gt).mean() * conf.reg
    return loss

# def loss_fn_occluded_contact(preds, targets, conf, ray_hit):
#     obj_occluded_vert_flags = ray_hit.obj_occluded_vert_flags
#     hit_finger_flags = torch.tensor(ray_hit.hit_finger_flags).to(preds['right.v3d_obj'].device)
#     finger_avail_idx = ray_hit.finger_avail_idx
#     losses = []
#     for i in range(len(obj_occluded_vert_flags)):
#         hit_finger_flag = hit_finger_flags[i]
#         if not ray_hit.valid_frames[i] or hit_finger_flag.sum() == 5:
#             continue
#         obj_occluded_vert_flag = obj_occluded_vert_flags[i]
#         obj_verts = preds['object.v3d_obj'][i]
#         obj_occluded_verts = obj_verts[torch.where(obj_occluded_vert_flag > 0)]

#         hand_verts = preds['right.v3d_obj'][i]
#         finger_occluded_finger_idxs = torch.where(hit_finger_flag == 0)[0]
#         finger_occluded_vert_idxs_list = []
#         for finger_occluded_finger_idx in finger_occluded_finger_idxs:
#             finger_occluded_vert_idxs_list.append(finger_avail_idx[finger_occluded_finger_idx])

#         finger_occluded_vert_idxs = np.concatenate(finger_occluded_vert_idxs_list)
#         finger_occluded_verts = hand_verts[finger_occluded_vert_idxs]

#         loss_occluded = knn_points(finger_occluded_verts[None], obj_occluded_verts[None], K=1, return_nn=False)[0].mean() * conf.occ_contact
#         losses.append(loss_occluded)
#     if len(losses) == 0:
#         loss = torch.tensor(0.0).to(preds['right.v3d_obj'].device)
#     else:
#         loss = torch.stack(losses).mean()

#     return loss
def gaussian_weight(incidence, sigma =0.7, mu = torch.pi):
    return torch.exp(-(incidence - mu) ** 2 / (2 * sigma ** 2))

def loss_fn_occluded_contact(preds, targets, conf, ray_hit, debug=False):
    hand_hit_idxs = ray_hit.hand_hit_idxs_occ
    obj_hits = ray_hit.obj_hits_occ
    incidences = ray_hit.incidence_rads_occ
    assert len(hand_hit_idxs) == len(obj_hits)
    losses = []
    loss = torch.tensor(0.0).to(preds['right.v3d_obj'].device)
    try:
        for i in range(len(hand_hit_idxs)):
            if not ray_hit.valid_frames[i]:
                continue
            hand_verts = preds['right.v3d_obj'][i]
            hand_hit = hand_verts[hand_hit_idxs[i]]
            obj_hit = torch.tensor(obj_hits[i]).to(hand_hit.device)
            incidence = torch.tensor(incidences[i]).to(hand_hit.device)
            if hand_hit.shape[0] == 0 or obj_hit.shape[0] == 0:
                continue
            assert hand_hit.shape[0] == obj_hit.shape[0] == incidence.shape[0]
            valid_obj_hit = ~torch.isnan(obj_hit).all(axis=1)
            hand_hit = hand_hit[valid_obj_hit]
            obj_hit = obj_hit[valid_obj_hit]
            incidence = incidence[valid_obj_hit]
            if len(obj_hit) == 0:
                continue
            weights = gaussian_weight(incidence)
            # print(f"i {i}, weights {weights}")
            # loss = l1_loss(hand_hit, obj_hit).mean() * weights * conf.vis_contact

            loss = (F.smooth_l1_loss(hand_hit, obj_hit, reduction="none") * weights[:,None]).mean() * conf.vis_contact
               
            # loss = gmof(hand_hit - obj_hit, sigma=1).sum(dim=-1).mean() * conf.vis_contact
            
            losses.append(loss)
        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0).to(preds['right.v3d_obj'].device)
    except:
        breakpoint()
        pass

    return loss

def loss_fn_knn_contact(preds, targets, conf):
    v3d_h = preds["right.v3d_obj"]
    v3d_o = preds["object.v3d_obj"]
    with open("./code/body_models/contact_zones.pkl", "rb") as f:
        contact_zones = pkl.load(f)
    contact_zones = contact_zones["contact_zones"]
    contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])

    v3d_tips = v3d_h[:, contact_idx]

    # contact
    loss_fine_ho = knn_points(v3d_tips, v3d_o, K=1, return_nn=False)[0].mean()
    loss = conf.vis_contact * loss_fine_ho
    return loss

def loss_fn_vis_contact(preds, targets, conf, ray_hit, debug=False):
    hand_hit_idxs = ray_hit.hand_hit_idxs
    obj_hits = ray_hit.obj_hits
    incidences = ray_hit.incidence_rads
    assert len(hand_hit_idxs) == len(obj_hits)
    losses = []
    loss = torch.tensor(0.0).to(preds['right.v3d_obj'].device)
    try:
        for i in range(len(hand_hit_idxs)):
            if not ray_hit.valid_frames[i]:
                continue
            hand_verts = preds['right.v3d_obj'][i]
            hand_hit = hand_verts[hand_hit_idxs[i]]
            obj_hit = torch.tensor(obj_hits[i]).to(hand_hit.device)
            incidence = torch.tensor(incidences[i]).to(hand_hit.device)
            if hand_hit.shape[0] == 0 or obj_hit.shape[0] == 0:
                continue
            assert hand_hit.shape[0] == obj_hit.shape[0] == incidence.shape[0]
            valid_obj_hit = ~torch.isnan(obj_hit).all(axis=1)
            hand_hit = hand_hit[valid_obj_hit]
            obj_hit = obj_hit[valid_obj_hit]
            incidence = incidence[valid_obj_hit]
            if len(obj_hit) == 0:
                continue

            weights = gaussian_weight(incidence)
            # print(f"i {i}, weights {weights}")
            # loss = l1_loss(hand_hit, obj_hit).mean() * weights * conf.vis_contact

            loss = (F.smooth_l1_loss(hand_hit, obj_hit, reduction="none") * weights[:,None]).mean() * conf.vis_contact
               
            # loss = gmof(hand_hit - obj_hit, sigma=1).sum(dim=-1).mean() * conf.vis_contact
            
            losses.append(loss)
        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0).to(preds['right.v3d_obj'].device)
    except:
        breakpoint()
        pass

    return loss

def loss_fn_penetrate(preds, targets, conf, valid_frames=None):
    if valid_frames is None:
        hand_3ds = preds["right.v3d_obj"]
    else:
        hand_3ds = preds["right.v3d_obj"][valid_frames]
    sdf_obj = preds["object.sdf"]
    sdfs = sdf_obj(hand_3ds.reshape(-1, 3)).reshape(hand_3ds.shape[0], -1, 1)
    loss = torch.clamp(-sdfs, min=0).mean() * conf.penetrate
    return loss



class PLModule(pl.LightningModule):
    def __init__(self, data, args, conf, device='cuda'):
        super().__init__()
        self.args = args
        self.conf = conf
        from src.alignment.params.hand import MANOParameters
        from src.alignment.params.object import ObjectParameters
        models = nn.ModuleDict()
        entities = data['entities']
        if self.args.mode == "o" or self.args.mode == "ho":
            ray_hit = RayHit(save_dir=f"{args.out_dir}/mano_fit_ckpt/r/")
            ray_hit.load_data()
            self.ray_hit = ray_hit
        self.register_buffer("best_frame", torch.tensor(-1))

        for key in entities.keys():
            if key == "object":
                models[key] = ObjectParameters(entities[key], data['meta'])          
            else:
                models[key] = MANOParameters(entities[key], data['meta'], is_right=key == "right")
        self.entities = data['entities']
        self.meta = data['meta']
        self.models = models
        
        for node_id in entities.keys():
            self.models[node_id].to(device)
        
    def on_save_checkpoint(self, checkpoint):
        return
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            for key in self.models.keys():
                if key == 'right':
                    state_dict[f'models.{key}.hand_rot'] = state_dict[f'models.{key}.hand_rot'][self.best_frame].unsqueeze(0).expand(self.models[key].hand_rot.shape[0], -1)
                    state_dict[f'models.{key}.hand_pose'] = state_dict[f'models.{key}.hand_pose'][self.best_frame].unsqueeze(0).expand(self.models[key].hand_pose.shape[0], -1)
                    h2c_transl = state_dict[f'models.{key}.hand_transl']
                    hand_scale = state_dict[f'models.{key}.hand_scale']
                    o2c_mat =  self.models['right'].o2c
                    c2o_mat = torch.inverse(o2c_mat)
                    h2c_mat = torch.tile(torch.eye(4).to("cuda"), (o2c_mat.shape[0], 1, 1))
                    h2c_mat[:, :3, 3] = h2c_transl
                    h2c_mat = h2c_mat * hand_scale # scale the hand
                    h2c_mat[:, 3, 3] = 1
                    h2o_mat = c2o_mat @ h2c_mat
                    best_h2o_mat = h2o_mat[self.best_frame]

                    h2o_mat_expand = torch.tile(best_h2o_mat, (h2o_mat.shape[0], 1, 1))
                    h2c_mat_expand = (o2c_mat @ h2o_mat_expand) / hand_scale
                    h2c_mat_expand[:, 3, 3] = 1
                    state_dict[f'models.{key}.hand_transl'] = h2c_mat_expand[:, :3, 3]
                    # state_dict[f'models.{key}.hand_transl'] = state_dict[f'models.{key}.hand_transl'][self.best_frame].unsqueeze(0).expand(self.models[key].hand_transl.shape[0], -1)
                    
    def training_step(self, batch, batch_idx):
        device = self.device
        self.condition_training()
        # if self.global_step % 100 == 0:
        #     print(f"scale: {self.models['right'].hand_scale}, transl: {self.models['right'].hand_transl[0]}")
        preds = xdict()
        for key in self.entities.keys():
            preds.merge(self.models[key]().prefix(key + '.'))       
            
        if self.global_step == 0:
            targets = preds.detach().to(device)
            for key in self.entities.keys():
                targets[f"{key}.j2d.gt"] = self.entities[key]['j2d.gt']
            device = self.device
            self.targets = targets
            if self.args.mode == "o" or self.args.mode == "ho":
                self.best_frame = torch.tensor(self.ray_hit.best_frame)              
        
        loss = 0.0
        if self.args.mode == "h":
            loss_j2d = loss_fn_h_j2d(preds, self.targets, self.conf)
            loss += loss_j2d
            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("j2d", loss_j2d, on_step=True, on_epoch=False, prog_bar=True)

            if 0:
                if self.global_step % 500 == 0:
                    with torch.no_grad():
                        i = 4
                        # for i in range(len(j2ds)):
                        if 1:
                            print(f"Processing frame {i}")
                            plot_2d_keypoints(self.meta['im_paths'][i], preds["right.j2d"][i].clone().detach().cpu().numpy())            
                            # plot_2d_keypoints(self.meta['im_paths'][i], self.targets["right.j2d.gt"][i].clone().detach().cpu().numpy())            
        elif self.args.mode == "o":            
            # loss_center = loss_fn_center(preds, self.targets, self.conf)
            # loss_smooth = loss_fn_smooth(preds, self.targets, self.conf)
            if self.conf.contact_type == "vis":
                loss_j2d = loss_fn_h_j2d(preds, self.targets, self.conf, torch.tensor(self.ray_hit.valid_frames).to(device)) * 20.0
                loss_contact = loss_fn_vis_contact(preds, self.targets, self.conf, self.ray_hit)  * 1.0
            elif self.conf.contact_type == "knn":
                loss_j2d = loss_fn_h_j2d(preds, self.targets, self.conf, torch.tensor(self.ray_hit.valid_frames).to(device)) * 100.0
                loss_contact = loss_fn_knn_contact(preds, self.targets, self.conf)
            else:
                raise NotImplementedError
            loss += loss_contact + loss_j2d
            # loss += loss_j2d + loss_center + loss_smooth + loss_contact
            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)                        
            self.log("j2d", loss_j2d, on_step=True, on_epoch=False, prog_bar=True)
            # self.log("center", loss_center, on_step=True, on_epoch=False, prog_bar=True)
            # self.log("smooth", loss_smooth, on_step=True, on_epoch=False, prog_bar=True)
            self.log("cnt", loss_contact, on_step=True, on_epoch=False, prog_bar=True)
            # self.log("cnt_occ", loss_contact_occluded, on_step=True, on_epoch=False, prog_bar=True)
            
        elif self.args.mode == "ho":
            if self.conf.contact_type == "vis":
                loss_j2d = loss_fn_h_j2d(preds, self.targets, self.conf, torch.tensor(self.ray_hit.valid_frames).to(device)) * 20.0
                loss_contact = loss_fn_vis_contact(preds, self.targets, self.conf, self.ray_hit)
            elif self.conf.contact_type == "knn":
                loss_j2d = loss_fn_h_j2d(preds, self.targets, self.conf, torch.tensor(self.ray_hit.valid_frames).to(device)) * 100.0
                loss_contact = loss_fn_knn_contact(preds, self.targets, self.conf)
            else:
                raise NotImplementedError
            # loss_center = loss_fn_center(preds, self.targets, self.conf)
            loss_smooth_pose = loss_fn_smooth_pose(preds, self.targets, self.conf)
            loss_smooth_verts = loss_fn_smooth_verts(preds, self.targets, self.conf)
            
            # loss_contact_occluded = loss_fn_occluded_contact(preds, self.targets, self.conf, self.ray_hit)
            loss_penetrate = loss_fn_penetrate(preds, self.targets, self.conf, torch.tensor(self.ray_hit.valid_frames).to(device))
            # loss_reg =  loss_fn_reg(preds, self.targets, self.conf)
            loss += loss_contact + loss_j2d + loss_smooth_pose + loss_smooth_verts + loss_penetrate
            # loss += loss_j2d + loss_center + loss_smooth + loss_contact + loss_smooth_verts
            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)                        
            self.log("j2d", loss_j2d, on_step=True, on_epoch=False, prog_bar=True)
            # self.log("center", loss_center, on_step=True, on_epoch=False, prog_bar=True)
            self.log("smt_pose", loss_smooth_pose, on_step=True, on_epoch=False, prog_bar=True)
            self.log("smt_verts", loss_smooth_verts, on_step=True, on_epoch=False, prog_bar=True)
            self.log("cnt", loss_contact, on_step=True, on_epoch=False, prog_bar=True)
            # self.log("cnt_occ", loss_contact_occluded, on_step=True, on_epoch=False, prog_bar=True)
            # self.log("reg", loss_reg, on_step=True, on_epoch=False, prog_bar=True)
            self.log("pen", loss_penetrate, on_step=True, on_epoch=False, prog_bar=True)
        else:
            raise NotImplementedError

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.conf.lr)
        self.optimizer = optimizer
        return optimizer

    def condition_training(self):
        step = self.global_step
        if step == 0:
            for val in self.models.values():
                torch_utils.toggle_parameters(val, requires_grad=False)

        # freeze the other model
        if self.args.mode == "h":
            # hand model schedule
            if step == 0:
                print("Hand: stage 0")
                for key, val in self.models.items():
                    if key in ['right', 'left']:
                        val.hand_transl.requires_grad = True
                        # val.hand_rot.requires_grad = True
                        # val.hand_scale.requires_grad = True
                get_learnable_parameters(self)

            if step == 3000:
                print("Hand: stage 1")
                # for key, val in self.models.items():
                #     if key in ['right', 'left']:
                #         val.hand_beta.requires_grad = True
                get_learnable_parameters(self)
            if step % self.conf.decay_every == 0:
                torch_utils.decay_lr(self.optimizer, self.conf.decay_factor)

        elif self.args.mode == "o":
            if step == 0:
                print("Object: stage 0")
                self.models['right'].hand_scale.requires_grad = True
                self.models['right'].hand_transl.requires_grad = True
                get_learnable_parameters(self)


            if step % self.conf.decay_every == 0:
                torch_utils.decay_lr(self.optimizer, self.conf.decay_factor)
        
        elif self.args.mode == "ho":
            # for key, val in self.models.items():
            #     if key in ['right', 'left']:
            #         val.hand_transl.requires_grad = True
            #     else:
            #         val.obj_scale.requires_grad = True
            #         val.obj_transl.requires_grad = True
            if step == 0:
                self.models['right'].hand_scale.requires_grad = True
                self.models['right'].hand_transl.requires_grad = True
                # self.models['right'].hand_pose.requires_grad = True
                get_learnable_parameters(self)
            if step % self.conf.decay_every == 0:
                print("Decay")
                torch_utils.decay_lr(self.optimizer, self.conf.decay_factor)
                get_learnable_parameters(self)
        else:
            raise NotImplementedError


class HOModule(PLModule):
    def __init__(self, data, args, conf):
        super().__init__(data, args, conf)
    
