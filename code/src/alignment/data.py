from glob import glob
import torch
import numpy as np
import cv2

from common.xdict import xdict
import trimesh


def read_data(seq_name, K, args, colmap_path=None):
    # load data
    im_ps = sorted(glob(f"./data/{seq_name}/processed/images/*.png"))
    mask_ps = sorted(glob(f"./data/{seq_name}/processed/masks/*.png"))
    
    meta = {}
    meta['K'] = K
    meta['im_paths'] = im_ps
    meta['mask_paths'] = mask_ps
    meta['object_cfg_f'] = args.object_cfg_f
    meta['object_ckpt_f'] = args.object_ckpt_f
    meta['object_mesh_f'] = args.object_mesh_f
    o2w_all = torch.FloatTensor(np.load(f"{colmap_path}/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy"))
    meta['o2c'] = o2w_all
    meta['im_H_W'] = cv2.imread(im_ps[0]).shape[:2] # H, W
    
    data_o = {}
    data_o['j2d.gt'] = torch.FloatTensor(
        np.load(f"./data/{seq_name}/processed/colmap_2d/keypoints.npy")
    )
    
    entities  = {}
    entities['object'] = data_o

    def read_hand_data(data):
        mydata = {}
        for k, v in data.items():
            mydata[k] = torch.FloatTensor(v)
        return mydata
    j2d_p = f"./data/{seq_name}/processed/j2d.full.npy"
    data = np.load(
        f"./data/{seq_name}/processed/hold_fit.slerp.npy", allow_pickle=True
    ).item()

    if 'right' in data:
        data_r = read_hand_data(data['right'])
        j2d_data = np.load(j2d_p, allow_pickle=True).item()
        j2d_right = j2d_data['j2d.right']
        right_valid = (~np.isnan(j2d_right.reshape(-1, 21*2).mean(axis=1))).astype(np.float32) # num_frames
        right_valid = np.repeat(right_valid[:, np.newaxis], 21, axis=1)
        j2d_right_pad = torch.FloatTensor(np.concatenate([j2d_right, right_valid[:, :, None]], axis=2))
        data_r['j2d.gt'] = j2d_right_pad
        # mydata['right'] = data_
        entities['right'] = data_r
    
    if 'left' in data:
        data_l = read_hand_data(data['left'])        
        j2d_left = j2d_data['j2d.left']
        left_valid = (~np.isnan(j2d_left.reshape(-1, 21*2).mean(axis=1))).astype(np.float32)
        left_valid = np.repeat(left_valid[:, np.newaxis], 21, axis=1)
        j2d_left_pad = torch.FloatTensor(np.concatenate([j2d_left, left_valid[:, :, None]], axis=2))
        
        data_l['j2d.gt'] = j2d_left_pad
        entities['left'] = data_l
    
    mydata = xdict()
    mydata['entities'] = entities
    mydata['meta'] = meta
    return mydata


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, num_iter):
        self.num_iter = num_iter

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        return idx
