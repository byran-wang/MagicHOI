import bisect
import math
import os
from dataclasses import dataclass, field
import json
import open3d as o3d
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import smplx

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

from utils_simba.data_read import readCamerasFromBlenderJson
from utils_simba.data_read import ReadHO3DGTPose, ReadHO3DFoundationPose
from utils_simba.pose import cartesian_to_spherical, get_relative_trans, scale_t, reposition_camera
from utils_simba.hand import initialize_mano_model
import re
import copy
from threestudio.data.co3d import get_bbox_from_mask, get_clamp_bbox, crop_around_box, resize_image
# from visibility_test import render_mesh_and_get_visibility, get_camera_intrinsics
# from src.datasets.tempo_dataset import TempoDataset, HOLDConfig


@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of container
    image_path: str = ""
    root_dir: str = ""        
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    cond_pose_from_selected_cam: bool = True
    user_elevation_deg: float = -1000.0
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    box_crop: bool = True
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    box_crop_ds: int = 2    
    requires_depth: bool = False
    requires_normal: bool = False
    train_num_rays: int = 4096

    rays_d_normalize: bool = True
    viz: Any = field(default_factory=lambda: {"show_cameras": False})
    data_type: str = "real_image_gt_pose"
    blender: dict = field(default_factory=dict)
    real_image_gt_pose: dict = field(default_factory=dict)
    real_image_foundation_pose: dict = field(default_factory=dict)
    align: dict = field(default_factory=dict)
    HOLD: dict = field(default_factory=dict)
    colmap_dir: str = ""
    train_mode: str = "3d_ref"


class SingleImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg


        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )
        # tempo_dataset_cfg = parse_structured(
        #         HOLDConfig, self.cfg.get("HOLD", {})
        # )
        # self.hold_dataset = TempoDataset(tempo_dataset_cfg)
        
        cond_info = self.get_cond_info()
        # self.hand_info = self.get_hand_info()            
        if self.cfg.align.do_3d_guidance_only and not self.cfg.align.do_ref_only:
            self.c2w = cond_info["c2w4x4"]
            self.light_position = cond_info["camera_position"]
            self.fovy = cond_info["fovy"]
            self.image_name = cond_info["image_name"]
            self.cxs = cond_info["cx"]
            self.cys = cond_info["cy"]
            self.cfg.height = cond_info["height"]
            self.cfg.width = cond_info["width"]
            self.focal_lengths = cond_info["focal_length"]
            ref_info = None
        else:
            ref_info = self.get_ref_info()
            self.c2w = ref_info["c2w4x4"]
            # self.camera_position = ref_info["camera_position"]
            self.light_position = ref_info["light_position"]
            self.fovy = ref_info["fovy"]
            self.image_name = ref_info["image_name"] # [self.cfg.image_path]
            # self.inpaint_shperes = ref_info["inpaint_shperes"]
            self.cxs = ref_info["cx"]
            self.cys = ref_info["cy"]
            self.cfg.height = ref_info["height"]
            self.cfg.width = ref_info["width"]  
            self.focal_lengths = ref_info["focal_length"]


        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.cx = self.cxs[0]
        self.cy = self.cys[0]
        self.focal_length = self.focal_lengths[0]
                
        self.load_images()
        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=[1.0, 1.0], principal=[cx, cy], use_pixel_centers=False) # disable use_pixel_centers for Rerunveiwer
            for (height, width, cx, cy) in zip(self.heights, self.widths, self.cxs, self.cys)
        ]
        self.directions_unit_focal = self.directions_unit_focals[0]
        if self.cfg.box_crop and (not (self.cfg.align.do_3d_guidance_only and (not self.cfg.align.do_ref_only))):
            print("cropping...")
            crop_masks = []
            crop_masks_obj_hand = []
            crop_imgs = []
            crop_directions = []
            crop_xywhs = []
            crop_instinsics = []
            max_sl = 0
            num_frames = len(self.rgb)
            crop_width = int(self.width / self.cfg.box_crop_ds)
            crop_height = int(self.height / self.cfg.box_crop_ds)
            
            for i in range(num_frames):
                bbox_xywh = np.array(
                    get_bbox_from_mask(np.array(self.mask[i].cpu()).squeeze(-1), self.cfg.box_crop_mask_thr)
                )
                clamp_bbox_xywh = get_clamp_bbox(bbox_xywh, self.cfg.box_crop_context)
                max_sl = max(clamp_bbox_xywh[2] - clamp_bbox_xywh[0], max_sl)
                max_sl = max(clamp_bbox_xywh[3] - clamp_bbox_xywh[1], max_sl)
                
                mask = crop_around_box(self.mask[i], clamp_bbox_xywh)
                img = crop_around_box(self.rgb[i], clamp_bbox_xywh)
                mask_obj_hand = crop_around_box(self.mask_obj_hand[i], clamp_bbox_xywh)
                # print(f"crop img {i} xywh with {clamp_bbox_xywh}")
                # from utils_simba.img import show_img
                # show_img(mask)
                # show_img(img)

                # resize to the same shape
                mask, _, _ = resize_image(np.array(mask.cpu()).astype(np.float32), crop_height, crop_width)
                mask_obj_hand, _, _ = resize_image(np.array(mask_obj_hand.cpu()).astype(np.float32), crop_height, crop_width)
                img, scale, _ = resize_image(np.array(img.cpu()).astype(np.float32), crop_height, crop_width)

                fx_crop, fy_crop = np.array(self.focal_length[0]) * scale, np.array(self.focal_length[0]) * scale
                cx_crop, cy_crop = (self.cx - clamp_bbox_xywh[0]) * scale, (self.cy - clamp_bbox_xywh[1]) * scale
                crop_directions.append(
                    get_ray_directions(
                        crop_height,
                        crop_width,
                        (fx_crop, fy_crop),
                        (cx_crop, cy_crop),)
                )
                crop_instinsic = np.array([
                        [fx_crop, 0.0, cx_crop, 0.0],
                        [0.0, fy_crop, cy_crop, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                crop_masks.append(mask)
                crop_imgs.append(img)
                crop_masks_obj_hand.append(mask_obj_hand)
                crop_xywhs.append(clamp_bbox_xywh)
                crop_instinsics.append(crop_instinsic)
            
            self.rgb = torch.tensor(np.array(crop_imgs)).contiguous().to(self.rank)
            self.mask = torch.tensor(np.array(crop_masks)).contiguous().to(self.rank)
            self.mask_obj_hand = torch.tensor(np.array(crop_masks_obj_hand)).contiguous().to(self.rank)
            crop_directions = [t.cpu().numpy() for t in crop_directions]
            self.crop_directions = torch.tensor(np.array(crop_directions)).contiguous()
            self.rays_o, self.rays_d = get_rays(
                self.crop_directions, self.c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale, normalize=self.cfg.rays_d_normalize,
            )

        else:
            self.set_rays()
            
        self.prev_height = self.height

     

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions,
            self.c2w,
            keepdim=True,
            noise_scale=self.cfg.rays_noise_scale,
            normalize=self.cfg.rays_d_normalize,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
    
    def get_hand_info(self):
        fit_f = f"{self.cfg.HOLD.preprocess_dir}/hold_fit.aligned.npy"
        data = np.load(fit_f, allow_pickle=True).item()
        # Initialize MANO model
        o2c_mat = torch.tensor(np.load(f"{self.cfg['colmap_dir']}/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy"))
        c2o_mat = torch.inverse(o2c_mat).float() # camera to object
        MANO_R_f = f"{self.cfg.HOLD.MANO_dir}/MANO_RIGHT.pkl"
        f3d_r = initialize_mano_model(MANO_R_f)
        betas = data['right']["hand_beta"]
        betas = torch.tensor(np.tile(betas, (o2c_mat.shape[0], 1)))
        h2c_rot = torch.tensor(data['right']["hand_rot"])
        h2c_transl = torch.tensor(data['right']["hand_transl"])
        global_orient = torch.tensor(data['right']["hand_rot"])
        hand_pose = torch.tensor(data['right']["hand_pose"])
        obj_scale = torch.tensor(data['object']['obj_scale'])

        # Extract and transform right hand vertices
        mano_layer = smplx.create(
                model_path=MANO_R_f, model_type="mano", use_pca=False, is_rhand=True
        )
        h2c_mat = torch.tile(torch.eye(4), (o2c_mat.shape[0], 1, 1))
        h2c_mat[:, :3, 3] = h2c_transl
        h2c_mat = h2c_mat / obj_scale # scale the hand
        h2c_mat[:, 3, 3] = 1
        h2o_mat = c2o_mat @ h2c_mat
        hand_scale = torch.tile(1 / obj_scale, (o2c_mat.shape[0], 1))

        
        full_pose = torch.cat((global_orient, hand_pose), dim=1)
        return {
            "full_pose": full_pose, 
            "betas": betas,
            "h2o": h2o_mat,
            "transl": h2c_transl,
            "scale": hand_scale,
            "mano_layer": mano_layer,
            "mano_faces": f3d_r,
        }
    
    def get_cond_info(self):
        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w, torch.zeros_like(c2w[:, :1])], dim=1
        )
        c2w4x4[:, 3, 3] = 1.0

        camera_position = camera_position
        elevation_deg, azimuth_deg = elevation_deg, azimuth_deg
        camera_distance = camera_distance
        fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))
        condition_image = cv2.imread(self.cfg.image_path, cv2.IMREAD_UNCHANGED)
        height, width = condition_image.shape[:2]
        image_name = [self.cfg.image_path]
        image_index = []
        for image_file in image_name:
            idx_match = re.search(r"\d+", os.path.basename(image_file))
            assert idx_match is not None
            frame_idx = int(idx_match.group(0))
            image_index.append(frame_idx)
        if self.split == "val":
            heights = [height]
            widths = [width]
            cys = [int((height-1)*0.5)]
            cxs = [int((width-1)*0.5)]
        else:
            heights = [height//4, height//2, height]
            widths = [width//4, width//2, width]
            cys = [int((height//4-1)*0.5), int((height//2-1)*0.5), int((height-1)*0.5)]
            cxs = [int((width//4-1)*0.5), int((width//2-1)*0.5), int((width-1)*0.5)]

        focal_lengths = [0.5 * height / torch.tan(0.5 * fovy) for height in heights]

        return {"c2w4x4": c2w4x4, 
                "camera_position": camera_position, 
                "light_position": camera_position, 
                "elevation_deg": elevation_deg, 
                "azimuth_deg": azimuth_deg, 
                "camera_distance": camera_distance, 
                "fovy": fovy,
                "image_name": image_name,
                "image_index": image_index,
                "cx": cxs,
                "cy": cys,
                "width": widths,
                "height": heights,
                "focal_length": focal_lengths
                }
    
    def get_ref_info(self):
        if self.cfg.data_type == "blender":
            cam_infos = readCamerasFromBlenderJson(self.cfg.blender)
            ref_views = self.cfg.blender.get("ref_views", [])
        elif self.cfg.data_type == "real_image_gt_pose":
            cam_infos = ReadHO3DGTPose(self.cfg.real_image_gt_pose)
            ref_views = self.cfg.real_image_gt_pose.get("ref_views", [])
        elif self.cfg.data_type == "real_image_foundation_pose":
            cam_infos = ReadHO3DFoundationPose(self.cfg.real_image_foundation_pose)
            ref_views = self.cfg.real_image_foundation_pose.get("ref_views", [])
        else:
            raise ValueError(f"Unknown data type {self.cfg.data_type}")
        assert ref_views, "ref_views should not be empty"
        inpaint_views = [int(i) for i in re.findall(r'\d+', os.path.basename(self.cfg.image_path))]
        c2w4x4 = []
        camera_position = []
        image_name = []
        fovy = []
        inpaint_shperes = []
        align_pose_type = self.cfg.real_image_foundation_pose.align_pose_type
        for info in cam_infos:
            if int(info.uid) in ref_views:
                c2w = info.c2w4x4
                c2w4x4.append(c2w)
                position = c2w[:3, 3]
                camera_position.append(position)
                image_name.append(info.rgba_name)
                fovy.append(info.fov_y)
                if info.uid in inpaint_views:
                    inpaint_shpere= cartesian_to_spherical(position[None])
                    inpaint_shpere["uid"] = info.uid
                    inpaint_shperes.append(inpaint_shpere)
                    if align_pose_type == "PnP":
                        refc2refw4x4 = torch.tensor(np.array(c2w))
                        ref_view_name = f"{info.uid:04d}.json"
                        ref_view_pose_f = os.path.join(self.cfg.real_image_foundation_pose.camera_PnP_align_path, ref_view_name)
                        try:
                            camera = json.load(open(ref_view_pose_f, 'r'))
                        except:
                            print(f"Failed to load camera file {ref_view_pose_f}")
                            assert False
                        refc2condw4x4_aline = torch.tensor(np.array(camera['pred_glc2blw']))
                        # calculate the scale
                        ref_dist = inpaint_shperes[0]["distance"]
                        inpaint_dist = self.cfg.default_camera_distance
                        dist_scale = inpaint_dist / ref_dist
                        scale = dist_scale * cam_infos[0].scale_inpaint
        
        if align_pose_type == "teaser":
            ref_view_pose_f = self.cfg.real_image_foundation_pose.selected_pose
            try:
                camera = json.load(open(ref_view_pose_f, 'r'))
            except:
                print(f"Failed to load camera file {ref_view_pose_f}")
                assert False
            scale = np.array(camera['que_blw2con_blw_scale'])
            refw2condw4x4_aline = torch.tensor(np.array(camera['que_blw2con_blw_T']))
            # breakpoint()
            # print()
        elif align_pose_type == "None":
            scale = 1.0
            cN2w_reposition = np.array(c2w4x4) * scale
        else:
            print(f"invalid align_pose_type {align_pose_type}")
            assert False
        
        if align_pose_type == "PnP":
            c2w4x4_relative = get_relative_trans(c2w4x4, refc2refw4x4)
            c2w4x4_scale = scale_t(c2w4x4_relative, scale)
            cN2w_reposition = reposition_camera(c2w4x4_scale, refc2condw4x4_aline)
        elif align_pose_type == "teaser":
            c2w4x4_relative = get_relative_trans(c2w4x4, np.eye(4))
            c2w4x4_scale = scale_t(c2w4x4_relative, scale)
            cN2w_reposition = refw2condw4x4_aline @ c2w4x4_scale

        ref_image = cv2.imread(image_name[0], cv2.IMREAD_UNCHANGED)
        height, width = ref_image.shape[:2]
        # TODO: first should center the object
        image_index = []
        for image_file in image_name:
            idx_match = re.search(r"\d+", os.path.basename(image_file))
            assert idx_match is not None
            frame_idx = int(idx_match.group(0))
            image_index.append(frame_idx)
        ds = self.cfg.real_image_foundation_pose.image_down_sample
        heights = [int(height/ds)]
        widths = [int(width/ds)]
        cxs = [cam_infos[0].cx / ds]
        cys = [cam_infos[0].cy / ds]
        fovy = torch.tensor([fovy[0]]).float()
        focal_lengths = [torch.tensor(cam_infos[0].fl_y / ds)[None]]
        final_c2w4x4 = cN2w_reposition
        # final_c2w4x4 = torch.tensor(c2w4x4) # debug origin pose

        return {"c2w4x4": torch.tensor(final_c2w4x4).float(),
                "camera_position": torch.tensor(final_c2w4x4[:,:3,3]).float(),
                "light_position": torch.tensor(final_c2w4x4[:,:3,3]).float(),
                "fovy": fovy,
                "image_name": image_name,
                "image_index": image_index,
                "inpaint_shperes": inpaint_shperes,
                "cx": cxs,
                "cy": cys,
                "height": heights,
                "width": widths,
                "focal_length": focal_lengths
                }

    def load_images(self):
        # load image
        self.rgb = []
        self.mask = []
        self.mask_obj_hand = [] # mask for object and hand
        for i, image_name in enumerate(self.image_name):
            image_name = str(image_name)
            assert os.path.exists(
                image_name
            ), f"Could not find image {image_name}!"
            rgba = cv2.cvtColor(
                cv2.imread(image_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
            )
            if (self.height, self.width) != rgba.shape[:2]:
                rgba = (
                    cv2.resize(
                        rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
                    ).astype(np.float32)
                    / 255.0
                )
            else:
                rgba = rgba.astype(np.float32) / 255.0
            rgb = rgba[..., :3]         

            mask = (rgba[..., 3:] > 0.5).astype(np.float32)
            if (self.height, self.width) != mask.shape[:2]:
                mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_AREA)
            rgb = (mask * rgb + 1.0 * (1 - mask)).astype(np.float32)
                # mask read from mask file
            mask_obj_hand_f = image_name.replace("rgbas", "masks")
            assert os.path.exists(
                mask_obj_hand_f
            ), f"Could not find mask {mask_obj_hand_f}!"
            mask_obj_hand = cv2.imread(mask_obj_hand_f, cv2.IMREAD_GRAYSCALE)
            if (self.height, self.width) != mask_obj_hand.shape[:2]:
                mask_obj_hand = cv2.resize(mask_obj_hand, (self.width, self.height), interpolation=cv2.INTER_AREA)
            mask_obj_hand = (mask_obj_hand > 0).astype(np.float32)[..., None]
            # Use alpha channel as mask
            
            self.rgb.append(rgb)
            self.mask.append(mask)
            self.mask_obj_hand.append(mask_obj_hand)
        self.rgb = torch.tensor(np.array(self.rgb)).contiguous().to(self.rank)
        self.mask = torch.tensor(np.array(self.mask)).contiguous().to(self.rank)
        self.mask_obj_hand = torch.tensor(np.array(self.mask_obj_hand)).contiguous().to(self.rank)

        # load depth
        self.depth = None

        # load normal
        self.normal = None 
    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_images()


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.get_all_images()))
        



    def collate(self, batch) -> Dict[str, Any]:
        idx = self.image_perm[self.idx]
        rays_o = self.rays_o[idx][None]
        rays_d = self.rays_d[idx][None]
        rgb = self.rgb[idx][None]
        mask = self.mask[idx][None]
        mask_obj_hand = self.mask_obj_hand[idx][None]

        # hand_scale = self.hand_info["scale"][idx][None]
        # hand_transl = self.hand_info["transl"][idx][None]
        # hand_full_pose = self.hand_info["full_pose"][idx][None]
        # hand_betas = self.hand_info["betas"][idx][None]
        # hand_h2o = self.hand_info["h2o"][idx][None]

        if (
            self.cfg.train_num_rays != -1
            and self.cfg.train_num_rays < self.height * self.width
        ):
            _, height, width, _ = rays_o.shape
            x = torch.randint(
                0, width, size=(self.cfg.train_num_rays,), device=rays_o.device
            )
            y = torch.randint(
                0, height, size=(self.cfg.train_num_rays,), device=rays_o.device
            )

            rays_o = rays_o[:, y, x].unsqueeze(-2)
            rays_d = rays_d[:, y, x].unsqueeze(-2)
            rgb = rgb[:, y, x].unsqueeze(-2)
            mask = mask[:, y, x].unsqueeze(-2)
            mask_obj_hand = mask_obj_hand[:, y, x].unsqueeze(-2)

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            # "mvp_mtx": self.mvp_mtx,
            # "camera_positions": self.camera_position,
            "light_positions": self.light_position[idx][None],
            # "elevation": self.elevation_deg,
            # "azimuth": self.azimuth_deg,
            # "camera_distances": self.camera_distance,
            "rgb": rgb,
            # "ref_depth": self.depth,
            # "ref_normal": self.normal,
            "mask": mask,
            "height": self.height,
            "width": self.width,
            "focal_length": self.focal_length,               
            "c2w": self.c2w,
            # "hand_info": self.hand_info,
            # "fovy": self.fovy,
            # "hand_scale": hand_scale,
            # "hand_transl": hand_transl,
            # "hand_full_pose": hand_full_pose,
            # "hand_betas": hand_betas,
            # "hand_h2o": hand_h2o,
            "image_index": idx,
            "mask_obj_hand": mask_obj_hand,
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)
        
        # batch["hold_data"] = self.hold_dataset.__getitem__(None)

        self.idx += 1
        if self.idx == len(self.get_all_images()):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.get_all_images()))

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SingleImageTestDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]

class SingleImageValDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.get_all_images())

    def __getitem__(self, index):

        idx = index
        batch = {
            "index" : idx,
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
            # "mvp_mtx": self.mvp_mtx[idx],
            "c2w": self.c2w[idx],
            "light_positions": self.light_position[idx],
            # "camera_positions": self.camera_position[idx],
            # "elevation": self.elevation_deg,
            # "azimuth": self.azimuth_deg,
            # "camera_distances": self.camera_distance,
            # "height": self.height,
            # "width": self.width,            
            # "fovy": self.fovy[idx],
            "focal_length": self.focal_length,
            "pp": torch.tensor([self.cx, self.cy]),

            # "hand_scale": self.hand_info["scale"][idx],
            # "hand_transl": self.hand_info["transl"][idx],
            # "hand_full_pose": self.hand_info["full_pose"][idx],
            # "hand_betas": self.hand_info["betas"][idx],
            # "hand_h2o": self.hand_info["h2o"][idx],
            "image_index": idx,          
        }
        return batch


@register("single-image-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageValDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageTestDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)