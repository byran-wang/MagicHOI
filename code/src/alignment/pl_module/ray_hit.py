import trimesh
import math
from PIL import Image
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from common.xdict import xdict
import pickle
from tqdm import tqdm 

from common.body_models import seal_mano_mesh
from common.transforms import project2d_batch
from src.fitting.utils import construct_targets
from src.fitting.utils import (
    create_meshes,
    create_silhouette_renderer,
)

import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.geometry import transform_points, get_incident_angle
from utils_simba.render import make_mesh_tensors, nvdiffrast_render, get_ray_origin_direction

def show_ray_hit(ray_origin, locations, mesh, index_tri=None, interval=1000):
    # Compute distances from the ray origin to all hit locations
    # distances = np.linalg.norm(locations - ray_origin, axis=1)
    
    # # Sort the hits by distance (optional)
    # sorted_indices = np.argsort(distances)
    # sorted_locations = locations[sorted_indices]
    # sorted_tri_ids = index_tri[sorted_indices]
    
    import pyvista as pv
    # Convert Trimesh mesh to PyVista mesh
    pv_mesh = pv.wrap(mesh)

    # Initialize the Plotter
    plotter = pv.Plotter()
    
    # Add the merged mesh
    plotter.add_mesh(pv_mesh, color='lightblue', show_edges=True, opacity=0.5)

    # Add the camera position as a red sphere
    plotter.add_mesh(pv.Sphere(radius=0.02, center=ray_origin), color='red', name='Camera')     

    # Iterate over all hit points
    for i, loc in enumerate(locations):
        if i % interval != 0:
            continue
        # Add a green line (ray) from the camera to the hit location
   
        ray_line = np.array([ray_origin, loc])
        plotter.add_lines(ray_line, color='gray', width=2, name=f'Ray_{i+1}')
        
        # Add a yellow sphere at the hit location
        plotter.add_mesh(pv.Sphere(radius=0.005, center=loc), color='yellow', name=f'Hit Location {i+1}')
        
        # Highlight the hit triangle by extracting it and adding as a separate mesh
        # if index_tri != None:
        if 1:
            hit_triangle = mesh.faces[index_tri[i]]
            hit_vertices = mesh.vertices[hit_triangle]
            triangle_mesh = trimesh.Trimesh(vertices=hit_vertices, faces=[[0, 1, 2]], process=False)
            pv_triangle = pv.wrap(triangle_mesh)
            plotter.add_mesh(pv_triangle, color='orange', opacity=1.0, show_edges=True, name=f'Triangle_{i+1}')
        
            # Optionally, add the triangle vertices as distinct points
            plotter.add_points(hit_vertices, color='purple', point_size=10, name=f'Triangle Vertices_{i+1}')
                            # Add axes for reference
    plotter.add_axes()
    # Display the plot
    plotter.show()                    

class RayHit():
    def save_masks(self, binary_mask, tg_mask_obj, tg_mask_hand, valid_mask, iou, save_path, show=False):
        plt.figure(figsize=(10, 7))
        binary_mask_np = binary_mask.cpu().numpy()
        mask_tgs_object_np = tg_mask_obj.cpu().numpy()
        mask_tgs_hand_np = tg_mask_hand.cpu().numpy()
        binary_mask_valid_np = valid_mask.cpu().numpy()


        # Overlay each mask with specified colors and transparency
        plt.imshow(mask_tgs_object_np, cmap='Greens', alpha=0.3, extent=(0, 640, 480, 0))
        plt.imshow(mask_tgs_hand_np, cmap='Oranges', alpha=0.3, extent=(0, 640, 480, 0))
        plt.imshow(binary_mask_np, cmap='Reds', alpha=0.5, extent=(0, 640, 480, 0))
        plt.imshow(binary_mask_valid_np, cmap='Blues', alpha=0.5, extent=(0, 640, 480, 0))

        # Add legends manually
        import matplotlib.patches as mpatches

        green_patch = mpatches.Patch(color='green', label='Object Target Mask')
        orange_patch = mpatches.Patch(color='orange', label='Hand Target Mask')
        red_patch = mpatches.Patch(color='red', label='Hand Predict Mask')
        blue_patch = mpatches.Patch(color='blue', label='Hand Valid Mask')

        plt.legend(handles=[green_patch, orange_patch, red_patch, blue_patch], loc='upper right')

        if isinstance(iou, torch.Tensor):
            iou_value = iou.item()
        else:
            iou_value = float(iou)
        
        # Define the text properties
        iou_text = f'IoU: {iou_value:.2f}'
        font_size = 14
        text_color = 'white'
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)

        # Position the text at the top-left corner
        plt.text(10, 30, iou_text, fontsize=font_size, color=text_color, bbox=bbox_props)


        # Remove axes
        plt.axis('off')

        # Set title
        plt.title('Overlayed Binary Masks with Different Colors and Transparency')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save and show the plot
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Mask image saved to {save_path}")
        if show:
            plt.show()

    def save_data(self):
        """
        Saves the instance's attributes to a pickle file.
        """
        # Ensure the directory exists
        filename = self.save_dir + '/ray_hit.pkl'
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created missing directories: {directory}")

        # Prepare the data dictionary
        ray_hit_data = {
            'hand_hit_idxs': self.hand_hit_idxs,
            'obj_hits': self.obj_hits,
            'incidence_rads': self.incidence_rads,
            'hand_hit_idxs_occ': self.hand_hit_idxs_occ,
            'obj_hits_occ': self.obj_hits_occ,
            'incidence_rads_occ': self.incidence_rads_occ,            
            'iou_list': self.iou_list,
            'best_frame': self.best_frame,
            'valid_frames': self.valid_frames,
            # "obj_occluded_vert_flags": self.obj_occluded_vert_flags,
            "finger_names": self.finger_names,
            "finger_avail_idx": self.finger_avail_idx,
            "hit_finger_flags": self.finger_hit_fags,


        }

        # Save the data using pickle
        with open(filename, 'wb') as file:
            pickle.dump(ray_hit_data, file)

        print(f"Data successfully saved to {filename}")

    def load_data(self):
        """
        Loads the instance's attributes from a pickle file.
        """
        filename=self.save_dir + '/ray_hit.pkl'
        with open(filename, 'rb') as file:
            ray_hit_data = pickle.load(file)

        # Assign the loaded data to the instance's attributes
        self.hand_hit_idxs = ray_hit_data['hand_hit_idxs']
        self.obj_hits = ray_hit_data['obj_hits']
        self.incidence_rads = ray_hit_data['incidence_rads']
        self.hand_hit_idxs_occ = ray_hit_data['hand_hit_idxs_occ']
        self.obj_hits_occ = ray_hit_data['obj_hits_occ']
        self.incidence_rads_occ = ray_hit_data['incidence_rads_occ']        
        self.best_frame = ray_hit_data['best_frame']
        self.valid_frames = ray_hit_data['valid_frames']
        # self.obj_occluded_vert_flags = ray_hit_data['obj_occluded_vert_flags']
        self.finger_names = ray_hit_data['finger_names']
        self.finger_avail_idx = ray_hit_data['finger_avail_idx']
        self.hit_finger_flags = ray_hit_data['hit_finger_flags']
        print("*************")
        print(f"best_frame: {self.best_frame}")
        print("*************")

        print(f"Data successfully loaded from {filename}")        

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def prepare_data(self, preds, meta):
       # read the meta data
        self.meta = meta
        device = "cuda"
        self.K = meta["K"]
        self.o2c = meta['o2c']
        self.c2o = torch.inverse(self.o2c)
        self.imsize = cv2.imread(meta["im_paths"][0]).shape[:2] # H, W
        
        # construct the target masks
        batch_masks = np.stack(
            [np.array(Image.open(mask_path)) for mask_path in meta['mask_paths']], axis=0
        )
        batch_masks = torch.tensor(batch_masks).to(preds['right.v3d_cam'].device)
        self.mask_tgs = construct_targets(batch_masks)

        # load mamo hand mesh in camera coordinate
        
        v3d_sealed_h, faces_sealed_h = seal_mano_mesh(preds['right.v3d_cam'], preds['right.f3d'], is_rhand=True)
        self.meshe_hs_c = create_meshes(v3d_sealed_h, faces_sealed_h, device)

        # load mamo hand mesh in object coordinate
        v3d_sealed_h, faces_sealed_h = seal_mano_mesh(preds['right.v3d_obj'], preds['right.f3d'], is_rhand=True)
        v3d_sealed_h = v3d_sealed_h.detach().cpu().numpy()
        faces_sealed_h = torch.tile(faces_sealed_h[None], (len(v3d_sealed_h), 1, 1)).cpu().numpy()
        self.mesh_hs_o = [trimesh.Trimesh(vertices=v3d_sealed_h[i], faces=faces_sealed_h[i], process=False) for i in range(faces_sealed_h.shape[0])]

        # load object mesh in object coordinate
        v3d_sealed_o = preds['object.v3d_obj']
        faces_sealed_o = preds['object.f3d']
        v3d_sealed_o = v3d_sealed_o.detach().cpu().numpy()
        faces_sealed_o = torch.tile(faces_sealed_o[None], (len(v3d_sealed_o), 1, 1)).cpu().numpy()

        self.mesh_os_o = [trimesh.Trimesh(vertices=v3d_sealed_o[i], faces=faces_sealed_o[i], process=False) for i in range(faces_sealed_o.shape[0])]

        K_4x4 = torch.eye(4).to(device)
        K_4x4[:3, :3] = self.K
        K_4x4 = torch.tile(K_4x4[None], (len(preds['right.v3d_cam']), 1, 1))
        # render the hand mask and get the iou between the hand rendering mask and target mask
        rasterizer, shader, self.renderer = create_silhouette_renderer(
            K_4x4[0][None], device, self.imsize
        )
        self.preds = preds
        self.hand_contact_idx = preds["right.contact_idx"]
        self.hand_contact_idx_index_finger = preds["right.contact_idx_index_finger"]
        self.hand_contact_idx_middle_finger = preds["right.contact_idx_middle_finger"]
        self.hand_contact_idx_ring_finger = preds["right.contact_idx_ring_finger"]
        self.hand_contact_idx_little_finger = preds["right.contact_idx_little_finger"]
        self.hand_contact_idx_thumb_finger = preds["right.contact_idx_thumb_finger"]

        self.hand_faces_idx_index_finger = preds["right.faces_idx_index_finger"]
        self.hand_faces_idx_middle_finger = preds["right.faces_idx_middle_finger"]
        self.hand_faces_idx_ring_finger = preds["right.faces_idx_ring_finger"]
        self.hand_faces_idx_little_finger = preds["right.faces_idx_little_finger"]
        self.hand_faces_idx_thumb = preds["right.faces_idx_thumb"]
        self.hand_faces = preds['right.f3d']

        self.finger_names = ["index", "middle", "ring", "little", "thumb"]
        self.finger_avail_idx = [self.hand_contact_idx_index_finger, 
                                    self.hand_contact_idx_middle_finger,
                                    self.hand_contact_idx_ring_finger,
                                    self.hand_contact_idx_little_finger,
                                    self.hand_contact_idx_thumb_finger
                                     ]        

    def get_hand_visible_mask(self):
        mask_h_visibles = []
        mask_h_visible_ious = []
        for hand_idx in tqdm(range(len(self.meshe_hs_c)), desc="Processing hand mask"):
            mask_h_render = self.renderer(
                meshes_world=self.meshe_hs_c[hand_idx], image_size=self.imsize, bin_size=-1
            )[..., 3]
            mask_h_render = (mask_h_render[0] > 0.5).float()
            # mask_o_target = 1 - mask_tgs["object"][hand_idx]
            mask_h_visible = mask_h_render  * self.mask_tgs["right"][hand_idx]
            mask_h_visibles.append(mask_h_visible)
            mask_h_visible_iou = (mask_h_visible).sum()
            mask_h_visible_ious.append(mask_h_visible_iou)
            # mask_save_path = self.save_dir + "/" + os.path.basename(self.meta["mask_paths"][hand_idx])
            # self.save_masks(mask_h_render, self.mask_tgs["object"][hand_idx], self.mask_tgs["right"][hand_idx], mask_h_visible, mask_h_visible_iou, mask_save_path)
            if 0 and hand_idx == 5:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 15))
                ax1.imshow(mask_h_render.detach().cpu().numpy())
                ax1.set_title('Rendered Mask')
                ax2.imshow(mask_h_visible.detach().cpu().numpy())
                ax2.set_title('Visible Mask')
                ax3.imshow(self.mask_tgs["right"][hand_idx].detach().cpu().numpy())
                ax3.set_title('Target Mask')
                plt.show()
        return mask_h_visibles, mask_h_visible_ious

    def obtain_visible_contact(self, mask_h_visibles):
        """Get visible contact points between hand and object meshes from the visible hand mask using ray casting.
        
        Args:
            mask_h_visible: Visible hand mask
            
        Returns:
            dict containing:
                hand_hit_idxs: Indices of hand vertices that make contact
                nearest_hits: 3D locations of nearest object intersection points
                incident_rads: Incident angles at intersection points
        """
        hand_hits_idxs_list = []
        nearest_hits_list = []
        incident_rads_list = []
        # get the ray hit for each hand+object mesh from the visible hand mask
        for mesh_i in tqdm(range(len(self.mesh_hs_o)), desc="Processing ray hit"):
            mesh_h = self.mesh_hs_o[mesh_i]
            mesh_o = self.mesh_os_o[mesh_i]
            mesh = mesh_h + mesh_o
            # mesh = mesh_h
            mask_h_visible = mask_h_visibles[mesh_i]
            hand_contact_points_all = torch.tensor(mesh_h.vertices[self.hand_contact_idx]).float().to("cuda")
            hand_contact_points_all = transform_points(hand_contact_points_all, self.o2c[mesh_i][None]).squeeze(0)
            # hand_contact_points_all = torch.tensor([0.0058418363, 0.10510006, 0.4382386]).float().to("cuda")[None][None]
            hand_contact_points_all_2d = project2d_batch(self.K[None], hand_contact_points_all[None])[0].int()
            hand_contact_idx_hit = []

            for i in range(len(hand_contact_points_all_2d)):
                x, y = hand_contact_points_all_2d[i]
                if x < 0 or x >= self.imsize[1] or y < 0 or y >= self.imsize[0]:
                    continue
                if mask_h_visible[y, x] == 1:
                    hand_contact_idx_hit.append(i)
            # hand_contact_idx_hit = np.array(hand_contact_idx_hit)

            
            if 1:
                # Find matching indices
                if len(hand_contact_idx_hit) > 0:
                    # get the ray shooting from camera origin to the hand contact points
                    hand_hit_idxs = self.hand_contact_idx[hand_contact_idx_hit]
                    nearest_hits = np.full((len(hand_hit_idxs), 3), np.nan, dtype=float)
                    incident_rads = np.full(len(hand_hit_idxs), math.pi/2, dtype=float)
                    cont_vetices = mesh.vertices[hand_hit_idxs]
                    
                    cont_vetices_w = torch.tensor(cont_vetices[None]).float().to("cuda")
                    cont_vetices_c = transform_points(cont_vetices_w, self.o2c[mesh_i][None])

                    cont_2ds = project2d_batch(self.K[None], cont_vetices_c)[0].int()
                    
                    ray_origin, ray_direction = get_ray_origin_direction(self.c2o[mesh_i], self.K, cont_2ds)
                    ray_origin = ray_origin.to('cpu').numpy()
                    ray_direction = ray_direction.to('cpu').numpy()
                    # 6. Perform ray-mesh intersection using trimesh's ray intersector
                    mesh = mesh_o
                    try:
                        # Use PyEmbree for faster ray casting if available
                        ray_mesh = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
                    except ImportError:
                        # Fallback to the slower ray_triangle method
                        ray_mesh = mesh.ray                                    
                    locations, index_ray, index_tri = ray_mesh.intersects_location(
                        ray_origin,
                        ray_direction
                    )

                    # obtain the closest hit point and incident angle for each ray
                    for ray_i in range(len(ray_direction)):
                        match_index_ray = (index_ray == ray_i)
                        if match_index_ray.sum() == 0:
                            continue
                        match_locations = locations[match_index_ray]
                        match_faces = index_tri[match_index_ray]
                        match_tri = mesh.faces[match_faces]
                        # get the distance from the ray origin to the hit locations
                        dists = np.linalg.norm(match_locations - ray_origin[ray_i], axis=1)
                        # get the closest point
                        min_dist_idx = np.argmin(dists)
                        nearest_hits[ray_i] = match_locations[min_dist_idx]
                        nereast_tri = match_tri[min_dist_idx]
                        nearest_vertices = mesh.vertices[nereast_tri]
                        v1 = nearest_vertices[1] - nearest_vertices[0]
                        v2 = nearest_vertices[2] - nearest_vertices[0]
                        normal = np.cross(v1, v2)
                        incident_rad, incident_deg = get_incident_angle(normal, ray_direction[ray_i])
                        incident_rads[ray_i] = incident_rad
                    if 0:
                        # w = np.abs(np.cos(incident_rads)).mean()
                        sigma = 0.5
                        weights = (np.exp(-(incident_rads - np.pi) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))).mean()
                        print(f"mesh_i: {mesh_i}, w: {weights}")
                else:
                    print(f"No contact found for hand mesh: {mesh_i}")
                    hand_hit_idxs = []
                    nearest_hits = []
                    incident_rads = []
            else:
                print(f"No intersection found for hand mesh: {mesh_i}")
                hand_hit_idxs = []
                nearest_hits = []
                incident_rads = []                    
            hand_hits_idxs_list.append(hand_hit_idxs)
            nearest_hits_list.append(nearest_hits)
            incident_rads_list.append(incident_rads)
            # if not debug:
            if 0:
                # if matching_indices != []:
                #     locations = mesh_h.vertices[matching_indices]
                # if nearest_hits != []:
                #     locations += nearest_hits
                # 
                    
                    # locations_show = locations
                # if mesh_i != 23:
                #     continue
                locations_show = []
                if nearest_hits != []:
                    locations_show = nearest_hits[~np.isnan(nearest_hits).all(axis=1)]
                if hand_hit_idxs != []:
                    locations_show = np.concatenate([locations_show, mesh_h.vertices[hand_hit_idxs]], axis=0)
                    if len(locations_show) > 0:
                        mesh = mesh_h + mesh_o
                        print(f"mesh_i: {mesh_i}, obj_hits: {(~np.isnan(nearest_hits).all(axis=1)).sum()}, hand_hits: {len(hand_contact_idx_hit)}")
                        show_ray_hit(ray_origin[0], locations_show, mesh, interval=1)

        return hand_hits_idxs_list, nearest_hits_list, incident_rads_list

    def delete_self_occluded_hits(self, hand_hits_idxs_list, nearest_hits_list, incident_rads_list, hand_hit_idxs_occ_list):
        for i in range(len(hand_hit_idxs_occ_list)):
            if hand_hit_idxs_occ_list[i] == []:
                continue
            hand_hit_idxs_occ = hand_hit_idxs_occ_list[i]
            hand_hits_idxs = hand_hits_idxs_list[i]
            del_idxs = np.where(np.isin(hand_hits_idxs, hand_hit_idxs_occ))[0]
            if len(del_idxs) > 0:
                hand_hits_idxs = np.delete(hand_hits_idxs, del_idxs)
                nearest_hits = np.delete(nearest_hits_list[i], del_idxs, axis=0)
                incident_rads = np.delete(incident_rads_list[i], del_idxs)
                hand_hits_idxs_list[i] = hand_hits_idxs
                nearest_hits_list[i] = nearest_hits
                incident_rads_list[i] = incident_rads
        return hand_hits_idxs_list, nearest_hits_list, incident_rads_list

    def fwd(self, preds, meta):
        self.prepare_data(preds, meta)
        mask_h_visibles, mask_h_visible_ious = self.get_hand_visible_mask()
        hand_hits_idxs_list, nearest_hits_list, incident_rads_list = self.obtain_visible_contact(mask_h_visibles)
        best_frame, finger_hit_flags = self.check_best_frame(hand_hits_idxs_list, nearest_hits_list, mask_h_visible_ious)
        finger_hit_flags = self.delete_self_occluded_fingers(finger_hit_flags)
        hand_hit_idxs_occ_list, obj_hits_occ_list, incidence_rads_occ_list = self.obtain_occluded_contact(finger_hit_flags)
        hand_hits_idxs_list, nearest_hits_list, incident_rads_list = self.delete_self_occluded_hits(hand_hits_idxs_list, nearest_hits_list, incident_rads_list, hand_hit_idxs_occ_list)
            
        # occluded_vert_flags_list = self.determine_obj_occluded_verts()
        self.hand_hit_idxs = hand_hits_idxs_list
        self.obj_hits = nearest_hits_list
        self.incidence_rads = incident_rads_list
        self.iou_list = mask_h_visible_ious        
        self.best_frame = best_frame
        self.finger_hit_fags = finger_hit_flags
        self.valid_frames = self.check_valid_frames()
        self.hand_hit_idxs_occ = hand_hit_idxs_occ_list
        self.obj_hits_occ = obj_hits_occ_list
        self.incidence_rads_occ = incidence_rads_occ_list            
        # self.obj_occluded_vert_flags = occluded_vert_flags_list


    def obtain_occluded_contact(self, finger_hit_fags):
        hand_hit_idxs_occ_list = []
        obj_hits_occ_list = []
        incidence_rads_occ_list = []
        for mesh_i in tqdm(range(len(self.mesh_hs_o)), desc="Processing occluded ray hit"):
            mesh_o = self.mesh_os_o[mesh_i]
            mesh_h = self.mesh_hs_o[mesh_i]
            o2c = self.o2c[mesh_i]
            finger_hit_fag = finger_hit_fags[mesh_i]
            
            c2o = torch.inverse(o2c)
        
            finger_occluded_idxs = np.where(finger_hit_fag == False)[0]
            if len(finger_occluded_idxs) == 0:
                hand_hit_idxs_occ_list.append([])
                obj_hits_occ_list.append([])
                incidence_rads_occ_list.append([])
                continue
            
            finger_occluded_vert_idxs_list = []
            for finger_occluded_idx in finger_occluded_idxs:
                finger_occluded_vert_idxs_list.append(self.finger_avail_idx[finger_occluded_idx])
            finger_occluded_vert_idxs = np.concatenate(finger_occluded_vert_idxs_list)
            finger_occluded_verts_w = torch.tensor(mesh_h.vertices[finger_occluded_vert_idxs]).float().cuda()
            finger_occluded_verts_c = transform_points(finger_occluded_verts_w[None], o2c[None])[0]
            verts_proj = project2d_batch(self.meta["K"].unsqueeze(0), finger_occluded_verts_c.unsqueeze(0)).squeeze(0)
            ray_origin, ray_direction = get_ray_origin_direction(c2o, self.meta['K'], verts_proj)

            ray_origin = ray_origin.to('cpu').numpy()
            ray_direction = ray_direction.to('cpu').numpy()


            mesh = mesh_o
            try:
                # Use PyEmbree for faster ray casting if available
                ray_mesh = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            except ImportError:
                # Fallback to the slower ray_triangle method
                ray_mesh = mesh.ray                                    
            locations, index_ray, index_tri = ray_mesh.intersects_location(
                ray_origin,
                ray_direction
            )
            
            if 0:
                show_ray_hit(ray_origin[0], locations, mesh, index_tri, interval=1)            

            far_hits = np.full((len(finger_occluded_verts_w), 3), np.nan, dtype=float)
            incident_rads = np.full(len(finger_occluded_verts_w), math.pi/2, dtype=float)

            for ray_i in range(len(ray_direction)):
                match_index_ray = (index_ray == ray_i)
                if match_index_ray.sum() == 0:
                    continue
                match_locations = locations[match_index_ray]
                match_faces = index_tri[match_index_ray]
                match_tri = mesh.faces[match_faces]
                # get the distance from the ray origin to the hit locations
                dists = np.linalg.norm(match_locations - ray_origin[ray_i], axis=1)
                # get the closest point
                max_dist_idx = np.argmax(dists)
                far_hits[ray_i] = match_locations[max_dist_idx]
                far_tri = match_tri[max_dist_idx]
                far_vertices = mesh.vertices[far_tri]
                v1 = far_vertices[1] - far_vertices[0]
                v2 = far_vertices[2] - far_vertices[0]
                normal = np.cross(v1, v2)
                incident_rad, incident_deg = get_incident_angle(normal, ray_direction[ray_i])
                incident_rads[ray_i] = incident_rad
            if 0:
                # w = np.abs(np.cos(incident_rads)).mean()
                sigma = 0.5
                weights = (np.exp(-(incident_rads - np.pi) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))).mean()
                print(f"occluded w {weights}")        
                
            hand_hit_idxs_occ_list.append(finger_occluded_vert_idxs)
            obj_hits_occ_list.append(far_hits)
            incidence_rads_occ_list.append(incident_rads)
        return hand_hit_idxs_occ_list, obj_hits_occ_list, incidence_rads_occ_list

    def visualize_projected_vertices(self, rgb_r, verts_proj):
        """
        Visualize the projected vertices on the rendered image.

        Args:
        - rgb_r (Tensor): The rendered RGB image.
        - verts_proj (Tensor): The 2D projected vertices on the image plane.
        """
        # Convert the image to a NumPy array (if necessary)
        rgb_r = rgb_r.cpu().numpy().astype(np.uint8)  # Convert to NumPy for plotting
        
        # Create a figure
        plt.figure(figsize=(10, 10))
        
        # Display the rendered RGB image
        plt.imshow(rgb_r)
        
        # Extract x and y coordinates of projected vertices
        x_coords = verts_proj[:, 0].cpu().numpy()  # Convert to NumPy for plotting
        y_coords = verts_proj[:, 1].cpu().numpy()
        
        # Plot the projected vertices as red points
        plt.scatter(x_coords, y_coords, color='red', s=10, label="Projected Vertices")
        
        # Optionally, add labels and title
        plt.title("Projected Vertices on Rendered Image")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='best')
        
        # Show the plot
        plt.show()

    def determine_obj_occluded_verts(self, epsilon=2e-2):
        """
        Determine which vertices of the mesh are occluded from the camera's perspective.
        Returns a boolean array where True indicates the vertex is occluded.
        """

        occluded_vert_flags_list = []
        for mesh_i in tqdm(range(len(self.mesh_os_o)), desc="occluded vertices"):
            mesh = self.mesh_os_o[mesh_i]
            o2c = self.o2c[mesh_i]
            # Render depth buffer
            mesh_o_tensors = make_mesh_tensors(mesh)
            rgb_r, depth_r, normal_r = nvdiffrast_render(
                                        K=np.array(self.meta['K'].to('cpu')), 
                                        H=self.meta['im_H_W'][0], 
                                        W=self.meta['im_H_W'][1], 
                                        ob_in_cams=o2c, 
                                        context='cuda', 
                                        get_normal=True, 
                                        glctx=dr.RasterizeCudaContext(), 
                                        mesh_tensors=mesh_o_tensors, 
                                        output_size=self.meta['im_H_W'], 
                                        bbox2d=None, 
                                    use_light=True)
            rgb_r = rgb_r[0]
            depth_r = depth_r[0]
            normal_r = normal_r[0]
            # show the rendered image
            if 0:
                plt.imshow(rgb_r.cpu().numpy())
                plt.show()
                # show the rendered normal
                plt.imshow(depth_r.cpu().numpy())
                plt.show()
            
            verts_w = torch.tensor(mesh.vertices).float().cuda()
            verts_c = transform_points(verts_w[None], o2c[None])[0]        
            # Project vertices to 2D
            verts_proj = project2d_batch(self.meta["K"].unsqueeze(0), verts_c.unsqueeze(0)).squeeze(0)  # Shape: (N, 2)
            x = verts_proj[:, 0]
            y = verts_proj[:, 1]
            z = verts_c[:, 2]  # Shape: (N,)

            # Image dimensions
            H, W = self.meta['im_H_W']

            # Floor the coordinates and clamp to valid range
            xi = torch.floor(x).long()
            yi = torch.floor(y).long()

            # Create masks for valid projections
            valid_x = (xi >= 0) & (xi < W)
            valid_y = (yi >= 0) & (yi < H)
            valid = valid_x & valid_y  # Shape: (N,)

            # Initialize occlusion array as all False
            is_occluded = torch.zeros(mesh.vertices.shape[0], dtype=torch.bool, device='cuda')

            # Handle out-of-bounds projections
            is_occluded[~valid] = True

            # Process only valid projections
            if valid.any():
                valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
                xi_valid = xi[valid]
                yi_valid = yi[valid]
                z_valid = z[valid]
                
                # Gather buffer_z from depth buffer
                buffer_z = depth_r[yi_valid, xi_valid]
                
                # Compare z to buffer_z + epsilon
                occluded_mask = z_valid > (buffer_z + epsilon)
                
                # Update occlusion array
                is_occluded[valid_indices] = occluded_mask
            occluded_vert_flags_list.append(is_occluded)
        # self.visualize_occlusion(mesh, is_occluded.cpu())
        return occluded_vert_flags_list
             

    def visualize_occlusion(self, mesh, is_occluded):
        """
        Visualize the mesh with occluded vertices highlighted in red.
        """
        # Create colors: visible vertices in white, occluded in red
        colors = np.tile(np.array([1.0, 1.0, 1.0, 1.0]), (mesh.vertices.shape[0], 1))
        colors[is_occluded] = np.array([1.0, 0.0, 0.0, 1.0])  # Red color for occluded vertices
        
        # Create a trimesh object with vertex colors
        mesh_visual = trimesh.Trimesh(vertices=mesh.vertices,
                                    faces=mesh.faces,
                                    vertex_colors=colors)
        # Show the mesh using trimesh's viewer
        mesh_visual.show()


    def check_valid_frames(self):
        valid_frames = np.zeros(len(self.iou_list)).astype(bool)
        for i in range(len(self.iou_list)):
            # if self.iou_list[i] > 2000:
            if 1:
                valid_frames[i] = True
        return valid_frames

    def delete_self_occluded_fingers(self, hand_hit_flags):
        faces_index_finger = self.hand_faces[self.hand_faces_idx_index_finger]
        faces_middle_finger = self.hand_faces[self.hand_faces_idx_middle_finger]
        faces_ring_finger = self.hand_faces[self.hand_faces_idx_ring_finger]
        faces_little_finger = self.hand_faces[self.hand_faces_idx_little_finger]
        faces_thumb = self.hand_faces[self.hand_faces_idx_thumb]
        faces_finger_list = [faces_index_finger, faces_middle_finger, faces_ring_finger, faces_little_finger, faces_thumb]

        for hand_idx in range(len(hand_hit_flags)):
            # print(f"frame_idx {frame_idx}")
            o2c = self.o2c[hand_idx]
            
            hand_hit_fag = hand_hit_flags[hand_idx]
            # print(f"before finger_hit_fags {finger_hit_fags}")
            finger_r_list = []
            mesh_h_o = self.mesh_hs_o[hand_idx]
            for finger_idx, finger_hit_flag in enumerate(hand_hit_fag):
                if not finger_hit_flag:
                    finger_r_list.append({"depth": torch.tensor([]), "mask": torch.tensor([])})
                else:
                    mesh_h = trimesh.Trimesh(vertices=mesh_h_o.vertices,
                        faces=faces_finger_list[finger_idx].cpu().numpy(),
                        )
                    if 0:
                        mesh_h.show()
                    mesh_h_tensors = make_mesh_tensors(mesh_h)
                    # width and height must be divisible by 8
                    H = self.meta['im_H_W'][0]
                    W = self.meta['im_H_W'][1]
                    H = H // 8 * 8
                    W = W // 8 * 8
                    rgb_r, depth_r, normal_r = nvdiffrast_render(
                            K=np.array(self.meta['K'].to('cpu')), 
                            H=H, 
                            W=W, 
                            ob_in_cams=o2c, 
                            context='cuda', 
                            get_normal=True, 
                            glctx=dr.RasterizeCudaContext(), 
                            mesh_tensors=mesh_h_tensors, 
                            output_size=(H, W), 
                            bbox2d=None, 
                        use_light=True)
                    rgb_r = rgb_r[0]
                    depth_r = depth_r[0]
                    normal_r = normal_r[0]
                    mask_r = depth_r > 0.01
                    if 0:
                        plt.imshow(depth_r.cpu().numpy())
                        plt.show()
                        # show the rendered normal
                        plt.imshow(mask_r.cpu().numpy())
                        plt.show()                    
                    finger_r_list.append({"depth": depth_r, "mask": mask_r})
            for finger_idx_first in range(len(finger_r_list)):
                for finger_idx_second in range(finger_idx_first+1, len(finger_r_list)):
                    if len(finger_r_list[finger_idx_first]['depth']) > 0 and len(finger_r_list[finger_idx_second]['depth']) > 0:
                        mask1 = finger_r_list[finger_idx_first]['mask']
                        mask2 = finger_r_list[finger_idx_second]['mask']
                        iou = self.compute_iou(mask1.cpu().numpy(), mask2.cpu().numpy())
                        # print(f"iou {iou}")
                        if iou > 0.3:
                            depth1 = finger_r_list[finger_idx_first]['depth']
                            depth2 = finger_r_list[finger_idx_second]['depth']
                            depth1_mean = depth1[mask1].mean()
                            depth2_mean = depth2[mask2].mean()
                            if depth1_mean > depth2_mean:
                                hand_hit_fag[finger_idx_first] = False
                            else:
                                hand_hit_fag[finger_idx_second] = False
            # print(f"after finger_hit_fags {finger_hit_fags}")
        return hand_hit_flags


    def compute_iou(self, mask1, mask2):
        # Ensure masks are boolean
        mask1_bool = mask1.astype(bool)
        mask2_bool = mask2.astype(bool)
        
        intersection = np.logical_and(mask1_bool, mask2_bool)
        union = np.logical_or(mask1_bool, mask2_bool)
        
        intersection_count = intersection.sum()
        union_count = union.sum()
        
        # Handle the case where union might be zero
        if union_count == 0:
            return 1.0 if intersection_count == 0 else 0.0
        
        iou = intersection_count / union_count
        return iou

    def check_best_frame(self, hand_hit_idxs, obj_hits, iou_list):
        hit_finger_flags = np.zeros((len(hand_hit_idxs), 5), dtype=bool)
        obj_hit_valids = []
        least_contact_number = 1
        for i in range(len(hand_hit_idxs)):
            hand_hit_idx = hand_hit_idxs[i]
            if hand_hit_idx == []:
                obj_hit_valids.append(np.array([]))
                continue
            obj_hit = obj_hits[i]
            obj_hit_valid = ~np.isnan(obj_hit).all(axis=1)
            obj_hit_valids.append(obj_hit_valid)
            hand_hit_idx_valid = hand_hit_idx[obj_hit_valid]
            if len(hand_hit_idx_valid) == 0:
                continue
            hand_hit_index_finger = np.isin(hand_hit_idx_valid, self.hand_contact_idx_index_finger).sum() >= least_contact_number 
            hand_middle_finger = np.isin(hand_hit_idx_valid, self.hand_contact_idx_middle_finger).sum() >= least_contact_number
            hand_ring_finger = np.isin(hand_hit_idx_valid, self.hand_contact_idx_ring_finger).sum() >= least_contact_number
            hand_little_finger = np.isin(hand_hit_idx_valid, self.hand_contact_idx_little_finger).sum() >= least_contact_number
            hand_thumb_finger = np.isin(hand_hit_idx_valid, self.hand_contact_idx_thumb_finger).sum() >= least_contact_number
            hit_finger_flags[i] = [hand_hit_index_finger, hand_middle_finger, hand_ring_finger, hand_little_finger, hand_thumb_finger]
        hit_finger_sums = hit_finger_flags.sum(axis=1)
        if 0:
            for i, hit_finger_flag in enumerate(hit_finger_flags):
                print(f"{i}, hit_finger_flag {hit_finger_flag}, {hit_finger_sums[i]}")        
        max_hit_frames = np.where(hit_finger_sums == np.max(hit_finger_sums))[0]
        best_frame = max_hit_frames[0]
        for frame in max_hit_frames:
            if iou_list[frame] > iou_list[best_frame]:
                best_frame = frame
            elif iou_list[frame] == iou_list[best_frame]:
                if obj_hit_valids[frame].sum() > obj_hit_valids[best_frame].sum() :
                    best_frame = frame
        # self.best_frame = 5
        print("*************")
        print(f"best_frame: {best_frame}")
        print("*************")        

        return best_frame, hit_finger_flags