import torch
import torch.nn as nn
from common.xdict import xdict
from common.transforms import project2d_batch
from src.alignment.loss_terms import gmof
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix
from utils_simba.geometry import transform_points
import sys
sys.path = ["./"] + sys.path
from threestudio.systems.zero123 import Zero123
from threestudio.utils.config import load_config
import trimesh

l1_loss = nn.L1Loss(reduction="none")

def loss_fn_o2(preds, targets_r, targets_l, targets_o, conf):
    targets_o2d = targets_o["o2d.gt"]
    r3d = targets_r['v3d']
    l3d = targets_l['v3d']
    h3d = torch.cat([r3d, l3d], dim=1)
    o3d = preds["o3d"]

    # coarse contact
    centroid_h = h3d.mean(dim=1)
    centroid_o = o3d.mean(dim=1)
    loss = l1_loss(centroid_h, centroid_o).mean() * conf.contact 

    # 2d reprojection
    loss += (
        gmof(preds["o2d"] - targets_o2d, sigma=conf.o2d_sigma).sum(dim=-1).mean()
        * conf.o2d
    )

    # encourage: in front of camera
    z_min = torch.clamp(-o3d[:, :, 2].mean(dim=1), min=0.0)
    if z_min.sum() > 0:
        loss_z = z_min.sum() / torch.nonzero(z_min).shape[0]
        loss += loss_z * conf.z_min
    return loss

def set_system_status(system, ckpt_path):
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    system.set_resume_status(ckpt["epoch"], ckpt["global_step"])   

def export_mesh(sdf, path='torus_mesh.obj'):
    # Define grid parameters
    resolution = 100
    cube_size = 1.0
    x_min, x_max = -cube_size, cube_size
    y_min, y_max = -cube_size, cube_size
    z_min, z_max = -cube_size, cube_size
    dx = (x_max - x_min) / (resolution - 1)
    dy = (y_max - y_min) / (resolution - 1)
    dz = (z_max - z_min) / (resolution - 1)

    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Flatten the grid for SDF evaluation
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Stack and convert to a PyTorch tensor
    points = torch.tensor(np.array([X_flat, Y_flat, Z_flat]), dtype=torch.float32).to('cuda').permute(1, 0)

    # Evaluate the SDF
    with torch.no_grad():
        sdf_values = sdf(points).detach().cpu().numpy()

    # Reshape the SDF values to a 3D grid and transpose to (z, y, x)
    sdf_values = sdf_values.reshape(resolution, resolution, resolution).transpose(0, 1, 2)
    from skimage import measure
    # Apply the Marching Cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(sdf_values, level=0.0, spacing=(dx, dy, dz))
    verts = verts - cube_size

    # Create and export the mesh
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(path)

class ObjectParameters(nn.Module):    
    def __init__(self, data, meta, debug=False, target_face_count=30000):
        super().__init__()
        # unpacking
        K = meta["K"]
        o2c_all = meta["o2c"]
        obj_rot = matrix_to_axis_angle(o2c_all[:, :3, :3])
        obj_transl = o2c_all[:, :3, 3]
        obj_mesh = trimesh.load(
            meta['object_mesh_f'],
            process=False,
        )
        # simplified_mesh = obj_mesh.simplify_quadratic_decimation(face_count=target_face_count)
        simplified_mesh = obj_mesh
        # simplified_mesh.show()

        obj_pts = simplified_mesh.vertices
        obj_f3d = simplified_mesh.faces

        obj_cano = torch.FloatTensor(obj_pts).to('cuda')
        obj_f3d = torch.LongTensor(obj_f3d).to('cuda')             


        obj_cano = obj_cano[:, :3].clone()
        obj_cano = obj_cano.T[None, :, :].repeat(len(o2c_all), 1, 1).permute(0, 2, 1)
        # get the object in camera coordinates
        obj_cam = transform_points(obj_cano, o2c_all)

        K = K[None, :, :].repeat(len(o2c_all), 1, 1)
        obj_2d = project2d_batch(K, obj_cam)    

        # object parameters
        self.register_buffer("o2c_all", o2c_all)
        self.register_buffer("obj_rot", obj_rot)
        self.register_buffer("obj_transl", obj_transl)
        self.register_buffer("obj_cano", obj_cano)
        self.register_buffer("obj_cam", obj_cam)
        self.register_buffer("obj_f3d", obj_f3d)
        self.register_buffer("obj_2d", obj_2d)

        self.K = meta["K"]

        targets = xdict()
        self.targets = targets
        self.im_paths = meta["im_paths"]
        
        cfg = load_config(meta['object_cfg_f'])
        sdf = Zero123.load_from_checkpoint(
            meta['object_ckpt_f'],
            cfg=cfg.system,
            resumed=meta['object_ckpt_f'] is not None)
        sdf.to('cuda')
        sdf.eval()
        set_system_status(sdf, meta['object_ckpt_f'])
        sdf.do_update_step(sdf.true_current_epoch, sdf.true_global_step)
        
        self.sdf = sdf.geometry.forward_sdf
        if debug:
            export_mesh(self.sdf, 'obj_mesh.obj')


    def forward(self):
        out = xdict()
        out["v3d_cam"] = self.obj_cam
        out["j3d_cam"] = self.obj_cam
        out["v3d_obj"] = self.obj_cano
        out["f3d"] = self.obj_f3d
        out["j2d"] = self.obj_2d
        out["sdf"] = self.sdf
        out["im_paths"] = self.im_paths
        out["K"] = self.K

        return out
