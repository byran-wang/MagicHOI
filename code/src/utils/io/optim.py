import os.path as op
from glob import glob

import numpy as np
import torch
import trimesh
from src.fitting.utils import remesh_and_clean_mesh
from common.xdict import xdict
from src.model.mano.server import MANOServer
from src.model.obj.server import ObjectServer


def load_file(vis_p: str) -> dict:
    """
    Loads visualization data from a NumPy file.

    Parameters:
        vis_p (str): Path to the NumPy file.

    Returns:
        dict: Loaded data.
    """
    return np.load(vis_p, allow_pickle=True).item()

def load_data(
        sd_p,
        # align_p = "data/hold_MC1_ho3d/processed//hold_fit.aligned.npy",
        # pose_p = "data/hold_MC1_ho3d/processed//colmap_hold_MC1_ho3d.80/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy",
        # object_mesh_f = "/home/simba/Documents/project/diff_object/threestudio/outputs/hold_MC1_ho3d.80/3d_ref/save/it1000-export/model.obj",
        align_p = "",
        pose_p = "",
        object_mesh_f = "",        
        ):
    
    device = "cuda:0"
    print("Loading data from", sd_p)
    o2c_mat = np.load(pose_p)

    ckpt = torch.load(sd_p, map_location="cpu")
    sd = xdict(ckpt["state_dict"])
    param_dict = sd.search(".params.").to(device)
    for k in param_dict.keys():
        param_dict[k].requires_grad = True

    obj_scale = list(sd.search(".obj_scale").values())[0]

    misc_ps = sorted(glob(op.join("logs", sd_p.split("/")[1], "misc", "*")))
    misc = np.load(misc_ps[-1], allow_pickle=True).item()

    fnames = misc["img_paths"]
    full_seq_name = fnames[0].split("/")[2]
    K = torch.FloatTensor(misc["K"]).to(device).view(1, 4, 4)[:, :3, :3]
    w2c = misc["w2c"].view(1, 4, 4).inverse().to(device)
    scene_scale = misc["scale"]
    w2c = torch.FloatTensor(o2c_mat).to(device)

        # mesh_c_o = misc["mesh_c_o"] if "mesh_c_o" in misc else misc["object_cano"]
    mesh_c_o = trimesh.load(object_mesh_f, process=False)
    mesh_c_o = decimate_mesh(sd_p, mesh_c_o)

    assert len(fnames) == list(param_dict.search(".global_orient").values())[0].shape[0]

    # node ids
    node_ids = []
    for k in param_dict.keys():
        node_id = k.split(".")[2]
        node_ids.append(node_id)
    node_ids = list(set(node_ids))

    servers = {}
    faces = {}
    for node_id in node_ids:
        if "right" in node_id or "left" in node_id:
            hand = "right" if "right" in node_id else "left"
            is_right = hand == "right"
            server = MANOServer(betas=None, is_rhand=is_right)
            myfaces = torch.LongTensor(server.faces.astype(np.int64)).to(device)
        elif "object" in node_id:
            server = ObjectServer(full_seq_name, template=mesh_c_o)
            server.object_model.obj_scale = obj_scale
            server.to(device)
            myfaces = torch.LongTensor(mesh_c_o.faces).to(device)

        else:
            assert False, f"Unknown node id: {node_id}"

        servers[node_id] = server
        faces[node_id] = myfaces

    boxes_p = op.join("data", full_seq_name, "build", "boxes.npy")
    if op.exists(boxes_p):
        boxes = np.load(boxes_p)
    else:
        boxes = None

    num_frames = len(fnames)
    scene_scale = torch.tensor([scene_scale]).float().to(device)
    out = {}
    out["fnames"] = fnames
    out["param_dict"] = param_dict
    out["servers"] = servers
    out["faces"] = faces
    out["node_ids"] = node_ids
    out["scene_scale"] = scene_scale
    out["w2c"] = w2c.cpu()
    out["K"] = K  # .cpu().numpy()
    out["boxes"] = boxes
    print("Done loading ckpt")
    out["num_frames"] = num_frames
    return out, ckpt


def decimate_mesh(sd_p, mesh_c_o):
    print("Decimating mesh")
    mesh_p = op.abspath(op.join(op.dirname(sd_p), "..", "mesh_c.obj"))

    mesh_c_o.export(mesh_p)
    print("Exported mesh to", mesh_p)

    mesh_decimate_p = op.join(op.dirname(mesh_p), "mesh_c_decimate.obj")

    remesh_and_clean_mesh(mesh_p, mesh_decimate_p, target_face_count=5000)
    print("Exported decimated mesh to", mesh_decimate_p)

    mesh_c_o = trimesh.load(mesh_decimate_p, process=False)
    print("Loaded mesh from", mesh_decimate_p)
    print(f"\tVertices count: {mesh_c_o.vertices.shape[0]}")
    print(f"\tFaces count: {mesh_c_o.faces.shape[0]}")
    print(f"\tIs watertight: {mesh_c_o.is_watertight}")
    return mesh_c_o
