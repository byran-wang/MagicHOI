import os.path as op
from glob import glob

import common.viewer as viewer_utils
import numpy as np
import torch
import trimesh
from common.body_models import build_mano_aa
from common.transforms import project2d_batch, rigid_tf_torch_batch
from common.viewer import ViewerData
from PIL import Image

# from src_data.preprocessing_utils import tf.cv2gl_mano
import common.transforms as tf

# from src_data.smplx import MANO
from common.xdict import xdict
from src.utils.eval_modules import compute_bounding_box_centers
from src.utils.const import SEGM_IDS

import trimesh

import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.geometry import transform_points








def load_data(full_seq_name, get_selected_fids_fn=None):
    from smplx import MANO

    # load in opencv format

    seq_name = full_seq_name.split("_")[1]

    device = "cuda:0"
    human_model = MANO(
        "./code/body_models", is_rhand=True, flat_hand_mean=False, use_pca=False
    ).to(device)
    data = torch.load(f"./ho3d_v3/processed/{seq_name}.pt")
    mano_layer = build_mano_aa(True, flat_hand=False)

    fnames = data["fnames"]
    fnames = [fname.replace("./generator/assets/", "./") for fname in fnames]
    
    hand_pose = data["hand_pose"]
    hand_beta = data["hand_beta"]
    hand_transl = data["hand_transl"]
    K = data["K"]
    obj_trans = data["obj_trans"]
    obj_rot = data["obj_rot"]
    obj_name = data["obj_name"]
    is_valid = data["is_valid"]
    not_valid = (1 - is_valid).bool()
    assert len(fnames) == obj_trans.shape[0]
    assert len(fnames) == obj_rot.shape[0]
    # get all png files in the directory and sort them
    if get_selected_fids_fn is None:
        # Default behavior
        glob_path = f"./data/{full_seq_name}/processed/images/*.png"
        selected_fnames = sorted(glob(glob_path))
        selected_fids = np.array(
            [(int(op.basename(fname).split(".")[0])) * 5 for fname in selected_fnames]
        )
    else:
        # Use callback function
        selected_fids = get_selected_fids_fn(full_seq_name)
    assert len(selected_fids) > 0
    obj_mesh = trimesh.load(
        f"./ho3d_v3/models/{obj_name}/textured_simple.obj",
        process=False,
    )

    # OpenGL to OpenCV
    num_frames = hand_pose.shape[0]
    hand_rot = hand_pose[:, :3].numpy()
    hand_pose = hand_pose[:, 3:].numpy()
    hand_transl = hand_transl.numpy()
    hand_rot_all = []
    hand_transl_all = []
    for idx in range(num_frames):
        T_hip = (
            human_model.get_T_hip(betas=hand_beta[idx : idx + 1].to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        hand_rot_cv, hand_transl_cv = tf.cv2gl_mano(
            hand_rot[idx], hand_transl[idx], T_hip.reshape(3)
        )
        hand_rot_all.append(hand_rot_cv)
        hand_transl_all.append(hand_transl_cv)
    hand_rot_all = np.array(hand_rot_all)
    hand_transl_all = np.array(hand_transl_all)

    hand_pose = torch.FloatTensor(hand_pose)
    hand_rot = torch.FloatTensor(hand_rot_all)
    hand_transl = torch.FloatTensor(hand_transl_all)

    with torch.no_grad():
        mano_output = mano_layer(
            betas=hand_beta,
            hand_pose=hand_pose,
            global_orient=hand_rot,
            transl=hand_transl,
        )

        v3d_h = mano_output.vertices  # .numpy()
        j3d_h = mano_output.joints  # .numpy()

        num_frames = hand_transl.shape[0]
    v_cano_o = torch.FloatTensor(obj_mesh.vertices).repeat(num_frames, 1, 1)
    c_cano_o = torch.FloatTensor(obj_mesh.visual.to_color().vertex_colors)

    Rt_o = torch.eye(4)[None, :, :].repeat(num_frames, 1, 1)
    Rt_o[:, :3, :3] = obj_rot
    Rt_o[:, :3, 3] = obj_trans
    Rt_o[:, 1:3] *= -1

    v3d_o_cam = rigid_tf_torch_batch(v_cano_o, Rt_o[:, :3, :3], Rt_o[:, :3, 3:])
    v2d_o = project2d_batch(K, v3d_o_cam)
    v2d_h = project2d_batch(K, v3d_h)

    # masks_ps = sorted(glob(f"./data/{full_seq_name}/build/mask/*"))
    # masks_gt = np.stack([Image.open(mask_p) for mask_p in masks_ps], axis=0)
    # boxes = np.load(f"./data/{full_seq_name}/build/boxes.npy")
    # from src.fitting.utils import crop_masks

    hand_id = SEGM_IDS["right"]
    obj_id = SEGM_IDS["object"]
    # masks_gt = crop_masks(masks_gt, boxes, hand_id, obj_id, scale=0.7)
    # masks_gt = torch.FloatTensor(masks_gt)

    DUMMPY_VAL = -1000
    v3d_h[not_valid, :] = DUMMPY_VAL
    v3d_o_cam[not_valid, :] = DUMMPY_VAL
    j3d_h[not_valid, :] = DUMMPY_VAL
    v2d_h[not_valid, :] = DUMMPY_VAL
    v2d_o[not_valid, :] = DUMMPY_VAL

    # Select ground truth data based on selected file IDs
    v3d_h = v3d_h[selected_fids]
    v3d_o_cam = v3d_o_cam[selected_fids]
    j3d_h = j3d_h[selected_fids]
    v2d_h = v2d_h[selected_fids]
    v2d_o = v2d_o[selected_fids]
    fnames = np.array(fnames)[selected_fids]
    is_valid = is_valid[selected_fids]

    out = {}
    out["fnames"] = fnames
    out["v3d_c.right"] = v3d_h.detach().numpy()
    out["v3d_c.object"] = v3d_o_cam.detach().numpy()
    out["j3d_c.right"] = j3d_h.detach().numpy()
    out["v2d_h"] = v2d_h.detach().numpy()
    out["v2d_o"] = v2d_o.detach().numpy()
    out["faces.object"] = np.array(obj_mesh.faces)
    out["faces.right"] = np.array(human_model.faces)
    out["K"] = K[0].numpy()
    out["c2o"] = Rt_o[selected_fids].detach().numpy()
    out["colors.object"] = c_cano_o.detach().numpy()
    # out["masks_gt"] = masks_gt.detach().numpy()
    out["is_valid"] = is_valid.detach().numpy()

    # rh
    root_j3d = j3d_h[:, :1]
    root_o = torch.FloatTensor(
        compute_bounding_box_centers(v3d_o_cam.detach().numpy())[:, None]
    )
    out["v3d_right.object"] = torch.stack(
        [verts - rj3d for verts, rj3d in zip(v3d_o_cam, root_j3d)], dim=0
    ).numpy()
    out["j3d_ra.right"] = j3d_h - root_j3d
    out["v3d_ra.object"] = v3d_o_cam - root_o
    out["root.object"] = root_o[:, 0]
    out = xdict(out).to_torch()
    return out


def load_viewer_data(args, get_selected_fids_fn=None):
    full_seq_name = args.seq_name
    data = load_data(full_seq_name, get_selected_fids_fn)

    v3d_h_c = data["v3d_c.right"].numpy()
    v3d_o_c = data["v3d_c.object"].numpy()

    faces_o = data["faces.object"].numpy()
    faces_h = data["faces.right"].numpy()
    K = data["K"].numpy().reshape(3, 3)
    
    fnames = data["fnames"]
    from common.body_models import seal_mano_mesh_np

    v3d_h_c, faces_h = seal_mano_mesh_np(v3d_h_c, faces_h, is_rhand=True)
    
    if 0:
        out_dir = f"outputs_gt/{args.seq_name}"
        os.makedirs(out_dir, exist_ok=True)

        # Export merged meshes (hand + object) per frame
        for i in range(len(v3d_h_c)):
            # Construct the first mesh
            trans_h = data['c2o'][i][:3, 3].numpy()[None, :]
            mesh_h = trimesh.Trimesh(vertices=v3d_h_c[i] - trans_h, faces=faces_h)

            # Construct the second mesh
            mesh_o = trimesh.Trimesh(vertices=v3d_o_c[i] - trans_h, faces=faces_o)

            # Combine vertices
            vertices = np.vstack([mesh_h.vertices, mesh_o.vertices])

            # Offset faces of the second mesh
            faces_o_offset = mesh_o.faces + len(mesh_h.vertices)

            # Combine faces
            faces = np.vstack([mesh_h.faces, faces_o_offset])

            # Create the merged mesh
            merged_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            mesh_h.export(f"{out_dir}/right-gt-{i}.obj")
            mesh_o.export(f"{out_dir}/object-gt-{i}.obj")
            merged_mesh.export(f"{out_dir}/right-object-gt-{i}.obj")
    

    vis_dict = {}
    vis_dict["right-gt"] = {
        "v3d": v3d_h_c,
        "f3d": faces_h,
        "vc": None,
        "name": "right-gt",
        "color": "gray_white",
        "flat_shading": True,
    }

    vis_dict["obj-gt"] = {
        "v3d": v3d_o_c,
        "f3d": faces_o,
        "vc": None,
        "name": "object-gt",
        "color": "green-blue",
        # "color": "red",
        "flat_shading": False,
    }

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )
    num_frames = len(fnames)
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    im = Image.open(fnames[0])
    cols, rows = im.size

    images = [Image.open(im_p) for im_p in fnames]
    data = ViewerData(Rt, K, cols, rows, images)
    return meshes, data


def load_data_diff_object(
        full_seq_name = "",
        mvs_root = "", 
        debug = False,
):

    seq_name = full_seq_name.split("_")[1]
    from smplx import MANO

    # load in opencv format
    device = "cuda:0"
    human_model = MANO(
        "./code/body_models", is_rhand=True, flat_hand_mean=False, use_pca=False
    ).to(device)

    data = torch.load(f"./ho3d_v3/processed/{seq_name}.pt")
    mano_layer = build_mano_aa(True, flat_hand=False)

    fnames = data["fnames"]
    fnames = [fname.replace("./", "../") for fname in fnames]
    hand_pose = data["hand_pose"]
    hand_beta = data["hand_beta"]
    hand_transl = data["hand_transl"]
    K = data["K"]
    obj_trans = data["obj_trans"]
    obj_rot = data["obj_rot"]
    obj_name = data["obj_name"]
    is_valid = data["is_valid"]
    not_valid = (1 - is_valid).bool()
    assert len(fnames) == obj_trans.shape[0]
    assert len(fnames) == obj_rot.shape[0]
    obj_mesh_f = f"./ho3d_v3/models/{obj_name}/textured_simple.obj"
    obj_mesh = trimesh.load(obj_mesh_f, process=False,)
    obj_3d_o = np.tile(np.array(obj_mesh.vertices), (obj_rot.shape[0], 1, 1))
    
    o2c = torch.tile(torch.eye(4), (obj_trans.shape[0], 1, 1))
    o2c[:, :3, :3] = obj_rot
    o2c[:, :3, 3] = obj_trans
    o2c[:, 1:3] *= -1 # gl to cv
    c2o = torch.inverse(o2c) # o2c -> c2o


    # OpenGL to OpenCV
    num_frames = hand_pose.shape[0]
    hand_rot = hand_pose[:, :3].numpy()
    hand_pose = hand_pose[:, 3:].numpy()
    hand_transl = hand_transl.numpy()
    hand_rot_all = []
    hand_transl_all = []
    for idx in range(num_frames):
        T_hip = (
            human_model.get_T_hip(betas=hand_beta[idx : idx + 1].to(device))
            .squeeze()
            .cpu()
            .numpy()
        )
        hand_rot_cv, hand_transl_cv = tf.cv2gl_mano(
            hand_rot[idx], hand_transl[idx], T_hip.reshape(3)
        )
        hand_rot_all.append(hand_rot_cv)
        hand_transl_all.append(hand_transl_cv)
    hand_rot_all = np.array(hand_rot_all)
    hand_transl_all = np.array(hand_transl_all)

    hand_pose = torch.FloatTensor(hand_pose)
    hand_rot = torch.FloatTensor(hand_rot_all)
    hand_transl = torch.FloatTensor(hand_transl_all)

    with torch.no_grad():
        mano_output = mano_layer(
            betas=hand_beta,
            hand_pose=hand_pose,
            global_orient=hand_rot,
            transl=torch.zeros_like(hand_transl),
        )

        v3d_h_can = mano_output.vertices.cpu().numpy()  # in camera_space
        j3d_h_can = mano_output.joints.cpu().numpy()  # in camera_sapce

        num_frames = hand_transl.shape[0]
    h2c_mat = np.tile(np.eye(4), (o2c.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = hand_transl.cpu().numpy()
    h2c_mat[:, 3, 3] = 1
    h2o_mat = c2o.cpu().numpy() @ h2c_mat        
    
    v3d_h_o = transform_points(v3d_h_can, h2o_mat) # in object space
    j3d_h_o = transform_points(j3d_h_can, h2o_mat) # in object space

    v3d_h_c = transform_points(v3d_h_o, o2c.cpu().numpy()) # in camera space
    j3d_h_c = transform_points(j3d_h_o, o2c.cpu().numpy()) # in camera space
    obj_3d_c = transform_points(obj_3d_o, o2c.cpu().numpy()) # in camera space

    DUMMPY_VAL = -1000
    v3d_h_c[not_valid, :] = DUMMPY_VAL
    # v3d_o_cam[not_valid, :] = DUMMPY_VAL
    j3d_h_c[not_valid, :] = DUMMPY_VAL
    # v2d_h[not_valid, :] = DUMMPY_VAL
    # v2d_o[not_valid, :] = DUMMPY_VAL


    
    selected_fnames = sorted(glob(f"{mvs_root}/../../HO3D/cameras/*.json"))
    assert len(selected_fnames) > 0
    # Get selected file IDs
    selected_fids = np.array(
    [int(op.basename(fname).split(".")[0])*5 for fname in selected_fnames]
    )
    assert len(selected_fids) > 0

    out = xdict()
    hand_v_o = v3d_h_c[selected_fids]
    hand_jnts_o = j3d_h_c[selected_fids]
    hand_jnts_can = j3d_h_can[selected_fids]
    object_verts_o = obj_3d_c[selected_fids]
    out['verts.right'] = hand_v_o
    out['jnts.right'] = hand_jnts_o
    out['root.right'] = hand_jnts_o[:,0,:]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :]
    out['verts.object'] = object_verts_o
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]
    out['is_valid'] = is_valid[selected_fids]
    out["faces.object"] = np.array(obj_mesh.faces)
    print("Done loading data")

    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)


    return out