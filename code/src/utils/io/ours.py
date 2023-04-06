import os.path as op
from glob import glob

import common.viewer as viewer_utils
import numpy as np
import torch
from common.viewer import ViewerData
from PIL import Image
from common.xdict import xdict
from src.model.mano.server import MANOServer
from src.model.obj.server import ObjectServer
from src.utils.eval_modules import compute_bounding_box_centers
import trimesh
from utils_simba.geometry import get_revert_tvec, transform_points
from utils_simba.hand import initialize_mano_model
import smplx
from common.body_models import seal_mano_mesh
from utils_simba.geometry import get_intrinsic_matrix_from_projection


def map_deform2eval(verts, scale):
    conversion_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    normalize_shift = np.array(
        [0.0085238, -0.01372686, 0.42570806]
    )  # dummy vec at all time

    src_verts = np.copy(verts)

    src_verts = np.dot(src_verts, conversion_matrix)
    src_verts *= scale
    src_verts += normalize_shift
    return src_verts

def load_data_diff_object(
        seq_name = "",
        data_root = "", 
        mvs_root = "", 
        object_mesh_f = "",
        ckpt_p = "",
        # object_mesh_f = "/home/simba/Documents/project/diff_object/threestudio/outputs/hold_MC1_ho3d.80/3d_ref/save/it8500-export/model.obj",
        MANO_f = ""
        ):
    device = "cuda:0"
    print("Loading data")
    
    from src.utils.io.optim import load_data
    # ckpt_p='logs/hold_GPMF14_ho3d_pre_train/checkpoints/last.pose_ref'
    # ckpt_p='logs/' + seq_name.split('.')[0] + '_pre_train/checkpoints/last.pose_ref'
    # "data/hold_MC1_ho3d/processed//colmap_hold_MC1_ho3d.80/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy",
    out, ckpt = load_data(ckpt_p, pose_p=f"{mvs_root}/o2w_normalized_aligned.npy", object_mesh_f=object_mesh_f)
    # breakpoint()
    intrinsic = out["K"]
    num_frames = out['num_frames']
    obj_scale = out["servers"]["object"].object_model.obj_scale.detach().cpu().numpy()                          
    betas = ckpt["state_dict"]['model.nodes.right.params.betas.weight'].clone().cuda().repeat(num_frames,1)
    h2c_rot = ckpt["state_dict"]['model.nodes.right.params.global_orient.weight'].clone().cuda()
    h2c_transl = ckpt["state_dict"]['model.nodes.right.params.transl.weight'].clone().cuda()
    hand_pose = ckpt["state_dict"]['model.nodes.right.params.pose.weight'].clone().cuda()
    img_fs = out['fnames']

    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)    
    
    # # K = data['object']["K"].to(device).view(1, 4, 4)[:, :3, :3]
    # scale = 1 / data['object']['obj_scale']
    # scale = torch.tensor([scale]).float().to(device)
    mesh_c_o = trimesh.load(object_mesh_f, process=False)

    # ## log object ##
    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)

    # Extract object data
    # pts_w = data['object']["j3d"]          # Shape: (B, number_3d_points, 3)
    # o2w_all = data['object']["o2w_all"]    # Shape: (B, 4, 4)
    o2c_mat = np.load(f"{mvs_root}/o2w_normalized_aligned.npy")
    c2o_mat = np.linalg.inv(o2c_mat) # camera to object
    obj_3d_o = trimesh.load(f"{mvs_root}/sparse_points_normalized_aligned.ply", process=False).vertices
    obj_3d_o = np.tile(obj_3d_o, (o2c_mat.shape[0], 1, 1))
    # img_fs = data['object']["im_paths"]
    # intrinsic = data['object']["K"]        # Camera intrinsics
    # obj_scale = data['object']['obj_scale']

    ###### log hand ########
    # Initialize MANO model
    f3d_r = initialize_mano_model(MANO_f)

    # betas = data['right']["hand_beta"]
    # betas = torch.tensor(np.tile(betas, (o2c_mat.shape[0], 1))).cuda()
    # h2c_rot = torch.tensor(data['right']["hand_rot"]).cuda()
    # h2c_transl = torch.tensor(data['right']["hand_transl"]).cuda()
    # hand_pose = torch.tensor(data['right']["hand_pose"]).cuda()
    

    # Extract and transform right hand vertices
    mano_layer = smplx.create(
            model_path=MANO_f, model_type="mano", use_pca=False, is_rhand=True
    )
    mano_layer.to(torch.device("cuda"))
    ### Note: Hand coordinates to object coordinates only involves translation, not rotation
    # hand vertices in canonical coordinates
    # breakpoint()
    hand_out_can = mano_layer(
        betas=betas,
        hand_pose=hand_pose,
        transl=torch.zeros_like(h2c_transl),
        global_orient=h2c_rot,
    )
    hand_v_can = hand_out_can.vertices.cpu().numpy()
    hand_jnts_can = hand_out_can.joints.cpu().numpy()
    h2c_mat = np.tile(np.eye(4), (o2c_mat.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = h2c_transl.cpu().numpy()
    h2c_mat = h2c_mat / obj_scale # scale the hand
    h2c_mat[:, 3, 3] = 1
    h2o_mat = c2o_mat @ h2c_mat
    # hand vertices in camera coordinates 
    # hand vertices in object coordinates
    hand_v_o = transform_points(hand_v_can, h2o_mat)
    hand_jnts_o = transform_points(hand_jnts_can, h2o_mat)


    hand_v_o *= obj_scale
    hand_jnts_o *= obj_scale

    object_v_o = mesh_c_o.vertices * obj_scale



    out = xdict()
    out['verts.right'] = hand_v_o
    out['jnts.right'] = hand_jnts_o
    out['root.right'] = hand_jnts_o[:,0,:]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :]
    out['verts.object'] = np.tile(np.array(object_v_o), (o2c_mat.shape[0], 1, 1))
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]

    faces = {
        'object' : np.array(mesh_c_o.faces),        
        'right' : np.array(f3d_r),

    }
    out["faces"] = faces
        
    print("Done loading data")
    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)

    out_dict = xdict()
    out_dict["fnames"] = img_fs#.tolist()

    out_dict["K"] = intrinsic[None]
    out_dict["full_seq_name"] = seq_name    
    out_dict.merge(out)
    return out_dict

def load_data_step(
        seq_name = "",
        data_root = "", 
        mvs_root = "", 
        object_mesh_f = "",
        ckpt_p = "",
        # object_mesh_f = "/home/simba/Documents/project/diff_object/threestudio/outputs/hold_MC1_ho3d.80/3d_ref/save/it8500-export/model.obj",
        MANO_f = "",
        common_fate = False,
        ):
    
    device = "cuda:0"
    print("Loading data")
    
    # from src.utils.io.optim import load_data
    # ckpt_p='logs/hold_GPMF14_ho3d_pre_train/checkpoints/last.pose_ref'
    # ckpt_p='logs/' + seq_name.split('.')[0] + '_pre_train/checkpoints/last.pose_ref'
    # "data/hold_MC1_ho3d/processed//colmap_hold_MC1_ho3d.80/sfm_superpoint+superglue/mvs/o2w_normalized_aligned.npy",
    # out, ckpt = load_data(ckpt_p, pose_p=f"{mvs_root}/o2w_normalized_aligned.npy", object_mesh_f=object_mesh_f)
    # breakpoint()
    vis_p = f"{data_root}/hold_fit.aligned.npy"
    data = np.load(vis_p, allow_pickle=True).item()
    intrinsic = data['object']["K"]        # Camera intrinsics
    img_fs = data['object']["im_paths"]

    ckpt = torch.load(ckpt_p, map_location='cpu')
    sd = xdict(ckpt["state_dict"])
    # param_dict = sd.search(".params.")
    hand_scale = np.array(list(sd.search(".hand_scale").values())[0]) 
    betas = sd['models.right.hand_beta'].clone().cuda()
    betas = torch.tile(betas, (len(img_fs), 1))
    h2c_rot = sd["models.right.hand_rot"].clone().cuda()
    h2c_transl = sd["models.right.hand_transl"].clone().cuda()
    hand_pose = sd["models.right.hand_pose"].clone().cuda()


    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)    
    
    # # K = data['object']["K"].to(device).view(1, 4, 4)[:, :3, :3]
    # scale = 1 / data['object']['obj_scale']
    # scale = torch.tensor([scale]).float().to(device)
    mesh_c_o = trimesh.load(object_mesh_f, process=False)

    # ## log object ##
    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    # data = load_data(vis_p)

    # Extract object data
    # pts_w = data['object']["j3d"]          # Shape: (B, number_3d_points, 3)
    # o2w_all = data['object']["o2w_all"]    # Shape: (B, 4, 4)
    o2c_mat = np.load(f"{mvs_root}/o2w_normalized_aligned.npy")
    c2o_mat = np.linalg.inv(o2c_mat) # camera to object
    # obj_3d_o = trimesh.load(f"{mvs_root}/sparse_points_normalized_aligned.ply", process=False).vertices
    # obj_3d_o = np.tile(obj_3d_o, (o2c_mat.shape[0], 1, 1))
    # img_fs = data['object']["im_paths"]
    # intrinsic = data['object']["K"]        # Camera intrinsics
    # obj_scale = data['object']['obj_scale']

    ###### log hand ########
    # Initialize MANO model
    f3d_r = initialize_mano_model(MANO_f)

    # betas = data['right']["hand_beta"]
    # betas = torch.tensor(np.tile(betas, (o2c_mat.shape[0], 1))).cuda()
    # h2c_rot = torch.tensor(data['right']["hand_rot"]).cuda()
    # h2c_transl = torch.tensor(data['right']["hand_transl"]).cuda()
    # hand_pose = torch.tensor(data['right']["hand_pose"]).cuda()
    

    # Extract and transform right hand vertices
    mano_layer = smplx.create(
            model_path=MANO_f, model_type="mano", use_pca=False, is_rhand=True
    )
    mano_layer.to(torch.device("cuda"))
    ### Note: Hand coordinates to object coordinates only involves translation, not rotation
    # hand vertices in canonical coordinates
    # breakpoint()
    hand_out_can = mano_layer(
        betas=betas,
        hand_pose=hand_pose,
        transl=torch.zeros_like(h2c_transl),
        global_orient=h2c_rot,
    )
    hand_v_can = hand_out_can.vertices.cpu().numpy()
    hand_jnts_can = hand_out_can.joints.cpu().numpy()
    h2c_mat = np.tile(np.eye(4), (o2c_mat.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = h2c_transl.cpu().numpy()
    h2c_mat = h2c_mat * hand_scale # scale the hand
    h2c_mat[:, 3, 3] = 1
    h2o_mat = c2o_mat @ h2c_mat
    # hand vertices in camera coordinates 
    # hand vertices in object coordinates
    hand_v_o = transform_points(hand_v_can, h2o_mat)
    hand_jnts_o = transform_points(hand_jnts_can, h2o_mat)


    hand_v_c = transform_points(hand_v_o, o2c_mat)
    hand_jnts_c = transform_points(hand_jnts_o, o2c_mat)

    object_v_c = transform_points( mesh_c_o.vertices, o2c_mat)

    hand_v_c /= hand_scale
    hand_jnts_c /= hand_scale    

    object_v_c = object_v_c / hand_scale



    out = xdict()
    out['verts.right'] = hand_v_c
    out['jnts.right'] = hand_jnts_c
    out['root.right'] = hand_jnts_c[:,0,:]
    out['j3d_ra.right'] = hand_jnts_can - hand_jnts_can[:,0:1, :]
    out['verts.object'] = object_v_c
    out['v3d_c.object'] = out['verts.object']
    out['root.object'] = compute_bounding_box_centers(out['verts.object'])
    out['v3d_ra.object'] = out['verts.object'] - out['root.object'][:,None,:]
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]

    faces = {
        'object' : np.array(mesh_c_o.faces),        
        'right' : np.array(f3d_r),

    }
    out["faces"] = faces
        
    print("Done loading data")
    out = out.to_torch()
    out['verts.right'].float().to(device)
    out['jnts.right'].float().to(device)
    out['verts.object'].float().to(device)

    out_dict = xdict()
    out_dict["fnames"] = img_fs#.tolist()

    out_dict["K"] = intrinsic[None]
    out_dict["full_seq_name"] = seq_name    
    out_dict.merge(out)
    return out_dict


def load_data(sd_p):
    device = "cuda:0"
    print("Loading data")
    data = torch.load(sd_p, map_location="cpu")
    sd = xdict(data["state_dict"])
    misc_ps = sorted(glob(op.join("logs", sd_p.split("/")[1], "misc", "*")))
    misc = np.load(misc_ps[-1], allow_pickle=True).item()

    fnames = misc["img_paths"]
    K = torch.FloatTensor(misc["K"]).to(device).view(1, 4, 4)[:, :3, :3]
    scale = misc["scale"]
    scale = torch.tensor([scale]).float().to(device)
    mesh_c_o = misc["mesh_c_o"] if "mesh_c_o" in misc else misc["object_cano"]

    node_ids = []
    for key in sd.keys():
        if ".nodes." not in key:
            continue
        node_id = key.split(".")[2]
        node_ids.append(node_id)
    node_ids = list(set(node_ids))

    params = {}
    for node_id in node_ids:
        params[node_id] = sd.search(".params.").search(node_id)
        params[node_id]["scene_scale"] = scale
        params[node_id] = params[node_id].to(device)

    scale_key = "model.nodes.object.server.object_model.obj_scale"
    obj_scale = sd[scale_key] if scale_key in sd.keys() else None

    seq_name = fnames[0].split("/")[2]

    servers = {}
    faces = {}
    for node_id in node_ids:
        if "right" in node_id or "left" in node_id:
            hand = "right" if "right" in node_id else "left"
            is_right = hand == "right"
            server = MANOServer(betas=None, is_rhand=is_right).to(device)
            myfaces = torch.LongTensor(server.faces.astype(np.int64)).to(device)
        elif "object" in node_id:
            server = ObjectServer(seq_name, template=mesh_c_o)
            server.object_model.obj_scale = obj_scale
            server.to(device)
            myfaces = torch.LongTensor(mesh_c_o.faces).to(device)
        else:
            assert False, f"Unknown node id: {node_id}"

        servers[node_id] = server
        faces[node_id] = myfaces

    if obj_scale is not None:
        servers["object"].object_model.obj_scale = obj_scale.to(device)

    out = xdict()
    for node_id in node_ids:
        out.merge(
            xdict(servers[node_id].forward_param(params[node_id])).postfix(
                f".{node_id}"
            )
        )

    # mapping to evaluation camera coordinate
    def map_deform2eval_batch(verts, inverse_scale):
        return np.array(
            [
                map_deform2eval(verts, inverse_scale)
                for verts in verts.cpu().detach().numpy()
            ]
        )

    # map predictions to evaluation space
    inverse_scale = float(1.0 / scale[0])
    for key, val in out.search("verts.").items():
        out[key.replace("verts.", "v3d_c.")] = map_deform2eval_batch(val, inverse_scale)
    for key, val in out.search("jnts.").items():
        out[key.replace("jnts.", "j3d_c.")] = map_deform2eval_batch(val, inverse_scale)

    # hand root relative
    for key, val in out.search("j3d_c.").items():
        # root
        out[key.replace("j3d_c.", "root.")] = val[:, :1].squeeze(1)
        # root relative
        out[key.replace("j3d_c.", "j3d_ra.")] = val - val[:, :1]
    out["root.object"] = compute_bounding_box_centers(out["v3d_c.object"])
    out["v3d_ra.object"] = out["v3d_c.object"] - out["root.object"][:, None, :]

    # object: relative to right hand
    out["v3d_right.object"] = out["v3d_c.object"] - out["root.right"][:, None, :]
    if "root.left" in out.keys():
        out["v3d_left.object"] = out["v3d_c.object"] - out["root.left"][:, None, :]
    out_dict = xdict()
    out_dict["fnames"] = fnames
    out_dict.merge(out)
    out_dict["faces"] = faces

    out_dict["servers"] = servers
    out_dict["K"] = K.cpu().numpy()
    out_dict["full_seq_name"] = fnames[0].split("/")[2]

    insta_p = sd_p + ".insta_map.npy"
    if op.exists(insta_p):
        insta_map = torch.FloatTensor(np.load(sd_p + ".insta_map.npy"))
        out_dict["insta_map"] = insta_map

    print("Done loading data")
    out_dict = out_dict.to_torch()
    return out_dict


def load_viewer_data(args):
    data = load_data(args.ckpt_p)
    faces = xdict(data["faces"]).to_np()
    K = data["K"].numpy().reshape(3, 3)
    fnames = data["fnames"]

    color_dict = {"right": "gray_white", "left": "white", "object": "green-blue"}
    vis_dict = {}
    pred = data.search("v3d_c.").to_np()
    for v3d_key in pred.keys():
        node_id = v3d_key.split(".")[1]
        vis_dict[node_id] = {
            "v3d": pred[v3d_key],
            "f3d": faces[node_id],
            "vc": None,
            "name": node_id,
            "color": color_dict[node_id],
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

def load_viewer_data_magicHOI(obj_mesh_f, ckpt_p, data_root, obj_pose_f, is_right=None):
    pred = {}
    faces = {}
    # vis_p = f"{data_root}/hold_fit.aligned.npy"
    vis_p = f"{op.dirname(ckpt_p)}/../../hold_fit.aligned.npy"
    data =  np.load(vis_p, allow_pickle=True).item()
    fnames = data['object']["im_paths"]

    K = data['object']["K"]        # Camera intrinsics

    ckpt = torch.load(ckpt_p, map_location='cpu')

    sd = xdict(ckpt["state_dict"])

    o2c_mat = np.load(obj_pose_f)  
    c2o_mat = np.linalg.inv(o2c_mat) # camera to object
    obj_mesh = trimesh.load(obj_mesh_f, process=False)
    obj_3d_o = obj_mesh.vertices
    obj_3d_o = np.tile(obj_3d_o, (o2c_mat.shape[0], 1, 1))
    obj_3d_c = transform_points(obj_3d_o, o2c_mat)

    hand_scale = np.array(list(sd.search(".hand_scale").values())[0])

    obj_3d_c = obj_3d_c / hand_scale
    pred["v3d_c.object"] = obj_3d_c
    faces["object"] = np.array(obj_mesh.faces)

    if is_right is None:
        hand_flag = "right"
    else:
        hand_flag = "left" if not is_right else "right"

    betas = torch.tensor(sd[f'models.{hand_flag}.hand_beta']).cuda()
    betas = torch.tile(betas, (len(fnames), 1))
    h2c_rot = torch.tensor(sd[f"models.{hand_flag}.hand_rot"]).cuda()
    h2c_transl = torch.tensor(sd[f"models.{hand_flag}.hand_transl"]).cuda()
    hand_pose = torch.tensor(sd[f"models.{hand_flag}.hand_pose"]).cuda()

    ###### log hand ########
    # Initialize MANO model
    if hand_flag == "right":
        MANO_dir = f"./code/body_models/MANO_RIGHT.pkl"    
    else:
        MANO_dir = f"./code/body_models/MANO_LEFT.pkl"
    f3d_r = initialize_mano_model(MANO_dir)
    # Extract and transform right hand vertices
    
    mano_layer = smplx.create(
            model_path=MANO_dir, model_type="mano", use_pca=False, is_rhand=is_right
    )
    mano_layer.to(torch.device("cuda"))
    ### Note: Hand coordinates to object coordinates only involves translation, not rotation
    # hand vertices in canonical coordinates

    output = mano_layer(betas=betas, 
                            hand_pose=hand_pose, 
                            transl=torch.zeros_like(h2c_transl), 
                            global_orient=h2c_rot,
                            )
    hand_v_can = output.vertices.cpu().numpy()
    h2c_mat = np.tile(np.eye(4), (o2c_mat.shape[0], 1, 1))
    h2c_mat[:, :3, 3] = h2c_transl.cpu().numpy()
    h2c_mat = h2c_mat * hand_scale # scale the hand
    h2c_mat[:, 3, 3] = 1
    h2o_mat = c2o_mat @ h2c_mat
    # hand vertices in camera coordinates
    hand_v_c = transform_points(hand_v_can, h2c_mat)
    hand_v_c /= hand_scale    


    pred[f"v3d_c.{hand_flag}"] = hand_v_c
    faces[f"{hand_flag}"] = np.array(f3d_r)



    color_dict = {"right": "gray_white", "left": "gray_white", "object": "green-blue"}
    vis_dict = {}
    # breakpoint()
    for v3d_key in pred.keys():
        node_id = v3d_key.split(".")[1]
        if node_id in ["right", "left"]:
            v3d_sealed, faces_sealed = seal_mano_mesh(
                torch.tensor(pred[v3d_key]), torch.tensor(faces[node_id].astype(np.int32)), is_rhand=(node_id == hand_flag)
            )
            v3d_sealed = v3d_sealed.numpy()
            faces_sealed = faces_sealed.numpy()
        else:
            v3d_sealed = pred[v3d_key]
            faces_sealed = faces[node_id]

        vis_dict[node_id] = {
            "v3d": v3d_sealed,
            "f3d": faces_sealed,
            "vc": None,
            "name": node_id,
            "color": color_dict[node_id],
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

def load_viewer_data_EasyHOI(data_dir, hand_data_f, is_tripo):
    pred = {}
    faces = {}

    hand_data =  torch.load(hand_data_f)
    name = hand_data["name"]
    fnames = [hand_data['img_path']]
    projection = hand_data['cam_projection']
    image = Image.open(fnames[0])
    width, height = image.size
    # Convert projection matrix to camera intrinsics K
    # Projection is a 4x4 matrix, we need to extract the 3x3 intrinsics
    K = get_intrinsic_matrix_from_projection(projection, width, height)
    
    mano_params = hand_data['mano_params']
    hand_scale = hand_data['hand_scale'].detach()
    hand_transl = hand_data['hand_transl'].detach()
    global_rot = hand_data['hand_rot'].detach()
    fullpose = mano_params['fullpose']
    betas = mano_params['betas']
    from manotorch.manolayer import ManoLayer
    mano_layer = ManoLayer(side="right")
    mano_layer.eval()
    hand_faces = mano_layer.get_mano_closed_faces()
    if not hand_data["is_right"]:
        hand_faces = hand_faces[:,[0,2,1]] # faces for left hand        
    mano_output = mano_layer(fullpose, betas)
    hand_verts_can = mano_output.verts

    hand_verts_can[:,:,0] = (2*hand_data["is_right"]-1)*hand_verts_can[:,:,0]
    hand_verts_can = hand_verts_can.squeeze()
    hand_verts = hand_verts_can

    hand_verts = hand_verts @ global_rot.mT
    hand_verts = hand_verts * hand_scale
    hand_verts = hand_verts + hand_transl

    hand_verts /= hand_scale

    pred["v3d_c.right"] = hand_verts[None].numpy()
    faces["right"] = hand_faces.numpy()


    if is_tripo:
        obj_mesh_dir = op.join(data_dir, "obj_recon/results/tripo/meshes/")
    else:
        obj_mesh_dir = op.join(data_dir, "obj_recon/results/instantmesh/instant-mesh-large/meshes/")
    

    obj_mesh = trimesh.load(op.join(obj_mesh_dir, str(name), "full.obj"))
    obj_verts = obj_mesh.vertices
    obj_faces = obj_mesh.faces

    if is_tripo:
        rot1 = np.array([[1,0,0],
                            [0,0,-1],
                            [0,1,0]])
        
        obj_verts = obj_verts @ rot1.T
    else:
        rot1 = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,-1]])
        obj_verts = obj_verts @ rot1.T
        obj_faces = np.array(obj_faces[:, ::-1])        

    obj_verts = (obj_verts / hand_scale)    

    pred["v3d_c.object"] = obj_verts[None].numpy()
    faces["object"] = obj_faces

    # Save object mesh
    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
    obj_mesh.export("object_mesh_1.obj")

    # Save hand mesh
    hand_mesh = trimesh.Trimesh(vertices=hand_verts.cpu().numpy(), faces=hand_faces.cpu().numpy())
    hand_mesh.export("hand_mesh_1.obj")

    color_dict = {"right": "gray_white", "left": "white", "object": "green-blue"}
    vis_dict = {}
    # breakpoint()
    for v3d_key in pred.keys():
        node_id = v3d_key.split(".")[1]
        if node_id in ["right", "left"]:
            v3d_sealed, faces_sealed = seal_mano_mesh(
                torch.tensor(pred[v3d_key]), torch.tensor(faces[node_id].astype(np.int32)), is_rhand=(node_id == "right")
            )
            v3d_sealed = v3d_sealed.numpy()
            faces_sealed = faces_sealed.numpy()
        else:
            v3d_sealed = pred[v3d_key]
            faces_sealed = faces[node_id]

        vis_dict[node_id] = {
            "v3d": v3d_sealed,
            "f3d": faces_sealed,
            "vc": None,
            "name": node_id,
            "color": color_dict[node_id],
        } 
    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )
    num_frames = len([fnames])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    im = Image.open(fnames[0])
    cols, rows = im.size

    images = [Image.open(im_p) for im_p in fnames]
    data = ViewerData(Rt, K, cols, rows, images)
    return meshes, data
