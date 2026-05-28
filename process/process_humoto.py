"""
Convert HUMOTO (Mixamo-rigged) sequences into InterAct's canonical SMPL-X format.

Pipeline per sequence:
  1. Load the up_bone humoto pickle, run Mixamo forward kinematics → joint positions.
  2. Convert z-up → y-up.
  3. Fit SMPL-X betas + per-frame (global_orient, body_pose, trans) to the Mixamo
     joint positions (joint-position objective; betas frozen after frame-0 fit).
  4. Write human.npz, object.npz, and the primary object's sample_points.npy + .obj.

Run as a script to batch-process every entry in SEQUENCE_CONFIG:
    python -m process.process_humoto
"""
import os
import sys
import pickle

import numpy as np
import torch
import torch.optim as optim
import trimesh
import smplx
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

# Portable paths, relative to the InterAct project root; resolve via symlinks you
# create (override with env vars if your layout differs). No machine-specific paths.
#
# NOTE: the HUMOTO *code* repo and the Adobe *data* release are both literally named
# "humoto" — keep them in separate trees to avoid confusion:
#   ./humoto               -> HUMOTO *code* repo    (contains human_model/, scripts/, ...)
#   ./data/humoto/raw      -> HUMOTO *data* release  (<seq>/<seq>.glb, .png)  [= humoto/humoto]
#   ./data/humoto/upbone   -> extracted up_bone pickles  (<seq>/<seq>.pkl)
#   ./models               -> SMPL-X / SMPL-H model files   [=inside InterAct repo, not humoto]
# Outputs are written under ./data/humoto/{sequences_canonical,objects}/
#
# Symlink commands (run once from the project root, point at YOUR locations):
#   ln -s /path/to/humoto              ./humoto                 [humoto code repo]  
#   ln -s /path/to/humoto/humoto       ./data/humoto/raw        [humoto data repo]
#   ln -s /path/to/humoto_upbone       ./data/humoto/upbone     [extracted up_bone pickles]
HUMOTO_REPO   = os.environ.get('HUMOTO_REPO',     './humoto')
HUMOTO_JSON   = os.path.join(HUMOTO_REPO, 'human_model', 'human_model_up_bone_zup.json')
HUMOTO_RAW    = os.environ.get('HUMOTO_RAW',      './data/humoto/raw')
HUMOTO_UPBONE = os.environ.get('HUMOTO_UPBONE',   './data/humoto/upbone')
MODEL_PATH    = os.environ.get('INTERACT_MODELS', './models')
OUTPUT_ROOT   = os.environ.get('HUMOTO_OUTPUT',   './data/humoto')

# humoto's differentiable Mixamo model lives in its repo
sys.path.insert(0, HUMOTO_REPO)
from human_model.human_model import HumanModelDifferentiable  # noqa: E402

# ─── Constants ──────────────────────────────────────────────────────────────
# z-up → y-up: (x, y, z) -> (x, z, -y)
Z_TO_Y_MAT = np.array([[1, 0, 0],
                       [0, 0, 1],
                       [0, -1, 0]], dtype=np.float32)
# Undoes the axis-convention rotation Blender bakes into GLB object meshes
R_XFLIP_MAT = Rot.from_euler('x', 90, degrees=True).as_matrix().astype(np.float32)

# SMPL-X body joint index → (name, Mixamo bone). Body only; hands/face zeroed for v1.
BODY_CORR = [
    (0,  'pelvis',         'mixamorig:Hips'),
    (1,  'left_hip',       'mixamorig:LeftUpLeg'),
    (2,  'right_hip',      'mixamorig:RightUpLeg'),
    (3,  'spine1',         'mixamorig:Spine'),
    (4,  'left_knee',      'mixamorig:LeftLeg'),
    (5,  'right_knee',     'mixamorig:RightLeg'),
    (6,  'spine2',         'mixamorig:Spine1'),
    (7,  'left_ankle',     'mixamorig:LeftFoot'),
    (8,  'right_ankle',    'mixamorig:RightFoot'),
    (9,  'spine3',         'mixamorig:Spine2'),
    (10, 'left_foot',      'mixamorig:LeftToeBase'),
    (11, 'right_foot',     'mixamorig:RightToeBase'),
    (12, 'neck',           'mixamorig:Neck'),
    (13, 'left_collar',    'mixamorig:LeftShoulder'),
    (14, 'right_collar',   'mixamorig:RightShoulder'),
    (15, 'head',           'mixamorig:Head'),
    (16, 'left_shoulder',  'mixamorig:LeftArm'),
    (17, 'right_shoulder', 'mixamorig:RightArm'),
    (18, 'left_elbow',     'mixamorig:LeftForeArm'),
    (19, 'right_elbow',    'mixamorig:RightForeArm'),
    (20, 'left_wrist',     'mixamorig:LeftHand'),
    (21, 'right_wrist',    'mixamorig:RightHand'),
]
# SMPL-X body kinematic tree (parent index per joint; -1 = root). Useful for plots.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
BODY_NAMES = [c[1] for c in BODY_CORR]

# Q1(a): hardcoded primary-object choice per sequence (inspect PNG thumbnails to fill in)
SEQUENCE_CONFIG = {
    'baking_with_spatula_mixing_bowl_and_scooping_to_tray-244': 'spatula',
}


def load_models(model_path=MODEL_PATH, humoto_json=HUMOTO_JSON, device='cpu'):
    """Build the SMPL-X model (smplx, 16 betas) and the Mixamo FK model."""
    smplx_model = smplx.create(
        model_path, model_type='smplx', gender='neutral',
        use_pca=False, flat_hand_mean=True, num_betas=16,
    )
    mx_model = HumanModelDifferentiable(humoto_json, device=device)
    return smplx_model, mx_model


def _frame_to_pose_params(frame_dict):
    """humoto pickle frame {bone: [w,x,y,z,lx,ly,lz]} → {bone: [1,4,4]} for FK."""
    pp = {}
    for bone, entry in frame_dict.items():
        e = np.asarray(entry, dtype=np.float32)
        w, x, y, z = e[:4]
        lx, ly, lz = e[4:]
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = Rot.from_quat([x, y, z, w]).as_matrix()
        M[:3, 3] = [lx, ly, lz]
        pp[bone] = torch.from_numpy(M).unsqueeze(0)
    return pp


def compute_targets(arm, mx_model, body_corr=BODY_CORR):
    """Mixamo armature frames → SMPL-X-indexed joint targets (T, 22, 3) in y-up."""
    T = len(arm)
    targets = np.zeros((T, 22, 3), dtype=np.float32)
    bone_index = {b: i for i, b in enumerate(mx_model.bone_names)}
    for t in range(T):
        with torch.no_grad():
            _, jp = mx_model(_frame_to_pose_params(arm[t]))
        jm_zup = np.stack([jp[b].numpy()[0] for b in mx_model.bone_names])
        jm_yup = jm_zup @ Z_TO_Y_MAT.T
        for smplx_idx, _, mx_name in body_corr:
            targets[t, smplx_idx] = jm_yup[bone_index[mx_name]]
    return targets


def load_targets(pkl_path, mx_model, body_corr=BODY_CORR):
    """Load a humoto pickle → (targets (T,22,3) y-up SMPL-X joint targets, objs dict).

    Convenience wrapper so callers (and the inspection notebook) can get the
    Mixamo joint targets without re-running the whole pipeline.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    arm, objs = data['armature'], data['objects']
    targets = compute_targets(arm, mx_model, body_corr)
    return targets, objs


def fit_sequence(targets, smplx_model, n_steps_beta=600, n_steps_pose=80):
    """Joint-position fit. Returns (betas, global_orient, body_pose, trans, errors)."""
    print(f"\n=== Fitting sequence (T={targets.shape[0]}) ===")
    
    T = targets.shape[0]

    # ── frame-0: β + pose + trans jointly ──
    target_0 = torch.from_numpy(targets[0])
    pelvis_neutral = smplx_model(betas=torch.zeros(1, 16)).joints[0, 0].detach()
    trans_init = (target_0[0] - pelvis_neutral)

    betas = torch.zeros(1, 16, requires_grad=True)
    go = torch.zeros(1, 3, requires_grad=True)
    bp = torch.zeros(1, 63, requires_grad=True)
    tr = trans_init.unsqueeze(0).clone().requires_grad_(True)
    opt = optim.Adam([betas, go, bp, tr], lr=0.05)

    pbar = tqdm(range(n_steps_beta), desc='beta+pose fit (frame 0)')
    for step in pbar:
        pred = smplx_model(global_orient=go, body_pose=bp, transl=tr, betas=betas).joints[0, :22]
        loss = (((pred - target_0) ** 2).mean()
                + 0.001 * (betas ** 2).mean()
                + 0.0005 * (bp ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 60 == 0 or step == n_steps_beta - 1:
            joint_err = (pred - target_0).norm(dim=1)
            pbar.set_postfix(loss=f'{loss.item():.5f}',
                             mean=f'{joint_err.mean().item()*100:.1f}cm',
                             max=f'{joint_err.max().item()*100:.1f}cm',
                             beta=f'{betas.norm().item():.2f}')

    betas_fit = betas.detach().clone()
    fitted_go = np.zeros((T, 3), dtype=np.float32)
    fitted_bp = np.zeros((T, 63), dtype=np.float32)
    fitted_tr = np.zeros((T, 3), dtype=np.float32)

    errors = np.zeros(T, dtype=np.float32)
    fitted_go[0] = go.detach()[0].numpy()
    fitted_bp[0] = bp.detach()[0].numpy()
    fitted_tr[0] = tr.detach()[0].numpy()

    # ── per-frame pose fit, warm-started, β frozen ──
    for t in tqdm(range(1, T), desc='per-frame pose fit'):
        go_v = torch.from_numpy(fitted_go[t - 1:t]).clone().requires_grad_(True)
        bp_v = torch.from_numpy(fitted_bp[t - 1:t]).clone().requires_grad_(True)
        tr_v = torch.from_numpy(fitted_tr[t - 1:t]).clone().requires_grad_(True)
        target_t = torch.from_numpy(targets[t])
        bp_prev = torch.from_numpy(fitted_bp[t - 1:t])

        opt = optim.Adam([go_v, bp_v, tr_v], lr=0.03)
        for _ in range(n_steps_pose):
            pred = smplx_model(global_orient=go_v, body_pose=bp_v, transl=tr_v, betas=betas_fit).joints[0, :22]
            loss = (((pred - target_t) ** 2).mean()
                    + 0.0005 * (bp_v ** 2).mean()
                    + 0.01 * ((bp_v - bp_prev) ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()

        fitted_go[t] = go_v.detach()[0].numpy()
        fitted_bp[t] = bp_v.detach()[0].numpy()
        fitted_tr[t] = tr_v.detach()[0].numpy()
        with torch.no_grad():
            errors[t] = (pred - target_t).norm(dim=1).mean().item()

    return betas_fit, fitted_go, fitted_bp, fitted_tr, errors


def build_object_npz(objs, primary_object):
    """humoto object trajectory (z-up) → (angles, trans) in y-up axis-angle."""
    obj_zup = np.array(objs[primary_object], dtype=np.float32)  # (T, 7) [w,x,y,z,lx,ly,lz]
    trans_yup = obj_zup[:, 4:] @ Z_TO_Y_MAT.T
    quat_xyzw = obj_zup[:, [1, 2, 3, 0]]
    rot_yup = Rot.from_matrix(Z_TO_Y_MAT.astype(np.float64)) * Rot.from_quat(quat_xyzw)
    angles = rot_yup.as_rotvec().astype(np.float32)
    return angles, trans_yup.astype(np.float32)


def build_object_mesh(glb_path, primary_object, n_sample=340, seed=0):
    """Extract object mesh from GLB, undo the GLB axis flip, sample N surface points."""
    scene = trimesh.load(glb_path)
    mesh = scene.geometry[primary_object]
    oriented = trimesh.Trimesh(vertices=mesh.vertices @ R_XFLIP_MAT.T, faces=mesh.faces)
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(oriented, n_sample)
    return oriented, pts.astype(np.float32)


def process_humoto_sequence(pkl_path, glb_path, seq_name, primary_object,
                            smplx_model, mx_model, output_root=OUTPUT_ROOT,
                            body_corr=BODY_CORR, seed=0, verbose=True):
    """End-to-end conversion of ONE humoto sequence. Returns a metrics dict."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    targets, objs = load_targets(pkl_path, mx_model, body_corr)
    if primary_object not in objs:
        raise ValueError(f"'{primary_object}' not in pickle objects: {list(objs.keys())}")
    T = len(targets)
    if verbose:
        print(f"[1/4] {seq_name}: T={T}, objects={list(objs.keys())}")
    betas_fit, fitted_go, fitted_bp, fitted_tr, errors = fit_sequence(targets, smplx_model)
    if verbose:
        print(f"[2/4] fit: mean={errors.mean() * 100:.2f}cm  max={errors.max() * 100:.2f}cm  "
              f"||beta||={betas_fit.norm():.2f}")

    out_seq_dir = os.path.join(output_root, 'sequences_canonical', seq_name)
    out_obj_dir = os.path.join(output_root, 'objects', primary_object)
    os.makedirs(out_seq_dir, exist_ok=True)
    os.makedirs(out_obj_dir, exist_ok=True)

    poses = np.concatenate([fitted_go, fitted_bp, np.zeros((T, 90), dtype=np.float32)], axis=1)
    np.savez(os.path.join(out_seq_dir, 'human.npz'),
             poses=poses.astype(np.float32),
             betas=betas_fit[0].numpy().astype(np.float32),
             trans=fitted_tr.astype(np.float32),
             gender='neutral')

    obj_angles, obj_trans = build_object_npz(objs, primary_object)
    np.savez(os.path.join(out_seq_dir, 'object.npz'),
             angles=obj_angles, trans=obj_trans, name=primary_object)
    if verbose:
        print(f"[3/4] wrote human.npz + object.npz → {out_seq_dir}")

    oriented_mesh, sample_pts = build_object_mesh(glb_path, primary_object, seed=seed)
    np.save(os.path.join(out_obj_dir, 'sample_points.npy'), sample_pts)
    oriented_mesh.export(os.path.join(out_obj_dir, f'{primary_object}.obj'))
    if verbose:
        print(f"[4/4] wrote {primary_object} mesh + sample_points → {out_obj_dir}")

    return {
        'seq_dir': out_seq_dir,
        'obj_dir': out_obj_dir,
        'betas': betas_fit[0].numpy(),
        'mean_err_cm': float(errors.mean() * 100),
        'max_err_cm': float(errors.max() * 100),
        'T': T,
    }


if __name__ == '__main__':
    smplx_model, mx_model = load_models()
    for seq_name, primary_object in SEQUENCE_CONFIG.items():
        pkl_path = os.path.join(HUMOTO_UPBONE, seq_name, f'{seq_name}.pkl')
        glb_path = os.path.join(HUMOTO_RAW, seq_name, f'{seq_name}.glb')
        result = process_humoto_sequence(
            pkl_path=pkl_path, glb_path=glb_path,
            seq_name=seq_name, primary_object=primary_object,
            smplx_model=smplx_model, mx_model=mx_model,
        )
        print(result)
