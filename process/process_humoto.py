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
# Outputs are written under ./data/humoto/{sequences_seg,objects}/
# (sequences_seg = pre-canonical staging; run canonicalize_human.py to produce
#  sequences_canonical with forward-facing alignment + floor normalization)
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
# Text source: animations.json from the HUMOTO release (has per-clip short_script).
#   ln -s /path/to/humoto_adobe/animations.json ./data/humoto/animations.json
HUMOTO_ANIMATIONS = os.environ.get('HUMOTO_ANIMATIONS', './data/humoto/animations.json')

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

# SMPL-X body joint index → (name, Mixamo bone). Face joints stay zeroed; hands are
# fit separately (see HAND_CORR).
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

# SMPL-X hand joint index → (name, Mixamo bone). 4 targets per finger: the three
# articulated joints (idx 25-54) plus the fingertip landmark (idx 66-75, regressed
# from the mesh) which pins down the distal joint's orientation. Mixamo finger bones
# are <Hand><Finger>{1,2,3} for the joints and {4} for the tip.
HAND_CORR = [
    (25, 'left_index1',   'mixamorig:LeftHandIndex1'),
    (26, 'left_index2',   'mixamorig:LeftHandIndex2'),
    (27, 'left_index3',   'mixamorig:LeftHandIndex3'),
    (67, 'left_index',    'mixamorig:LeftHandIndex4'),
    (28, 'left_middle1',  'mixamorig:LeftHandMiddle1'),
    (29, 'left_middle2',  'mixamorig:LeftHandMiddle2'),
    (30, 'left_middle3',  'mixamorig:LeftHandMiddle3'),
    (68, 'left_middle',   'mixamorig:LeftHandMiddle4'),
    (31, 'left_pinky1',   'mixamorig:LeftHandPinky1'),
    (32, 'left_pinky2',   'mixamorig:LeftHandPinky2'),
    (33, 'left_pinky3',   'mixamorig:LeftHandPinky3'),
    (70, 'left_pinky',    'mixamorig:LeftHandPinky4'),
    (34, 'left_ring1',    'mixamorig:LeftHandRing1'),
    (35, 'left_ring2',    'mixamorig:LeftHandRing2'),
    (36, 'left_ring3',    'mixamorig:LeftHandRing3'),
    (69, 'left_ring',     'mixamorig:LeftHandRing4'),
    (37, 'left_thumb1',   'mixamorig:LeftHandThumb1'),
    (38, 'left_thumb2',   'mixamorig:LeftHandThumb2'),
    (39, 'left_thumb3',   'mixamorig:LeftHandThumb3'),
    (66, 'left_thumb',    'mixamorig:LeftHandThumb4'),
    (40, 'right_index1',  'mixamorig:RightHandIndex1'),
    (41, 'right_index2',  'mixamorig:RightHandIndex2'),
    (42, 'right_index3',  'mixamorig:RightHandIndex3'),
    (72, 'right_index',   'mixamorig:RightHandIndex4'),
    (43, 'right_middle1', 'mixamorig:RightHandMiddle1'),
    (44, 'right_middle2', 'mixamorig:RightHandMiddle2'),
    (45, 'right_middle3', 'mixamorig:RightHandMiddle3'),
    (73, 'right_middle',  'mixamorig:RightHandMiddle4'),
    (46, 'right_pinky1',  'mixamorig:RightHandPinky1'),
    (47, 'right_pinky2',  'mixamorig:RightHandPinky2'),
    (48, 'right_pinky3',  'mixamorig:RightHandPinky3'),
    (75, 'right_pinky',   'mixamorig:RightHandPinky4'),
    (49, 'right_ring1',   'mixamorig:RightHandRing1'),
    (50, 'right_ring2',   'mixamorig:RightHandRing2'),
    (51, 'right_ring3',   'mixamorig:RightHandRing3'),
    (74, 'right_ring',    'mixamorig:RightHandRing4'),
    (52, 'right_thumb1',  'mixamorig:RightHandThumb1'),
    (53, 'right_thumb2',  'mixamorig:RightHandThumb2'),
    (54, 'right_thumb3',  'mixamorig:RightHandThumb3'),
    (71, 'right_thumb',   'mixamorig:RightHandThumb4'),
]
HAND_SMPLX_IDX = [c[0] for c in HAND_CORR]   # gather indices into the (127,3) joint output
HAND_NAMES = [c[1] for c in HAND_CORR]
# left_hand_pose / right_hand_pose are 45 dims each (15 joints × 3 axis-angle), in
# SMPL-X joint order; poses[:, 66:111] = left, poses[:, 111:156] = right.

# Q1(a): hardcoded primary-object choice per sequence (inspect PNG thumbnails to fill in)
SEQUENCE_CONFIG = {
    'baking_with_spatula_mixing_bowl_and_scooping_to_tray-244':                         'spatula',
    'carry_organizer_with_both_hands_at_chest_height-436':                              'draw_organizer_tray',
    'carry_side_table_with_both_hands_walk_around-536':                                 'side_table',
    'carry_vase_right_hand_transfer_to_left_hand_transfer_to_right_hand_walk_around-942':'vase',
    'carry_wok_turner_right_hand_walk_around-968':                                      'wok_turner',
    'carrying_cutting_board_with_right_hand-213':                                       'cutting_board',
    'carrying_whisk_with_right_hand-037':                                               'whisk',
    'checking_floor_lamp_with_right_hand-024':                                          'floor_lamp',
    'checking_organizer_medium_on_table-289':                                           'organizer_medium',
    'chopping_and_slitting_with_knife-706':                                             'knife',
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


def _mixamo_joints_yup(arm, mx_model):
    """Run Mixamo FK on every frame → (joints (T, n_bones, 3) in y-up, bone_index)."""
    bone_names = list(mx_model.bone_names)
    bone_index = {b: i for i, b in enumerate(bone_names)}
    out = np.zeros((len(arm), len(bone_names), 3), dtype=np.float32)
    for t in range(len(arm)):
        with torch.no_grad():
            _, jp = mx_model(_frame_to_pose_params(arm[t]))
        jm_zup = np.stack([jp[b].numpy()[0] for b in bone_names])
        out[t] = jm_zup @ Z_TO_Y_MAT.T
    return out, bone_index


def compute_targets(arm, mx_model, body_corr=BODY_CORR):
    """Mixamo armature frames → SMPL-X-indexed body joint targets (T, 22, 3) in y-up."""
    jm, bone_index = _mixamo_joints_yup(arm, mx_model)
    targets = np.zeros((len(arm), 22, 3), dtype=np.float32)
    for smplx_idx, _, mx_name in body_corr:
        targets[:, smplx_idx] = jm[:, bone_index[mx_name]]
    return targets


def load_targets(pkl_path, mx_model, body_corr=BODY_CORR):
    """Load a humoto pickle → (body targets (T,22,3) y-up, objs dict).

    Convenience wrapper so callers (and the inspection notebook) can get the
    Mixamo body joint targets without re-running the whole pipeline.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    arm, objs = data['armature'], data['objects']
    return compute_targets(arm, mx_model, body_corr), objs


def load_all_targets(pkl_path, mx_model, body_corr=BODY_CORR, hand_corr=HAND_CORR):
    """Load a humoto pickle, run FK once → (body (T,22,3), hand (T,40,3), objs).

    `hand` columns are aligned to `hand_corr` order (== HAND_SMPLX_IDX gather order).
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    arm, objs = data['armature'], data['objects']
    jm, bone_index = _mixamo_joints_yup(arm, mx_model)
    body = np.zeros((len(arm), 22, 3), dtype=np.float32)
    for smplx_idx, _, mx_name in body_corr:
        body[:, smplx_idx] = jm[:, bone_index[mx_name]]
    hand = np.zeros((len(arm), len(hand_corr), 3), dtype=np.float32)
    for j, (_, _, mx_name) in enumerate(hand_corr):
        hand[:, j] = jm[:, bone_index[mx_name]]
    return body, hand, objs


def _fit_hands_frame(smplx_model, go, bp, tr, betas, hand_target, hand_idx,
                     lh_init, rh_init, n_steps, lr=0.05):
    """Fit left+right hand pose for ONE frame with the body frozen.

    go/bp/tr/betas are detached (1,*) tensors; hand_target is (n_hand, 3);
    hand_idx gathers the hand joints from the (127,3) joint output.
    Returns (lh (1,45), rh (1,45), mean_err_m).
    """
    lh = lh_init.clone().requires_grad_(True)
    rh = rh_init.clone().requires_grad_(True)
    tgt = torch.as_tensor(hand_target)
    opt = optim.Adam([lh, rh], lr=lr)
    for _ in range(n_steps):
        pred = smplx_model(global_orient=go, body_pose=bp, transl=tr, betas=betas,
                           left_hand_pose=lh, right_hand_pose=rh).joints[0]
        loss = (((pred[hand_idx] - tgt) ** 2).mean()
                + 1e-4 * (lh ** 2).mean() + 1e-4 * (rh ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = smplx_model(global_orient=go, body_pose=bp, transl=tr, betas=betas,
                           left_hand_pose=lh, right_hand_pose=rh).joints[0]
        err = (pred[hand_idx] - tgt).norm(dim=1).mean().item()
    return lh.detach(), rh.detach(), err


def fit_sequence(targets, smplx_model, hand_targets=None, hand_idx=HAND_SMPLX_IDX,
                 n_steps_beta=600, n_steps_pose=80, n_steps_hand0=250, n_steps_hand=60):
    """Joint-position fit. Body is fit first (beta+pose, then per-frame pose); if
    `hand_targets` (T, len(hand_idx), 3) is given, hands are fit per frame with the
    body frozen.
    Returns (betas, global_orient, body_pose, trans, left_hand, right_hand,
             errors, hand_errors). left_hand/right_hand are (T, 45); zero if no
    hand_targets.
    """
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
    fitted_lh = np.zeros((T, 45), dtype=np.float32)
    fitted_rh = np.zeros((T, 45), dtype=np.float32)

    errors = np.zeros(T, dtype=np.float32)
    hand_errors = np.zeros(T, dtype=np.float32)
    fitted_go[0] = go.detach()[0].numpy()
    fitted_bp[0] = bp.detach()[0].numpy()
    fitted_tr[0] = tr.detach()[0].numpy()

    # ── frame-0 hands (cold start, body frozen) ──
    if hand_targets is not None:
        lh0, rh0, hand_errors[0] = _fit_hands_frame(
            smplx_model, go.detach(), bp.detach(), tr.detach(), betas_fit,
            hand_targets[0], hand_idx, torch.zeros(1, 45), torch.zeros(1, 45),
            n_steps_hand0)
        fitted_lh[0] = lh0[0].numpy()
        fitted_rh[0] = rh0[0].numpy()

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

        # hands: warm-started from t-1, body frozen at this frame's fit
        if hand_targets is not None:
            lh, rh, hand_errors[t] = _fit_hands_frame(
                smplx_model, go_v.detach(), bp_v.detach(), tr_v.detach(), betas_fit,
                hand_targets[t], hand_idx,
                torch.from_numpy(fitted_lh[t - 1:t]), torch.from_numpy(fitted_rh[t - 1:t]),
                n_steps_hand)
            fitted_lh[t] = lh[0].numpy()
            fitted_rh[t] = rh[0].numpy()

    return betas_fit, fitted_go, fitted_bp, fitted_tr, fitted_lh, fitted_rh, errors, hand_errors


def build_object_npz(objs, primary_object):
    """humoto object trajectory (z-up) → (angles, trans) in y-up axis-angle."""
    obj_zup = np.array(objs[primary_object], dtype=np.float32)  # (T, 7) [w,x,y,z,lx,ly,lz]
    trans_yup = obj_zup[:, 4:] @ Z_TO_Y_MAT.T
    quat_xyzw = obj_zup[:, [1, 2, 3, 0]]
    rot_yup = Rot.from_matrix(Z_TO_Y_MAT.astype(np.float64)) * Rot.from_quat(quat_xyzw)
    angles = rot_yup.as_rotvec().astype(np.float32)
    return angles, trans_yup.astype(np.float32)


def build_object_mesh(glb_path, object_name, n_sample=340, seed=0, scene=None):
    """Extract object mesh from GLB, undo the GLB axis flip, sample N surface points.
    Pass a pre-loaded `scene` to avoid reloading the GLB for each object."""
    if scene is None:
        scene = trimesh.load(glb_path)
    mesh = scene.geometry[object_name]
    oriented = trimesh.Trimesh(vertices=mesh.vertices @ R_XFLIP_MAT.T, faces=mesh.faces)
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(oriented, n_sample)
    return oriented, pts.astype(np.float32)


def _pos_tag(sentence, nlp):
    """Replicate process_text.process_text: drop non-alpha tokens, lemmatize
    NOUN/VERB (except 'left'), return 'word/POS word/POS ...'."""
    sentence = sentence.replace('-', '')
    toks = []
    for token in nlp(sentence):
        word = token.text
        if not word.isalpha():
            continue
        if token.pos_ in ('NOUN', 'VERB') and word != 'left':
            word = token.lemma_
        toks.append(f'{word}/{token.pos_}')
    return ' '.join(toks)


def write_text_files(output_root=OUTPUT_ROOT, animations_json=HUMOTO_ANIMATIONS,
                     seq_names=None, verbose=True):
    """Write text.txt (InterAct format) into each sequences_seg/<seq>/ from the
    HUMOTO animations.json short_script. Single-level ('natural') text only;
    LLM paraphrase/shorten lines can be appended later for augmentation parity.

    Line format: '<sentence>#<word/POS ...>#0.0#0.0'
    """
    import json
    import spacy
    nlp = spacy.load('en_core_web_sm')
    with open(animations_json) as f:
        anims = {a['fileName']: a['short_script'] for a in json.load(f)}

    seg_root = os.path.join(output_root, 'sequences_seg')
    if seq_names is None:
        seq_names = sorted(os.listdir(seg_root))

    written = 0
    for seq in seq_names:
        if seq not in anims:
            if verbose:
                print(f'  [skip text] no short_script for {seq}')
            continue
        sentence = anims[seq].strip()
        line = f'{sentence}#{_pos_tag(sentence, nlp)}#0.0#0.0'
        with open(os.path.join(seg_root, seq, 'text.txt'), 'w') as f:
            f.write(line)
        written += 1
    if verbose:
        print(f'wrote text.txt for {written}/{len(seq_names)} sequences')
    return written


def process_humoto_sequence(pkl_path, glb_path, seq_name, primary_object,
                            smplx_model, mx_model, output_root=OUTPUT_ROOT,
                            body_corr=BODY_CORR, seed=0, verbose=True):
    """End-to-end conversion of ONE humoto sequence. Returns a metrics dict."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    targets, hand_targets, objs = load_all_targets(pkl_path, mx_model, body_corr)
    if primary_object not in objs:
        raise ValueError(f"'{primary_object}' not in pickle objects: {list(objs.keys())}")
    T = len(targets)
    if verbose:
        print(f"[1/4] {seq_name}: T={T}, objects={list(objs.keys())}")
    (betas_fit, fitted_go, fitted_bp, fitted_tr,
     fitted_lh, fitted_rh, errors, hand_errors) = fit_sequence(
        targets, smplx_model, hand_targets=hand_targets)
    if verbose:
        print(f"[2/4] fit: body mean={errors.mean() * 100:.2f}cm  max={errors.max() * 100:.2f}cm  "
              f"hand mean={hand_errors.mean() * 100:.2f}cm  ||beta||={betas_fit.norm():.2f}")

    out_seq_dir = os.path.join(output_root, 'sequences_seg', seq_name)
    out_obj_dir = os.path.join(output_root, 'objects', primary_object)
    os.makedirs(out_seq_dir, exist_ok=True)
    os.makedirs(out_obj_dir, exist_ok=True)

    poses = np.concatenate([fitted_go, fitted_bp, fitted_lh, fitted_rh], axis=1)  # (T, 156)
    np.savez(os.path.join(out_seq_dir, 'human.npz'),
             poses=poses.astype(np.float32),
             betas=betas_fit[0].numpy().astype(np.float32),
             trans=fitted_tr.astype(np.float32),
             gender='neutral')

    # ── Objects: write EVERY object as object_<name>.npz (+ mesh + sample_points).
    #    Multi-object STORAGE is faithful here: canonicalize_human canonicalizes
    #    every object_<name>.npz. `primary_object` is the one motion.npy will use
    #    as a TEMPORARY SCAFFOLD.
    #    TODO(thesis, open question): true multi-object motion representation.
    #    InterAct's motion.npy is single-object (476 human + 486 one-object). How
    #    to encode N objects (concat? per-object reps? attention?) is unresolved —
    #    flagged for advisor. For now motion.npy uses only `primary_object`.
    scene = trimesh.load(glb_path)
    written_objs = []
    for obj_name in objs:
        if obj_name not in scene.geometry:
            if verbose:
                print(f"   [skip obj] '{obj_name}': no GLB geometry")
            continue
        ang, tr = build_object_npz(objs, obj_name)
        np.savez(os.path.join(out_seq_dir, f'object_{obj_name}.npz'),
                 angles=ang, trans=tr, name=obj_name)
        obj_dir = os.path.join(output_root, 'objects', obj_name)
        os.makedirs(obj_dir, exist_ok=True)
        mesh, pts = build_object_mesh(glb_path, obj_name, seed=seed, scene=scene)
        np.save(os.path.join(obj_dir, 'sample_points.npy'), pts)
        mesh.export(os.path.join(obj_dir, f'{obj_name}.obj'))
        written_objs.append(obj_name)

    if primary_object not in written_objs:
        raise ValueError(f"primary '{primary_object}' not written (no GLB geometry?)")
    if verbose:
        print(f"[3/4] wrote human.npz + {len(written_objs)} object_<name>.npz "
              f"(primary='{primary_object}') + meshes/sample_points → {out_seq_dir}")

    return {
        'seq_dir': out_seq_dir,
        'obj_dir': out_obj_dir,
        'betas': betas_fit[0].numpy(),
        'mean_err_cm': float(errors.mean() * 100),
        'max_err_cm': float(errors.max() * 100),
        'hand_mean_err_cm': float(hand_errors.mean() * 100),
        'hand_max_err_cm': float(hand_errors.max() * 100),
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

    # Optional: write text.txt from animations.json (needs spaCy). Skips if absent.
    if os.path.exists(HUMOTO_ANIMATIONS):
        write_text_files(seq_names=list(SEQUENCE_CONFIG.keys()))
    else:
        print(f'(skip text) {HUMOTO_ANIMATIONS} not found - symlink animations.json to enable')
