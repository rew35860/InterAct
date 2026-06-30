"""DIRECT Mixamo -> InterMimic .pt packer (bypasses SMPL-X fitting).

Produces a schema-correct .pt for OmniRetarget directly from HUMOTO's Mixamo
skeleton, WITHOUT the SMPL-X body-model fit. This is the "direct" arm of the
comparison:

    via InterAct : Mixamo -> fit SMPL-X -> forward -> 52 joints -> .pt
    direct (here): Mixamo joint positions -> dropped straight into 52 slots -> .pt

Both produce a valid smplh-format .pt (52 joint POSITIONS in SMPLH_DEMO_JOINTS
slot order + object pose). OmniRetarget reads only those positions; it never
sees a skeleton, so "Mixamo positions wearing SMPL-H slot labels" is exactly
what it expects.

The Mixamo -> SMPL-H slot mapping is BODY_CORR (process_humoto.py). We reuse
process_humoto's FK (compute_targets) to get Mixamo joints at SMPL-X body slots
in y-up, remap to the 52 SMPLH slots, rotate y-up -> z-up (the same 90deg-X that
interact2mimic uses), and pack. OmniRetarget re-does floor-normalization +
scaling, so only orientation + slot ordering must be right here.

Run (interact env, from simulation/):
    python humoto_direct_to_pt.py
"""
import os

os.environ.setdefault('HUMOTO_UPBONE',
                       '/media/natcha/Natcha_T7/InterAct_workspace/humoto_full/humoto_upbone_pkl')

import sys
sys.path.insert(0, os.path.abspath('..'))   # project root for `process` package

import numpy as np
import torch
import pickle
from scipy.spatial.transform import Rotation as sRot

from process.process_humoto import (
    load_models, compute_targets, build_object_npz, HUMOTO_UPBONE, HAND_CORR,
)

# clip -> primary object (must match models/g1/g1_29dof_w_<obj>.xml + pkl object key)
CLIPS = {
    'checking_floor_lamp_with_right_hand-024':    'floor_lamp',
    'carrying_cutting_board_with_right_hand-213':  'cutting_board',
    'carrying_whisk_with_right_hand-037':          'whisk',
    # chairs (matched to OMOMO woodchair/whitechair for the limits comparison)
    'carrying_working_chair_with_both_hands-443':  'working_chair',
    'carrying_low_chair_with_right_hand-599':      'low_chair',
    'carrying_dining_chair_with_both_hands-737':   'dining_chair',
    # small in-hand instruments + tools (extra limit data points)
    'move_around_while_playing_ukelele-932':                              'ukelele',
    'play_guitar_then_bow_and_wave-525':                                  'guitar',
    'working_with_hammer_in_garage-729':                                  'hammer',
    'put_screwdriver_on_the_ground_stand_up_and_lean_against_table-567':  'screwdriver',
    # chunky object starting at TABLE height (controlled "well-behaved" case)
    'checking_organizer_medium_on_table-289':                            'organizer_medium',
    # multi-object cooking scene; primary = the in-hand spatula
    'baking_with_spatula_mixing_bowl_and_scooping_to_tray-244':          'spatula',
}

OUT_DIR = 'intermimic/InterAct/humoto_direct'

# SMPL-X body joint index (compute_targets output) -> SMPLH_DEMO_JOINTS slot index.
# Only the 15 joints OmniRetarget's JOINTS_MAPPING(smplh,g1) actually matches.
SMPLX_TO_SMPLH = {
    0:  0,   # pelvis        -> Pelvis
    1:  1,   # left_hip      -> L_Hip
    4:  2,   # left_knee     -> L_Knee
    7:  3,   # left_ankle    -> L_Ankle
    10: 4,   # left_foot     -> L_Toe
    2:  5,   # right_hip     -> R_Hip
    5:  6,   # right_knee    -> R_Knee
    8:  7,   # right_ankle   -> R_Ankle
    11: 8,   # right_foot    -> R_Toe
    16: 15,  # left_shoulder -> L_Shoulder
    18: 16,  # left_elbow    -> L_Elbow
    20: 17,  # left_wrist    -> L_Wrist
    17: 34,  # right_shoulder-> R_Shoulder
    19: 35,  # right_elbow   -> R_Elbow
    21: 36,  # right_wrist   -> R_Wrist
}

# Same 90deg about X as interact2mimic (y-up canonical -> z-up mujoco).
R_X = sRot.from_euler('x', np.pi / 2)

PT_WIDTH = 331 + 52 + 52 * 4   # 591, matches interact2mimic


def pack_clip(seq, primary_object, mx_model):
    pkl_path = os.path.join(HUMOTO_UPBONE, seq, f'{seq}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    arm, objs = data['armature'], data['objects']

    # 1) Mixamo joints at SMPL-X body slots (T, 22, 3), y-up  [the DIRECT positions]
    targets = compute_targets(arm, mx_model)          # (T, 22, 3)
    T = targets.shape[0]

    # 2) place into 52 SMPLH slots (others stay zero)
    joints52 = np.zeros((T, 52, 3), dtype=np.float32)
    for smplx_idx, smplh_idx in SMPLX_TO_SMPLH.items():
        joints52[:, smplh_idx] = targets[:, smplx_idx]

    # 3) object trajectory (y-up), same as the InterAct path uses
    if primary_object not in objs:
        primary_object = next(iter(objs))             # fallback to first object
        print(f'  (primary not in pkl; using {primary_object!r})')
    obj_angles, obj_trans = build_object_npz(objs, primary_object)  # rotvec, trans (y-up)

    # 4) y-up -> z-up (same rotation applied to joints AND object => relative
    #    geometry preserved exactly)
    joints_z = R_X.apply(joints52.reshape(-1, 3)).reshape(T, 52, 3)
    obj_trans_z = R_X.apply(obj_trans)
    obj_quat_z = (R_X * sRot.from_rotvec(obj_angles)).as_quat()    # xyzw

    # 4b) Floor-normalize the whole scene (the step canonicalize_human does that the
    #     raw pkl skips). Without it a floor-standing object whose mesh origin is at
    #     its centroid sinks under the floor after G1 down-scaling. We compute the
    #     lowest point of (foot joints + transformed object mesh) across the
    #     trajectory and subtract it, so the lowest contact rests at z=0.
    used_idx = list(SMPLX_TO_SMPLH.values())
    foot_min = joints_z[:, [3, 4, 7, 8], 2].min()   # ankles+toes (SMPLH slots)
    obj_floor = foot_min
    # IMPORTANT: use the SAME mesh the retargeter renders with (holosoma models/),
    # NOT the InterAct data mesh. Their origins differ (e.g. floor_lamp: retargeter
    # mesh origin is near the top, extends ~1.48m down; data mesh is centroid-origin
    # +/-0.89). Using the wrong one mis-computes the floor offset and sinks the object.
    HOLOSOMA_MODELS = '/home/natcha/Downloads/projects/holosoma/src/holosoma_retargeting/holosoma_retargeting/models'
    obj_mesh_path = os.path.join(HOLOSOMA_MODELS, primary_object, f'{primary_object}.obj')
    if not os.path.isfile(obj_mesh_path):
        obj_mesh_path = os.path.join('..', 'data', 'humoto_full', 'objects',
                                     primary_object, f'{primary_object}.obj')
    if os.path.isfile(obj_mesh_path):
        import trimesh
        v0 = np.asarray(trimesh.load(obj_mesh_path, force='mesh').vertices)  # (n,3) z-up mesh
        Robj = sRot.from_quat(obj_quat_z).as_matrix()                        # (T,3,3)
        obj_v = np.einsum('nj,tij->tni', v0, Robj) + obj_trans_z[:, None, :]  # (T,n,3)
        obj_floor = obj_v[..., 2].min()
    floor_z = min(foot_min, obj_floor)
    joints_z[..., 2] -= floor_z
    obj_trans_z[..., 2] -= floor_z
    print(f'  floor-normalize: foot_min={foot_min:.3f} obj_min={obj_floor:.3f} -> shift {-floor_z:.3f}')

    # 5) pack (T, 591) — only the columns OmniRetarget reads need to be correct
    out = torch.zeros((T, PT_WIDTH), dtype=torch.float64)
    out[:, 162:162 + 52 * 3] = torch.from_numpy(joints_z.reshape(T, -1)).double()
    out[:, 318:321] = torch.from_numpy(obj_trans_z).double()
    out[:, 321:325] = torch.from_numpy(obj_quat_z).double()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{seq}.pt')
    torch.save(out, out_path)
    print(f'  wrote {out_path}  (T={T})')
    return out_path


def main():
    smplx_model, mx_model = load_models()
    for seq, obj in CLIPS.items():
        print(f'=== {seq}  (object: {obj}) ===')
        pack_clip(seq, obj, mx_model)
    print(f'\nDone. Direct .pt files in {os.path.abspath(OUT_DIR)}')


if __name__ == '__main__':
    main()
