"""Direct Mixamo -> InterMimic .pt packer for HUMOTO (no SMPL-X fit).

Drops HUMOTO's Mixamo joint POSITIONS straight into the 52 SMPLH slots and packs
a schema-correct .pt for OmniRetarget. (The other "arm" fits SMPL-X first; both
end up as 52 joint positions + an object pose, which is all OmniRetarget reads.)

Pipeline:  Mixamo FK (process_humoto.compute_targets) -> 52 SMPLH slots
           -> y-up to z-up -> floor-normalize -> pack columns
           [162:318]=joints, [318:325]=object pose.

Run (interact env, from simulation/):
    export HUMOTO_UPBONE=.../humoto_data/humoto_upbone_pkl   # Stage-A pkls   (required)
    export HUMOTO_REPO=.../humoto                            # for human_model (required)
    export HUMOTO_OBJECTS=.../humoto_objects_0805            # <obj>/<obj>.obj  (optional, floor-norm)
    python humoto_direct_to_pt.py [seq ...]                  # default: all CLIPS

Output:  $HUMOTO_PT_OUT  (default <InterAct>/result/humoto_pt/<seq>.pt)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # InterAct root, for the `process` package

for _var, _hint in [('HUMOTO_UPBONE', 'the up_bone pkl dir (Stage-A output)'),
                    ('HUMOTO_REPO', 'the humoto repo (provides human_model)')]:
    if _var not in os.environ:
        sys.exit(f"Set ${_var} to {_hint}, e.g.\n  export {_var}=...")

import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from process.process_humoto import compute_targets, build_object_npz, HUMOTO_UPBONE, HUMOTO_JSON
from human_model.human_model import HumanModelDifferentiable

# clip -> primary object (must match models/g1/g1_29dof_w_<obj>.xml and the pkl's object key)
CLIPS = {
    'checking_floor_lamp_with_right_hand-024':     'floor_lamp',
    'carrying_cutting_board_with_right_hand-213':  'cutting_board',
    'carrying_whisk_with_right_hand-037':          'whisk',
    'carrying_working_chair_with_both_hands-443':  'working_chair',
    'carrying_low_chair_with_right_hand-599':      'low_chair',
    'carrying_dining_chair_with_both_hands-737':   'dining_chair',
    'move_around_while_playing_ukelele-932':       'ukelele',
    'play_guitar_then_bow_and_wave-525':           'guitar',
    'working_with_hammer_in_garage-729':           'hammer',
    'put_screwdriver_on_the_ground_stand_up_and_lean_against_table-567': 'screwdriver',
    'checking_organizer_medium_on_table-289':      'organizer_medium',
    'baking_with_spatula_mixing_bowl_and_scooping_to_tray-244': 'spatula',
}

OUT_DIR = os.environ.get('HUMOTO_PT_OUT',
                         os.path.join(os.path.dirname(_HERE), 'result', 'humoto_pt'))

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

R_X = sRot.from_euler('x', np.pi / 2)    # y-up -> z-up (same 90deg as interact2mimic)
PT_WIDTH = 331 + 52 + 52 * 4             # 591


def _floor_z(joints, obj, quat, trans):
    """Lowest contact point (feet + object mesh), to anchor the scene at z=0.

    Uses the SAME .obj the retargeter renders with (mesh origins differ from the
    InterAct data mesh), found via $HUMOTO_OBJECTS / $HOLOSOMA_MODELS.
    """
    foot_min = joints[:, [3, 4, 7, 8], 2].min()      # ankles + toes
    obj_min = foot_min
    obj_dirs = [d for d in (os.environ.get('HUMOTO_OBJECTS'),
                            os.environ.get('HOLOSOMA_MODELS'),
                            os.path.join(os.path.dirname(_HERE), 'data', 'humoto_full', 'objects')) if d]
    mesh = None
    for d in obj_dirs:
        cand = os.path.join(d, obj, f'{obj}.obj')
        if os.path.isfile(cand):
            mesh = cand
            break
    if mesh:
        import trimesh
        v = np.asarray(trimesh.load(mesh, force='mesh').vertices)      # (n, 3)
        R = sRot.from_quat(quat).as_matrix()                          # (T, 3, 3)
        obj_min = (np.einsum('nj,tij->tni', v, R) + trans[:, None])[..., 2].min()
    print(f'  floor-norm: feet={foot_min:.3f} obj={obj_min:.3f} -> shift {-min(foot_min, obj_min):.3f}')
    return min(foot_min, obj_min)


def pack_clip(seq, obj, mx_model):
    data = pickle.load(open(os.path.join(HUMOTO_UPBONE, seq, f'{seq}.pkl'), 'rb'))
    arm, objs = data['armature'], data['objects']

    # Mixamo joints (y-up) -> 52 SMPLH slots (unmapped slots stay zero)
    targets = compute_targets(arm, mx_model)          # (T, 22, 3)
    T = targets.shape[0]
    joints = np.zeros((T, 52, 3), np.float32)
    for sx, sh in SMPLX_TO_SMPLH.items():
        joints[:, sh] = targets[:, sx]

    # object trajectory (y-up)
    if obj not in objs:
        obj = next(iter(objs))
        print(f'  (primary not in pkl; using {obj!r})')
    ang, trans = build_object_npz(objs, obj)          # rotvec, trans

    # y-up -> z-up (same rotation on joints AND object preserves relative geometry)
    joints = R_X.apply(joints.reshape(-1, 3)).reshape(T, 52, 3)
    trans = R_X.apply(trans)
    quat = (R_X * sRot.from_rotvec(ang)).as_quat()    # xyzw

    floor = _floor_z(joints, obj, quat, trans)
    joints[..., 2] -= floor
    trans[..., 2] -= floor

    # pack only the columns OmniRetarget reads
    out = torch.zeros((T, PT_WIDTH), dtype=torch.float64)
    out[:, 162:318] = torch.from_numpy(joints.reshape(T, -1)).double()
    out[:, 318:321] = torch.from_numpy(trans).double()
    out[:, 321:325] = torch.from_numpy(quat).double()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{seq}.pt')
    torch.save(out, path)
    print(f'  wrote {path}  (T={T})')


def main():
    args = sys.argv[1:]
    if len(args) >= 2:                 # explicit: <seq> <object>  (any clip)
        jobs = [(args[0], args[1])]
    elif len(args) == 1:               # <seq> -> look up object in CLIPS
        jobs = [(args[0], CLIPS.get(args[0]))]
    else:                              # no args -> all known clips
        jobs = list(CLIPS.items())

    mx_model = HumanModelDifferentiable(HUMOTO_JSON, device='cpu')   # Mixamo FK; SMPL-X not needed
    for seq, obj in jobs:
        if obj is None:
            print(f"skip {seq!r}: no object given and not in CLIPS")
            continue
        print(f'=== {seq}  ({obj}) ===')
        pack_clip(seq, obj, mx_model)
    print(f'\nDone -> {os.path.abspath(OUT_DIR)}')


if __name__ == '__main__':
    main()
