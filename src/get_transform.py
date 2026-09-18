"""Eye-to-hand calibration via FK-derived robot poses.

Setup: a camera is mounted on the robot wrist and observes a STATIC ArUco board
fixed to the bench. This is an EYE-TO-HAND problem: the camera moves with the
end-effector while the observed target stays fixed in the base frame.

Worksheet notation:
    base frame   : 0          end-effector frame : 6
    camera frame : C          ArUco artifact     : W

The unknown is T_6_C (end-effector -> camera). The loop closure per pose i is
    T_0_W = T_0_6(i) @ T_6_C @ T_C_W(i)
where T_0_W (board position in base frame) is constant across all poses. We solve
for T_6_C with a self-contained conjugation-based solver (see solve_eye_to_hand);
a correct fit makes the reconstructed T_0_W collapse onto a single point.

For each pose the base->EE transform is taken from the stored T_0_6 (computed by
forward kinematics at capture time), falling back to recomputing FK from the
stored joint angles, then to the controller's cartesian get_position() for legacy
data that has neither.

NOTE: this deliberately does NOT use cv2.calibrateHandEye -- that API implements
the eye-IN-hand AX=BX form (target moves with the gripper), which is the wrong
model here, and it is absent from the installed OpenCV build anyway.

Verbose output: every captured pose is echoed up front, and the per-pose
artifact-position residual (deviation of T_0_W from its mean) plus an aggregate
error norm are printed so you can see exactly where any remaining scatter comes
from.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from fk_lite6 import fk_lite6
from lab_config import (
    S_MAX_EXCELLENT_MM, S_MAX_GOOD_MM, S_MAX_ACCEPTABLE_MM,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'


def fk_pose(joints_deg):
    """Base->EE transform (mm) from 6 joint angles in degrees via Craig-DH FK."""
    return fk_lite6(np.deg2rad(np.asarray(joints_deg, dtype=np.float64)))


def cartesian_pose(raw_pose):
    """Fallback Base->EE transform from controller [x,y,z,roll,pitch,yaw] (mm,deg)."""
    x, y, z, roll, pitch, yaw = raw_pose
    rmat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rmat
    T[:3, 3] = [x, y, z]
    return T


def resolve_T_0_6(d):
    """Pick the best available source for a pose's Base->EE transform.

    Prefers the stored T_0_6 (FK at capture time); falls back to recomputing FK
    from stored joints, then to cartesian for legacy captures.
    Returns (T_0_6, source_label).
    """
    if d.get('T_0_6') is not None:
        return np.asarray(d['T_0_6'], dtype=np.float64), 'stored-FK'
    if d.get('robot_joints') is not None and len(d['robot_joints']) == 6:
        return fk_pose(d['robot_joints']), 'FK-recompute'
    if d.get('robot_pose_raw') is not None:
        return cartesian_pose(d['robot_pose_raw']), 'cartesian'
    raise KeyError(f"pose has no usable robot pose (keys: {list(d.keys())})")


def solve_eye_to_hand(T_0_6_list, T_C_W_list):
    """Solve T_6_C for an EYE-TO-HAND setup (wrist camera observing a static artifact).

    Geometry: the camera is mounted on the end-effector and observes a fixed
    ArUco board. The board's position in the base frame, T_0_W, is CONSTANT. The
    loop closure per pose i is

        T_0_W = T_0_6(i) @ X @ T_C_W(i),      X = T_6_C  (unknown)

    Taking the ratio of two consecutive poses eliminates the constant T_0_W:

        A_i = inv(T_0_6(i)) @ T_0_6(i+1)   ==   X @ Bp_i @ inv(X)
        Bp_i = T_C_W(i) @ inv(T_C_W(i+1))

    This is a CONJUGATION relation (A = X Bp X^-1), not the classic eye-in-hand
    AX=BX form -- using the latter on this geometry is what produced the earlier
    "POOR" / s_max ~ 171 mm result.

    Rotation: conjugation preserves rotation angle but rotates the axis by X_r,
    so rotvec(A_i) = X_r @ rotvec(Bp_i). Stacking all pairs gives the least-squares
    problem X_r @ RB = RA, solved as X_r = RA @ pinv(RB) and re-orthonormalized.

    Translation: expanding A_t = X_r @ Rp_bp @ X_r^T ... yields the linear system
        (I - X_r @ Rp_bp @ X_r^T) @ x = a_t - X_r @ bp_t
    stacked over all pairs and solved by least squares.

    Validated against synthetic ground truth (exact recovery, recon spread -> 0)
    before use. Returns (T_6_C, ok); ok is False when the input set is too small
    or degenerate (rank-deficient rotation stack), which RANSAC must tolerate.
    """
    n = len(T_0_6_list)
    if n < 3:
        return None, False

    # --- Rotation part -------------------------------------------------------
    RA_cols = []
    RB_cols = []
    for i in range(n - 1):
        A = np.linalg.inv(T_0_6_list[i]) @ T_0_6_list[i + 1]
        Bp = T_C_W_list[i] @ np.linalg.inv(T_C_W_list[i + 1])
        RA_cols.append(R.from_matrix(A[:3, :3]).as_rotvec())
        RB_cols.append(R.from_matrix(Bp[:3, :3]).as_rotvec())

    RA = np.array(RA_cols).T   # 3 x (n-1)
    RB = np.array(RB_cols).T   # 3 x (n-1)

    Rx = RA @ np.linalg.pinv(RB)
    U, S, Vt = np.linalg.svd(Rx)
    # Rank deficiency means the sampled rotations don't span enough directions
    # to determine the transform; flag it so callers can retry with other poses.
    if min(S) < 1e-8:
        return None, False
    Rx = U @ Vt
    if np.linalg.det(Rx) < 0:
        U[:, -1] *= -1
        Rx = U @ Vt

    # --- Translation part ----------------------------------------------------
    M_rows = []
    r_vecs = []
    for i in range(n - 1):
        A = np.linalg.inv(T_0_6_list[i]) @ T_0_6_list[i + 1]
        Bp = T_C_W_list[i] @ np.linalg.inv(T_C_W_list[i + 1])
        coef = np.eye(3) - Rx @ Bp[:3, :3] @ Rx.T
        rhs = A[:3, 3] - Rx @ Bp[:3, 3]
        M_rows.append(coef)
        r_vecs.append(rhs.reshape(-1))

    try:
        x, *_ = np.linalg.lstsq(np.vstack(M_rows), np.concatenate(r_vecs), rcond=None)
    except np.linalg.LinAlgError:
        return None, False

    T_6_C = np.eye(4)
    T_6_C[:3, :3] = Rx
    T_6_C[:3, 3] = x
    return T_6_C, True


def evaluate_consistency(T_6_C, data):
    """Reconstruct the artifact position in the base frame for every pose.

    T_0_W = T_0_6 @ T_6_C @ T_C_W ; the translation column of T_0_W is the
    artifact origin expressed in the base frame. A correct calibration makes all
    these land on one point.

    Returns per-pose positions, their mean, per-axis std, max deviation, and the
    RMS error norm (root-mean-square of each pose's deviation from the mean).
    """
    positions = []
    for d in data:
        T_0_6, _src = resolve_T_0_6(d)
        T_0_W = T_0_6 @ T_6_C @ d['T_C_W']
        positions.append(T_0_W[:3, 3])

    positions = np.array(positions)
    mean_pos = positions.mean(axis=0)
    deviations = positions - mean_pos
    rms_norm = float(np.sqrt(np.mean(np.sum(deviations ** 2, axis=1))))
    std_pos = positions.std(axis=0)
    max_dev = float(np.max(np.linalg.norm(deviations, axis=1)))

    return {'positions': positions, 'mean': mean_pos, 'std': std_pos,
            'rms': rms_norm, 'max_dev': max_dev}


def save_result(T_6_C, filename_npy='T_6_C_normal.npy', filename_txt='T_6_C_normal.txt'):
    np.save(str(_DATA_DIR / filename_npy), T_6_C)

    with open(str(_DATA_DIR / filename_txt), 'w') as f:
        f.write("Eye-to-Hand Calibration Result: T_6_C (Direct Solver)\n")
        f.write("=" * 60 + "\n\n")
        f.write("4x4 Transformation Matrix:\n")
        for row in T_6_C:
            f.write(f"{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}\n")

        f.write("\nTranslation [mm]:\n")
        f.write(f"X: {T_6_C[0,3]:.6f}\n")
        f.write(f"Y: {T_6_C[1,3]:.6f}\n")
        f.write(f"Z: {T_6_C[2,3]:.6f}\n")

        euler = R.from_matrix(T_6_C[:3, :3]).as_euler('xyz', degrees=True)
        f.write("\nEuler xyz [deg]:\n")
        f.write(f"Roll:  {euler[0]:.6f}\n")
        f.write(f"Pitch: {euler[1]:.6f}\n")
        f.write(f"Yaw:   {euler[2]:.6f}\n")


if __name__ == "__main__":
    data = np.load(str(_DATA_DIR / "calibration_data.npy"), allow_pickle=True).tolist()
    print(f"Loaded {len(data)} poses")

    valid_data = [d for d in data if 'T_C_W' in d]
    print(f"Valid poses with T_C_W: {len(valid_data)}")

    if len(valid_data) < 5:
        print("Not enough valid data.")
        raise SystemExit

    # ---- Verbose dump of every captured pose -------------------------------
    print("\n" + "=" * 90)
    print("CAPTURED POSES (source used for T_0_6 shown per pose)")
    print("=" * 90)
    for i, d in enumerate(valid_data):
        T_0_6, src = resolve_T_0_6(d)
        pos = T_0_6[:3, 3]
        rpy = R.from_matrix(T_0_6[:3, :3]).as_euler('xyz', degrees=True)
        t_cw = d['T_C_W'][:3, 3]
        rep = d.get('reproj_error_px', float('nan'))
        jtag = ""
        if d.get('robot_joints') is not None:
            jtag = "  q(deg)=" + ",".join(f"{j:.1f}" for j in d['robot_joints'])
        print(f"[{i+1:2d}] ({src:14s}) EE=({pos[0]:7.1f},{pos[1]:7.1f},{pos[2]:7.1f}) mm  "
              f"RPY=({rpy[0]:6.1f},{rpy[1]:6.1f},{rpy[2]:6.1f})°  "
              f"cam→art=({t_cw[0]:6.1f},{t_cw[1]:6.1f},{t_cw[2]:6.1f}) mm  "
              f"reproj={rep:5.2f}px{jtag}")

    # Solve T_6_C with the self-contained eye-to-hand solver over all poses.
    T_0_6_list = [resolve_T_0_6(d)[0] for d in valid_data]
    T_C_W_list = [np.asarray(d['T_C_W'], dtype=np.float64) for d in valid_data]

    print("\n" + "=" * 90)
    print("EYE-TO-HAND SOLVE (static bench artifact, wrist camera)")
    print("=" * 90)

    T_6_C, ok = solve_eye_to_hand(T_0_6_list, T_C_W_list)
    if not ok or T_6_C is None:
        print("\nCalibration failed: pose set too small or degenerate "
              "(insufficient rotational variety). Re-capture with broader motion.")
        raise SystemExit

    cons = evaluate_consistency(T_6_C, valid_data)
    best = {'method_name': 'eye-to-hand', 'T_6_C': T_6_C, **cons}

    print(f"\n--- {best['method_name']} ---")
    print(f"{'pose':>4} | artifact@base (mm)                       | resid(mm)")
    print("-" * 68)
    for k, p in enumerate(cons['positions']):
        resid = float(np.linalg.norm(p - cons['mean']))
        print(f"{k+1:>4} | ({p[0]:7.2f},{p[1]:7.2f},{p[2]:7.2f})          | {resid:7.3f}")
    print("-" * 68)
    print(f"mean art@base : ({cons['mean'][0]:.2f},{cons['mean'][1]:.2f},{cons['mean'][2]:.2f}) mm")
    print(f"std XYZ       : ({cons['std'][0]:.3f},{cons['std'][1]:.3f},{cons['std'][2]:.3f}) mm")
    print(f"RMS error norm: {cons['rms']:.3f} mm   max dev: {cons['max_dev']:.3f} mm")

    T_6_C = best['T_6_C']
    euler_xyz = R.from_matrix(T_6_C[:3, :3]).as_euler('xyz', degrees=True)

    print("\n" + "=" * 90)
    print("BEST RESULT")
    print("=" * 90)
    print(f"Best HE method   : {best['method_name']}")
    print(f"RMS error norm   : {best['rms']:.3f} mm")
    print(f"Std XYZ [mm]     : {best['std']}")
    print(f"Max dev [mm]     : {best['max_dev']:.3f}")
    print("\nT_6_C =")
    print(T_6_C)
    print(f"\nTranslation [mm]: {T_6_C[:3, 3]}")
    print(f"Euler xyz [deg]: {euler_xyz}")

    s_max = float(np.max(best['std']))
    if s_max < S_MAX_EXCELLENT_MM:
        print(f"\n✓ Calibration quality (s_max={s_max:.1f} mm): EXCELLENT")
    elif s_max < S_MAX_GOOD_MM:
        print(f"\n✓ Calibration quality (s_max={s_max:.1f} mm): GOOD")
    elif s_max < S_MAX_ACCEPTABLE_MM:
        print(f"\n⚠ Calibration quality (s_max={s_max:.1f} mm): ACCEPTABLE")
    else:
        print(f"\n✗ Calibration quality (s_max={s_max:.1f} mm): POOR - "
              f"re-capture with broader wrist motion / depth variety")

    save_result(T_6_C)
    print("\nSaved: T_6_C_normal.npy, T_6_C_normal.txt (in data/)")
