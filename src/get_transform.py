"""Hand-eye calibration via FK-derived robot poses + OpenCV solvers.

Loads captured pairs from ../data/calibration_data.npy. For each pose the base->EE
transform is derived by forward kinematics from the stored joint angles (fk_lite6),
falling back to the controller's cartesian get_position() only for legacy data that
has no joints. Several OpenCV hand-eye methods are then scored by reconstructed
board-position consistency; the best T_ee_cam is saved to ../data/.

Verbose output: every captured pose is echoed up front, and for each method the
per-pose board-position residual (deviation of T_base_board from its mean) plus an
aggregate error norm are printed so you can see exactly where the scatter comes from.
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from fk_lite6 import fk_lite6

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


def resolve_T_base_ee(d):
    """Pick the best available source for a pose's Base->EE transform.

    Prefers FK from stored joints; falls back to cartesian for legacy captures.
    Returns (T_base_ee, source_label).
    """
    if d.get('robot_joints') is not None and len(d['robot_joints']) == 6:
        return fk_pose(d['robot_joints']), 'FK'
    if d.get('T_base_ee') is not None:
        # Stored FK matrix from capture time (newer data without a recompute path)
        return np.asarray(d['T_base_ee'], dtype=np.float64), 'stored-FK'
    if d.get('robot_pose_raw') is not None:
        return cartesian_pose(d['robot_pose_raw']), 'cartesian'
    raise KeyError(f"pose has no usable robot pose (keys: {list(d.keys())})")


def evaluate_consistency(T_ee_cam, data):
    """Reconstruct the board position in the base frame for every pose.

    Returns per-pose positions, their mean, per-axis std, max deviation, and the
    RMS error norm (root-mean-square of each pose's deviation from the mean).
    """
    positions = []
    for d in data:
        T_base_ee, _src = resolve_T_base_ee(d)
        T_base_board = T_base_ee @ T_ee_cam @ d['T_cam_board']
        positions.append(T_base_board[:3, 3])

    positions = np.array(positions)
    mean_pos = positions.mean(axis=0)
    deviations = positions - mean_pos
    rms_norm = float(np.sqrt(np.mean(np.sum(deviations ** 2, axis=1))))
    std_pos = positions.std(axis=0)
    max_dev = float(np.max(np.linalg.norm(deviations, axis=1)))

    return {'positions': positions, 'mean': mean_pos, 'std': std_pos,
            'rms': rms_norm, 'max_dev': max_dev}


def save_result(T_ee_cam, filename_npy='T_ee_cam_normal.npy', filename_txt='T_ee_cam_normal.txt'):
    np.save(str(_DATA_DIR / filename_npy), T_ee_cam)

    with open(str(_DATA_DIR / filename_txt), 'w') as f:
        f.write("Hand-Eye Calibration Result: T_ee_cam (Normal / Direct)\n")
        f.write("=" * 60 + "\n\n")
        f.write("4x4 Transformation Matrix:\n")
        for row in T_ee_cam:
            f.write(f"{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}\n")

        f.write("\nTranslation [mm]:\n")
        f.write(f"X: {T_ee_cam[0,3]:.6f}\n")
        f.write(f"Y: {T_ee_cam[1,3]:.6f}\n")
        f.write(f"Z: {T_ee_cam[2,3]:.6f}\n")

        euler = R.from_matrix(T_ee_cam[:3, :3]).as_euler('xyz', degrees=True)
        f.write("\nEuler xyz [deg]:\n")
        f.write(f"Roll:  {euler[0]:.6f}\n")
        f.write(f"Pitch: {euler[1]:.6f}\n")
        f.write(f"Yaw:   {euler[2]:.6f}\n")


if __name__ == "__main__":
    data = np.load(str(_DATA_DIR / "calibration_data.npy"), allow_pickle=True).tolist()
    print(f"Loaded {len(data)} poses")

    valid_data = [d for d in data if 'T_cam_board' in d]
    print(f"Valid poses with T_cam_board: {len(valid_data)}")

    if len(valid_data) < 5:
        print("Not enough valid data.")
        raise SystemExit

    # ---- Verbose dump of every captured pose -------------------------------
    print("\n" + "=" * 90)
    print("CAPTURED POSES (source used for T_base_ee shown per pose)")
    print("=" * 90)
    for i, d in enumerate(valid_data):
        T_be, src = resolve_T_base_ee(d)
        pos = T_be[:3, 3]
        rpy = R.from_matrix(T_be[:3, :3]).as_euler('xyz', degrees=True)
        t_cb = d['T_cam_board'][:3, 3]
        rep = d.get('reproj_error_px', float('nan'))
        jtag = ""
        if d.get('robot_joints') is not None:
            jtag = "  q(deg)=" + ",".join(f"{j:.1f}" for j in d['robot_joints'])
        print(f"[{i+1:2d}] ({src:10s}) EE=({pos[0]:7.1f},{pos[1]:7.1f},{pos[2]:7.1f}) mm  "
              f"RPY=({rpy[0]:6.1f},{rpy[1]:6.1f},{rpy[2]:6.1f})°  "
              f"cam→board=({t_cb[0]:6.1f},{t_cb[1]:6.1f},{t_cb[2]:6.1f}) mm  "
              f"reproj={rep:5.2f}px{jtag}")

    # Build A/B lists once (same for every method)
    R_gripper2base, t_gripper2base = [], []
    for d in valid_data:
        T_be, _src = resolve_T_base_ee(d)
        R_gripper2base.append(T_be[:3, :3])
        t_gripper2base.append(T_be[:3, 3])
    R_target2cam = [d['T_cam_board'][:3, :3] for d in valid_data]
    t_target2cam = [d['T_cam_board'][:3, 3] for d in valid_data]

    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
    ]

    best = None
    best_rms = float("inf")

    print("\n" + "=" * 90)
    print("TESTING HAND-EYE METHODS (FK-derived robot poses)")
    print("=" * 90)

    for method, method_name in methods:
        try:
            R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam, method=method
            )
            T_ee_cam = np.eye(4, dtype=np.float64)
            T_ee_cam[:3, :3] = R_ee_cam
            T_ee_cam[:3, 3] = t_ee_cam.flatten()

            cons = evaluate_consistency(T_ee_cam, valid_data)

            print(f"\n--- {method_name} ---")
            print(f"{'pose':>4} | board@base (mm)                          | resid(mm)")
            print("-" * 68)
            for k, p in enumerate(cons['positions']):
                resid = float(np.linalg.norm(p - cons['mean']))
                print(f"{k+1:>4} | ({p[0]:7.2f},{p[1]:7.2f},{p[2]:7.2f})          | {resid:7.3f}")
            print("-" * 68)
            print(f"mean board@base : ({cons['mean'][0]:.2f},{cons['mean'][1]:.2f},{cons['mean'][2]:.2f}) mm")
            print(f"std XYZ         : ({cons['std'][0]:.3f},{cons['std'][1]:.3f},{cons['std'][2]:.3f}) mm")
            print(f"RMS error norm  : {cons['rms']:.3f} mm   max dev: {cons['max_dev']:.3f} mm")

            if cons['rms'] < best_rms:
                best_rms = cons['rms']
                best = {'method_name': method_name, 'T_ee_cam': T_ee_cam, **cons}

        except Exception as e:
            print(f"{method_name:10s} | failed: {e}")

    if best is None:
        print("\nCalibration failed.")
        raise SystemExit

    T_ee_cam = best['T_ee_cam']
    euler_xyz = R.from_matrix(T_ee_cam[:3, :3]).as_euler('xyz', degrees=True)

    print("\n" + "=" * 90)
    print("BEST RESULT")
    print("=" * 90)
    print(f"Best HE method   : {best['method_name']}")
    print(f"RMS error norm   : {best['rms']:.3f} mm")
    print(f"Std XYZ [mm]     : {best['std']}")
    print(f"Max dev [mm]     : {best['max_dev']:.3f}")
    print("\nT_ee_cam =")
    print(T_ee_cam)
    print(f"\nTranslation [mm]: {T_ee_cam[:3, 3]}")
    print(f"Euler xyz [deg]: {euler_xyz}")

    if np.max(best['std']) < 5:
        print("\n✓ Calibration quality: EXCELLENT")
    elif np.max(best['std']) < 10:
        print("\n✓ Calibration quality: GOOD")
    elif np.max(best['std']) < 20:
        print("\n⚠ Calibration quality: ACCEPTABLE")
    else:
        print("\n✗ Calibration quality: POOR")

    save_result(T_ee_cam)
    print("\nSaved: T_ee_cam_normal.npy, T_ee_cam_normal.txt (in data/)")
