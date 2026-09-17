"""Validate hand-eye calibrations with a RELOCATED board (live, keygated).

Workflow:
  1. Complete BOTH calibrations first:
       - Normal:  transformation_calibration.py + get_tranform_3D.py
                  → writes data/T_ee_cam_normal.npy
       - RANSAC:  ransac_calibration.py
                  → writes data/T_ee_cam_ransac.npy
  2. Physically MOVE the ArUco board to a different spot on the table and hold
     it stationary there.
  3. Run this script. Move the arm through 8-10 different poses while keeping
     the board still; press 'c' to record each one. Press 's' to finish.

Why relocate? Validating against the same board location used for calibration
can let a systematic error in T_ee_cam partially cancel out. A fresh location
means any bias shows up as scatter across the new poses, so the reported RMS is
an honest measure of whether the transform generalises.

For every captured pose we reconstruct where the board sits in the base frame:
    T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
Because the board is physically fixed, a correct calibration makes all these
reconstructions land on essentially one point. The RMS spread of those points
is the validation metric. Both calibrations are evaluated on the SAME set of
validation poses so the comparison is apples-to-apples.

Requires the robot connected and the wrist camera open. Saves the collected
pairs to ../data/validation_data.npy.
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from transformation_calibration import ArucoBoardDetector
from lab_config import (
    ROBOT_IP,
    VALIDATION_MIN_POSES as MIN_POSES,
    VALIDATION_MAX_POSES as MAX_POSES,
    QUALITY_EXCELLENT_MM, QUALITY_GOOD_MM, QUALITY_ACCEPTABLE_MM,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'


def reconstruct_board_positions(T_ee_cam, pairs):
    """Return the board centre in the base frame for each pair (mm)."""
    positions = []
    for d in pairs:
        T_base_board = d['T_base_ee'] @ T_ee_cam @ d['T_cam_board']
        positions.append(T_base_board[:3, 3])
    return np.array(positions)


def report(name, positions):
    """Print mean/std/max-deviation summary for an array of 3D positions."""
    mean = np.mean(positions, axis=0)
    std = np.std(positions, axis=0)
    max_dev = float(np.max(np.linalg.norm(positions - mean, axis=1)))
    rms = float(np.sqrt(np.mean(np.sum((positions - mean) ** 2, axis=1))))

    print(f"\n{name}")
    print("-" * 50)
    print(f"  Poses evaluated : {len(positions)}")
    print(f"  Mean position   : ({mean[0]:8.2f}, {mean[1]:8.2f}, {mean[2]:8.2f}) mm")
    print(f"  Std dev (xyz)   : ({std[0]:6.2f}, {std[1]:6.2f}, {std[2]:6.2f}) mm")
    print(f"  Max deviation   : {max_dev:.2f} mm from mean")
    print(f"  RMS scatter     : {rms:.2f} mm")

    if max_dev < QUALITY_EXCELLENT_MM:
        grade = f"EXCELLENT (< {QUALITY_EXCELLENT_MM:g} mm)"
    elif max_dev < QUALITY_GOOD_MM:
        grade = f"GOOD (< {QUALITY_GOOD_MM:g} mm)"
    elif max_dev < QUALITY_ACCEPTABLE_MM:
        grade = f"ACCEPTABLE (< {QUALITY_ACCEPTABLE_MM:g} mm)"
    else:
        grade = (f"POOR (>= {QUALITY_ACCEPTABLE_MM:g} mm) - "
                 f"consider re-capturing with more variation")
    print(f"  Quality         : {grade}")
    return max_dev


def _load_transform(path_str, label):
    """Load a T_ee_cam .npy file, returning None with a warning if absent."""
    p = Path(path_str)
    if not p.exists():
        print(f"⚠ {label}: {p.name} not found — skipping. "
              f"Run the corresponding calibration first.")
        return None
    T = np.load(p)
    euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    print(f"  {label}: trans=({T[0,3]:7.2f}, {T[1,3]:7.2f}, {T[2,3]:7.2f}) mm  "
          f"rot=({euler[0]:6.2f}, {euler[1]:6.2f}, {euler[2]:6.2f}) deg")
    return T


def main():
    # Load whichever calibration results are available.
    T_normal = _load_transform(str(_DATA_DIR / 'T_ee_cam_normal.npy'),
                               "Normal (direct)")
    T_ransac = _load_transform(str(_DATA_DIR / 'T_ee_cam_ransac.npy'),
                               "RANSAC")

    if T_normal is None and T_ransac is None:
        raise FileNotFoundError(
            "Neither T_ee_cam_normal.npy nor T_ee_cam_ransac.npy found in data/. "
            "Run get_tranform_3D.py and/or ransac_calibration.py first."
        )

    print("\nMove the board to a NEW location and keep it STILL.")
    print(f"Then move the arm through {MIN_POSES}-{MAX_POSES} varied poses.")

    det = ArucoBoardDetector()
    if not det.start_camera():
        return
    det.calibration_data = []

    print("\nControls:")
    print(f"  'c' - Capture current pose (need {MIN_POSES}-{MAX_POSES})")
    print("  's' - Finish & compute RMS spread")
    print("  'q' - Quit without saving")
    print("=" * 60 + "\n")

    while True:
        ret, frame = det.cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        success, T_cam_board, rvec, tvec, _reproj = det.detect_board(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detector.detectMarkers(gray)
        frame = det.draw_detection(frame, rvec, tvec, ids, corners, success)

        n = len(det.calibration_data)
        cv2.putText(frame, f"Validation poses: {n}/{MAX_POSES}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.imshow('Calibration Validation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if n >= MAX_POSES:
                print(f"Already at {MAX_POSES}; press 's' to finish.")
            elif not success:
                print("Board not detected - try again.")
            else:
                T_base_ee, joints_deg, raw_pose = det.get_robot_end_effector_pose()
                if T_base_ee is None:
                    print("Failed to read robot pose")
                else:
                    det.calibration_data.append({
                        'T_base_ee': T_base_ee,          # FK-derived
                        'robot_joints': joints_deg,
                        'robot_pose_raw': raw_pose,
                        'T_cam_board': T_cam_board,
                        'timestamp': cv2.getTickCount(),
                    })
                    ee_pos = T_base_ee[:3, 3]
                    print(f"✓ Captured validation pose #{len(det.calibration_data)} "
                          f"EE=({ee_pos[0]:.1f},{ee_pos[1]:.1f},{ee_pos[2]:.1f}) mm")
        elif key == ord('s'):
            if n < MIN_POSES:
                print(f"Need at least {MIN_POSES} poses (have {n}).")
            else:
                break

    det.cap.release()
    cv2.destroyAllWindows()

    pairs = det.calibration_data
    if len(pairs) < MIN_POSES:
        print(f"\nAborted: only {len(pairs)} poses captured "
              f"(need {MIN_POSES}-to-{MAX_POSES}). No results saved.")
        return

    np.save(str(_DATA_DIR / 'validation_data.npy'), pairs)
    print(f"✓ Saved {len(pairs)} validation poses -> data/validation_data.npy")

    # Evaluate each available calibration on the SAME validation poses.
    results = {}
    if T_normal is not None:
        pos_n = reconstruct_board_positions(T_normal, pairs)
        results['Normal (direct)'] = (pos_n, report(
            "NORMAL CALIBRATION — reconstructed board position", pos_n))
    if T_ransac is not None:
        pos_r = reconstruct_board_positions(T_ransac, pairs)
        results['RANSAC'] = (pos_r, report(
            "RANSAC CALIBRATION — reconstructed board position", pos_r))

    # Side-by-side comparison table.
    if len(results) == 2:
        (pos_n, dev_n), (pos_r, dev_r) = list(results.values())
        rms_n = float(np.sqrt(np.mean(np.sum((pos_n - pos_n.mean(axis=0))**2, axis=1))))
        rms_r = float(np.sqrt(np.mean(np.sum((pos_r - pos_r.mean(axis=0))**2, axis=1))))
        print("\n" + "=" * 60)
        print("SIDE-BY-SIDE COMPARISON (same validation poses)")
        print("=" * 60)
        print(f"{'Metric':<25} {'Normal':>12} {'RANSAC':>12}")
        print("-" * 50)
        print(f"{'Max deviation (mm)':<25} {dev_n:>12.2f} {dev_r:>12.2f}")
        print(f"{'RMS scatter (mm)':<25} {rms_n:>12.2f} {rms_r:>12.2f}")
        print(f"{'Std X (mm)':<25} {np.std(pos_n,axis=0)[0]:>12.2f} "
              f"{np.std(pos_r,axis=0)[0]:>12.2f}")
        print(f"{'Std Y (mm)':<25} {np.std(pos_n,axis=0)[1]:>12.2f} "
              f"{np.std(pos_r,axis=0)[1]:>12.2f}")
        print(f"{'Std Z (mm)':<25} {np.std(pos_n,axis=0)[2]:>12.2f} "
              f"{np.std(pos_r,axis=0)[2]:>12.2f}")
        winner = 'RANSAC' if dev_r < dev_n else ('Normal' if dev_n < dev_r else 'Tie')
        diff = abs(dev_n - dev_r)
        print(f"\n  → {winner} is tighter by {diff:.2f} mm (max-deviation metric).")
    elif len(results) == 1:
        name, (pos, _) = list(results.items())[0]
        print(f"\nPer-axis spread for {name} (mm):")
        for axis, label in enumerate(['X', 'Y', 'Z']):
            vals = pos[:, axis]
            print(f"  {label}: min={vals.min():8.2f}  max={vals.max():8.2f}  "
                  f"range={np.ptp(vals):7.2f}")

    print("\nTip: large spread usually means the board moved during capture,")
    print("some frames were mis-detected, or T_ee_cam itself is inaccurate.")


if __name__ == "__main__":
    print(f"Connecting to UFACTORY Lite 6 at {ROBOT_IP}...")
    main()
