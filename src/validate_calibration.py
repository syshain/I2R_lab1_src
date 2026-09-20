"""Validate hand-eye calibrations with a RELOCATED artifact (live, keygated).

Notation:
    base frame   : 0          end-effector frame : 6
    camera frame : C          ArUco artifact     : W

Workflow:
  1. Complete BOTH calibrations first:
        - Normal:  transformation_calibration.py + get_transform.py
                   → writes data/T_6_C_normal.npy
        - RANSAC:  ransac_calibration.py
                   → writes data/T_6_C_ransac.npy
  2. Physically MOVE the ArUco artifact to a different spot on the table and
     hold it stationary there.
  3. Run this script. Move the arm through 8-10 different poses while keeping
      the artifact still; press SPACE to record each one. Press ESC to finish.

For every captured pose we reconstruct where the artifact sits in the base frame:
    T_0_W = T_0_6 @ T_6_C @ T_C_W
Because the artifact is physically fixed, a correct calibration makes all these
reconstructions land on essentially one point. The RMS spread of those points is
the validation metric. Both calibrations are evaluated on the SAME set of
validation poses so the comparison is apples-to-apples.

Requires the robot connected and the wrist camera open. Saves:
    ../data/validation_data.npy       # the captured (T_0_6, T_C_W) pairs
    ../data/validation_results.txt    # per-transform stats + side-by-side table
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from transformation_calibration import ArucoArtifactDetector, get_robot_end_effector_pose
from get_transform import resolve_T_0_6
from lab_config import (
    ROBOT_IP,
    VALIDATION_MIN_POSES as MIN_POSES,
    VALIDATION_MAX_POSES as MAX_POSES,
    S_MAX_EXCELLENT_MM, S_MAX_GOOD_MM, S_MAX_ACCEPTABLE_MM,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'


def reconstruct_artifact_positions(T_6_C, pairs):
    """Return the artifact origin in the base frame for each pair (mm)."""
    positions = []
    for d in pairs:
        T_0_6, _src = resolve_T_0_6(d)
        T_0_W = T_0_6 @ T_6_C @ d['T_C_W']
        positions.append(T_0_W[:3, 3])
    return np.array(positions)


def _stats(positions):
    """Compute the validation metrics for an array of 3D positions."""
    mean = np.mean(positions, axis=0)
    std = np.std(positions, axis=0)
    max_dev = float(np.max(np.linalg.norm(positions - mean, axis=1)))
    rms = float(np.sqrt(np.mean(np.sum((positions - mean) ** 2, axis=1))))
    return {'n': len(positions), 'mean': mean, 'std': std,
            'max_dev': max_dev, 'rms': rms}


def _quality_grade(s_max_mm):
    # Grade on s_max = max per-axis std of the reconstructed artifact position,
    # matching the worksheet's hand-eye quality bands.
    if s_max_mm < S_MAX_EXCELLENT_MM:
        return f"EXCELLENT (s_max < {S_MAX_EXCELLENT_MM:g} mm)"
    if s_max_mm < S_MAX_GOOD_MM:
        return f"GOOD (s_max < {S_MAX_GOOD_MM:g} mm)"
    if s_max_mm < S_MAX_ACCEPTABLE_MM:
        return f"ACCEPTABLE (s_max < {S_MAX_ACCEPTABLE_MM:g} mm)"
    return (f"POOR (s_max >= {S_MAX_ACCEPTABLE_MM:g} mm) - "
            f"consider re-capturing with more variation")


def report(name, s):
    """Pretty-print one transform's validation stats (dict from _stats)."""
    mean, std = s['mean'], s['std']
    print(f"\n{name}")
    print("-" * 50)
    print(f"  Poses evaluated : {s['n']}")
    print(f"  Mean position   : ({mean[0]:8.2f}, {mean[1]:8.2f}, {mean[2]:8.2f}) mm")
    print(f"  Std dev (xyz)   : ({std[0]:6.2f}, {std[1]:6.2f}, {std[2]:6.2f}) mm")
    print(f"  Max deviation   : {s['max_dev']:.2f} mm from mean")
    print(f"  RMS scatter     : {s['rms']:.2f} mm")
    print(f"  Quality         : {_quality_grade(s['max_dev'])}")


def _load_transform(path_str, label):
    """Load a T_6_C .npy file, returning None with a warning if absent."""
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


def _transform_lines(label, T):
    """Matrix / translation / euler lines for one transform."""
    euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    lines = [f"T_6_C — {label}:\n"]
    for row in T:
        lines.append(f"  {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}\n")
    lines.append("\nTranslation (mm):\n")
    lines.append(f"  X: {T[0,3]:.2f}\n")
    lines.append(f"  Y: {T[1,3]:.2f}\n")
    lines.append(f"  Z: {T[2,3]:.2f}\n")
    lines.append("\nRotation (degrees, xyz Euler):\n")
    lines.append(f"  Roll:  {euler[0]:.2f}\n")
    lines.append(f"  Pitch: {euler[1]:.2f}\n")
    lines.append(f"  Yaw:   {euler[2]:.2f}\n")
    return ''.join(lines)


def write_validation_report(transforms, results):
    """Write validation_results.txt covering every evaluated transform.

    `transforms` maps name -> T_6_C matrix; `results` maps name -> stats dict
    (from _stats). Both are written so a reader can reproduce the numbers from
    the saved matrices alone.
    """
    path = str(_DATA_DIR / 'validation_results.txt')
    with open(path, 'w') as f:
        f.write("HAND-EYE CALIBRATION VALIDATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Worksheet notation: base=0, end-effector=6, camera=C,\n")
        f.write("ArUco artifact=W. Metric: scatter of reconstructed artifact\n")
        f.write("position T_0_W = T_0_6 @ T_6_C @ T_C_W across relocated poses.\n\n")

        # Per-transform detail.
        for name, s in results.items():
            mean, std = s['mean'], s['std']
            f.write(f"--- {name.upper()} ---\n")
            if name in transforms:
                f.write(_transform_lines(name, transforms[name]))
            f.write(f"Poses evaluated : {s['n']}\n")
            f.write(f"Mean position   : ({mean[0]:8.2f}, {mean[1]:8.2f}, "
                    f"{mean[2]:8.2f}) mm\n")
            f.write(f"Std dev (xyz)   : ({std[0]:6.2f}, {std[1]:6.2f}, "
                    f"{std[2]:6.2f}) mm\n")
            f.write(f"Max deviation   : {s['max_dev']:.2f} mm from mean\n")
            f.write(f"RMS scatter     : {s['rms']:.2f} mm\n")
            f.write(f"Quality         : {_quality_grade(s['max_dev'])}\n\n")

        # Side-by-side comparison table.
        names = list(results.keys())
        if len(names) == 2:
            a, b = names
            f.write("--- SIDE-BY-SIDE COMPARISON (same validation poses) ---\n")
            f.write(f"{'Metric':<25} {a:>14} {b:>14}\n")
            f.write("-" * 55 + "\n")
            sa, sb = results[a], results[b]
            f.write(f"{'Max deviation (mm)':<25} {sa['max_dev']:>14.2f} "
                    f"{sb['max_dev']:>14.2f}\n")
            f.write(f"{'RMS scatter (mm)':<25} {sa['rms']:>14.2f} "
                    f"{sb['rms']:>14.2f}\n")
            for axis, label in enumerate(['X', 'Y', 'Z']):
                f.write(f"{'Std ' + label + ' (mm)':<25} {sa['std'][axis]:>14.2f} "
                        f"{sb['std'][axis]:>14.2f}\n")
            winner = b if sb['max_dev'] < sa['max_dev'] else \
                     (a if sa['max_dev'] < sb['max_dev'] else 'Tie')
            diff = abs(sa['max_dev'] - sb['max_dev'])
            f.write(f"\nWinner (tighter max-deviation): {winner} by {diff:.2f} mm\n")

    print(f"\n✓ Saved: validation_results.txt")


def main():
    # Load whichever calibration results are available.
    T_normal = _load_transform(str(_DATA_DIR / 'T_6_C_normal.npy'),
                               "Normal (direct)")
    T_ransac = _load_transform(str(_DATA_DIR / 'T_6_C_ransac.npy'),
                               "RANSAC")

    if T_normal is None and T_ransac is None:
        raise FileNotFoundError(
            "Neither T_6_C_normal.npy nor T_6_C_ransac.npy found in data/. "
            "Run get_transform.py and/or ransac_calibration.py first."
        )

    print("\nMove the ArUco artifact to a NEW location and keep it STILL.")
    print(f"Then move the arm through {MIN_POSES}-{MAX_POSES} varied poses.")

    det = ArucoArtifactDetector()
    if not det.start_camera():
        return
    det.calibration_data = []

    print("\nControls:")
    print(f"  SPACE - Capture current pose (need {MIN_POSES}-{MAX_POSES})")
    print("  ESC   - Finish & compute RMS spread")
    print("=" * 60 + "\n")

    pose_id = 1
    while True:
        ret, frame = det.cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        success, T_C_W, rvec, tvec, _reproj = det.detect_artifact(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detector.detectMarkers(gray)
        frame = det.draw_detection(frame, rvec, tvec, ids, corners, success)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        n = len(det.calibration_data)
        # The shared detector overlay prints X/Y/Z artifact coordinates below
        # the "Artifact Detected" line; they are not needed here, so blank them
        # out and put the pose count + key hints in their place instead.
        if success:
            cv2.rectangle(frame, (0, 45), (160, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Poses captured: {n}/{MAX_POSES}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 2)
        cv2.putText(frame, "SPACE: capture    ESC: finish & save",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Calibration Validation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if n >= MAX_POSES:
                print(f"Already at {MAX_POSES}; press ESC to finish.")
            elif not success:
                print("Artifact not detected - try again.")
            else:
                T_0_6, joints_deg, raw_pose = get_robot_end_effector_pose(det.arm)
                if T_0_6 is None:
                    print("Failed to read robot pose")
                else:
                    det.calibration_data.append({
                        'pose_id': pose_id,
                        'T_0_6': T_0_6,           # FK-derived
                        'robot_joints': joints_deg,
                        'robot_pose_raw': raw_pose,
                        'T_C_W': T_C_W,
                        'timestamp': cv2.getTickCount(),
                    })
                    ee_pos = T_0_6[:3, 3]
                    print(f"✓ Captured validation pose #{pose_id} "
                          f"EE=({ee_pos[0]:.1f},{ee_pos[1]:.1f},{ee_pos[2]:.1f}) mm")
                    pose_id += 1

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
    transforms = {}   # name -> T_6_C matrix (for the report file)
    results = {}      # name -> stats dict (from _stats)
    if T_normal is not None:
        pos_n = reconstruct_artifact_positions(T_normal, pairs)
        s_n = _stats(pos_n)
        transforms['Normal (direct)'] = T_normal
        results['Normal (direct)'] = s_n
        report("NORMAL CALIBRATION — reconstructed artifact position", s_n)
    if T_ransac is not None:
        pos_r = reconstruct_artifact_positions(T_ransac, pairs)
        s_r = _stats(pos_r)
        transforms['RANSAC'] = T_ransac
        results['RANSAC'] = s_r
        report("RANSAC CALIBRATION — reconstructed artifact position", s_r)

    # Print side-by-side comparison to console.
    if len(results) == 2:
        a, b = list(results.keys())
        sa, sb = results[a], results[b]
        print("\n" + "=" * 60)
        print("SIDE-BY-SIDE COMPARISON (same validation poses)")
        print("=" * 60)
        print(f"{'Metric':<25} {a[:12]:>12} {b[:12]:>12}")
        print("-" * 50)
        print(f"{'Max deviation (mm)':<25} {sa['max_dev']:>12.2f} {sb['max_dev']:>12.2f}")
        print(f"{'RMS scatter (mm)':<25} {sa['rms']:>12.2f} {sb['rms']:>12.2f}")
        for axis, label in enumerate(['X', 'Y', 'Z']):
            print(f"{'Std ' + label + ' (mm)':<25} {sa['std'][axis]:>12.2f} "
                  f"{sb['std'][axis]:>12.2f}")
        winner = b if sb['max_dev'] < sa['max_dev'] else \
                 (a if sa['max_dev'] < sb['max_dev'] else 'Tie')
        diff = abs(sa['max_dev'] - sb['max_dev'])
        print(f"\n  → {winner} is tighter by {diff:.2f} mm (max-deviation metric).")

    # Persist everything to validation_results.txt.
    write_validation_report(transforms, results)


if __name__ == "__main__":
    print(f"Connecting to UFACTORY Lite 6 at {ROBOT_IP}...")
    main()
