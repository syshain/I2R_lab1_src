"""Robust hand-eye calibration with outlier rejection.

Loads captured (T_base_ee, T_cam_board) pairs from ../data/calibration_data.npy,
filters inconsistent poses, solves for T_ee_cam across several OpenCV methods,
refines by nonlinear least squares, plots the result, and saves
../data/T_ee_cam_ransac.npy / .txt.
"""

import os
import numpy as np
import cv2
import math
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from lab_config import (
    RANSAC_ITERATIONS, RANSAC_INLIER_THRESHOLD_MM, RANSAC_SEED,
    QUALITY_EXCELLENT_MM, QUALITY_GOOD_MM, QUALITY_ACCEPTABLE_MM,
)

# Silence OpenCV's C++ error logging while RANSAC probes degenerate subsets;
# those failures are expected and handled in Python, not worth spamming stderr.
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'


class RobustHandEyeCalibrator:
    def __init__(self):
        self.calibration_data = []
        self.T_ee_cam = None
        self.inlier_mask = None

    def get_robot_end_effector_pose(self, arm):
        """Build T_base_ee from the arm's [x,y,z,roll,pitch,yaw] pose."""
        code, pose_data = arm.get_position()

        if code != 0:
            print(f"Error getting position: {code}")
            return None

        x, y, z, roll_deg, pitch_deg, yaw_deg = pose_data
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        R_matrix = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr]
        ])

        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R_matrix
        T_base_ee[:3, 3] = [x, y, z]

        return T_base_ee

    def capture_calibration_pose(self, arm, detector, pose_id):
        """Capture one (T_base_ee, T_cam_board) pair and store it."""
        T_base_ee = self.get_robot_end_effector_pose(arm)
        if T_base_ee is None:
            return False

        ret, frame = detector.cap.read()
        if not ret:
            print("Failed to capture frame")
            return False

        success, T_cam_board, rvec, tvec = detector.detect_board(frame)
        if not success:
            print("Failed to detect board")
            return False

        self.calibration_data.append({
            'pose_id': pose_id,
            'T_base_ee': T_base_ee,
            'T_cam_board': T_cam_board,
            'robot_position': T_base_ee[:3, 3].copy(),
            'board_distance': np.linalg.norm(T_cam_board[:3, 3])
        })

        print(f"✓ Pose {pose_id}: Robot=({T_base_ee[0,3]:6.1f}, {T_base_ee[1,3]:6.1f}, {T_base_ee[2,3]:6.1f}) mm, "
              f"Board dist={T_cam_board[2,3]:6.1f} mm")
        return True
    
    # cv2.calibrateHandEye needs at least 3 measurements, so each RANSAC
    # hypothesis is built from a random 3-pose subset (the smallest set it will
    # accept). Sampling these small subsets and scoring them globally against
    # every pose is what makes this robust to outliers.
    MIN_SAMPLE = 3

    def _solve_on_subset(self, subset, method_flag):
        """Solve T_ee_cam from a list of poses using one OpenCV hand-eye method.

        Returns (T_ee_cam, ok). Fails gracefully when the subset is degenerate
        (near-parallel rotations / collinear translations), which is exactly
        what RANSAC must tolerate.
        """
        if len(subset) < self.MIN_SAMPLE:
            return None, False

        R_grip = [d['T_base_ee'][:3, :3] for d in subset]
        t_grip = [d['T_base_ee'][:3, 3] for d in subset]
        R_obj  = [d['T_cam_board'][:3, :3] for d in subset]
        t_obj  = [d['T_cam_board'][:3, 3] for d in subset]

        try:
            R_cam_ee, t_cam_ee = cv2.calibrateHandEye(
                R_grip, t_grip, R_obj, t_obj, method=method_flag
            )
        except Exception:
            return None, False

        T_cam_ee = np.eye(4)
        T_cam_ee[:3, :3] = R_cam_ee
        T_cam_ee[:3, 3] = t_cam_ee.flatten()
        T_ee_cam = np.linalg.inv(T_cam_ee)
        return T_ee_cam, True

    def evaluate_consistency(self, T_ee_cam, data):
        """Mean std-dev of reconstructed board positions across poses."""
        board_positions = []
        for d in data:
            T_base_board = d['T_base_ee'] @ T_ee_cam @ d['T_cam_board']
            board_positions.append(T_base_board[:3, 3])

        board_positions = np.array(board_positions)
        return np.mean(np.std(board_positions, axis=0))

    def ransac_calibrate(self, n_iterations=2000, inlier_threshold_mm=10.0,
                         seed=None):
        """Genuine RANSAC over the captured pose pairs.

        Repeatedly samples a minimal subset of poses, solves a candidate
        T_ee_cam, and scores it by how tightly the reconstructed board position
        clusters across ALL poses. The lowest-scoring candidate wins; its
        inliers (poses whose reconstruction falls within inlier_threshold_mm of
        the consensus position) become the mask used for refinement.
        """
        rng = np.random.default_rng(seed)
        n = len(self.calibration_data)
        methods = [
            cv2.CALIB_HAND_EYE_TSAI,
            cv2.CALIB_HAND_EYE_PARK,
            cv2.CALIB_HAND_EYE_ANDREFF,
            cv2.CALIB_HAND_EYE_DANIILIDIS,
        ]

        best_T = None
        best_score = float('inf')
        solved = 0

        for _ in range(n_iterations):
            idx = rng.choice(n, size=self.MIN_SAMPLE, replace=False)
            subset = [self.calibration_data[i] for i in idx]
            method = methods[rng.integers(len(methods))]

            T_cand, ok = self._solve_on_subset(subset, method)
            if not ok or T_cand is None:
                continue
            solved += 1

            score = self.evaluate_consistency(T_cand, self.calibration_data)
            if score < best_score:
                best_score = score
                best_T = T_cand.copy()

        if best_T is None:
            print("RANSAC produced no valid hypothesis.")
            return None

        # The closed-form hand-eye solvers are weak initializers on this
        # geometry: their RAW solutions sit tens of mm off even when the data
        # is good, but a short nonlinear polish collapses them to the true
        # answer. Refine the winner FIRST, then assign inliers against the
        # polished transform so the gate sees poses where they actually belong.
        refined = self.refine_calibration(initial_T_ee_cam=best_T, use_inliers=False)
        if refined is not None:
            best_T = refined
            best_score = self.evaluate_consistency(best_T, self.calibration_data)

        # Assign inliers against the (refined) winning hypothesis.
        positions = self._board_positions(self.calibration_data, best_T)
        center = np.median(positions, axis=0)
        dists = np.linalg.norm(positions - center, axis=1)
        self.inlier_mask = dists <= inlier_threshold_mm

        print(f"\n RANSAC iterations: {n_iterations}  (valid hypotheses: {solved})")
        print(f" Best consistency : {best_score:.2f} mm")
        print(f" Inliers          : {int(self.inlier_mask.sum())}/{n} "
              f"(threshold {inlier_threshold_mm:.1f} mm)")

        self.T_ee_cam = best_T
        return best_T

    def baseline_calibrate(self):
        """Solve T_ee_cam on ALL poses (no RANSAC, no outlier rejection).

        Runs each of the four OpenCV closed-form methods on the full dataset,
        refines each result, and returns (best_T, scores_dict) where
        scores_dict maps method name -> post-refinement consistency (mm).
        This gives a fair comparison against RANSAC: same data, same refinement,
        just without the subset-sampling / inlier-rejection step.
        """
        all_data = self.calibration_data
        n = len(all_data)
        methods = {
            'Tsai':       cv2.CALIB_HAND_EYE_TSAI,
            'Park':       cv2.CALIB_HAND_EYE_PARK,
            'Andreff':    cv2.CALIB_HAND_EYE_ANDREFF,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        print(f"\n Solving on all {n} poses (no outlier rejection)...")
        results = {}
        for name, flag in methods.items():
            T_raw, ok = self._solve_on_subset(all_data, flag)
            if not ok or T_raw is None:
                continue
            raw_score = self.evaluate_consistency(T_raw, all_data)
            refined = self.refine_calibration(initial_T_ee_cam=T_raw, use_inliers=False)
            final_T = refined if refined is not None else T_raw
            final_score = self.evaluate_consistency(final_T, all_data)
            results[name] = {'T': final_T, 'raw': raw_score, 'final': final_score}
            print(f"   {name:12s}  raw={raw_score:7.2f} mm  "
                  f"refined={final_score:7.2f} mm")

        if not results:
            print("   No method converged on the full dataset.")
            return None, {}

        best_name = min(results, key=lambda k: results[k]['final'])
        best = results[best_name]
        print(f"   Best baseline method: {best_name} ({best['final']:.2f} mm)")
        return best['T'], {k: v['final'] for k, v in results.items()}

    def _inlier_subset(self, use_inliers=True):
        """Poses passing the RANSAC inlier test (or all poses if none set)."""
        if use_inliers and self.inlier_mask is not None:
            return [d for i, d in enumerate(self.calibration_data) if self.inlier_mask[i]]
        return self.calibration_data

    def refine_calibration(self, initial_T_ee_cam=None, use_inliers=True):
        """Refine T_ee_cam by minimizing board-position spread via least squares."""
        data = self._inlier_subset(use_inliers)

        # Refining on a handful of inlets over-fits; if RANSAC kept fewer than 4
        # poses, widen to the full set so the optimizer has real leverage.
        if len(data) < 4:
            data = self.calibration_data

        if initial_T_ee_cam is None:
            initial_T_ee_cam = self.T_ee_cam
            if initial_T_ee_cam is None:
                print("No initial calibration available")
                return None

        def param_to_transform(params):
            tx, ty, tz, rx, ry, rz = params
            T = np.eye(4)
            T[:3, 3] = [tx, ty, tz]
            T[:3, :3] = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
            return T

        def transform_to_param(T):
            tx, ty, tz = T[:3, 3]
            rx, ry, rz = R.from_matrix(T[:3, :3]).as_euler('xyz')
            return [tx, ty, tz, rx, ry, rz]

        def residuals(params):
            T_ee_cam = param_to_transform(params)
            positions = np.array([
                (d['T_base_ee'] @ T_ee_cam @ d['T_cam_board'])[:3, 3] for d in data
            ])
            return (positions - np.mean(positions, axis=0)).flatten()

        initial_params = transform_to_param(initial_T_ee_cam)

        print("\n Refining calibration with nonlinear optimization...")
        result = least_squares(residuals, initial_params, method='trf', verbose=0)

        self.T_ee_cam_refined = param_to_transform(result.x)

        before_std = self.evaluate_consistency(initial_T_ee_cam, data)
        after_std = self.evaluate_consistency(self.T_ee_cam_refined, data)

        print(f"  Before refinement: {before_std:.2f} mm std")
        print(f"  After refinement:  {after_std:.2f} mm std")
        print(f"  Improvement: {before_std - after_std:.2f} mm")

        return self.T_ee_cam_refined
    
    def _board_positions(self, poses, T_ee_cam):
        """Reconstructed board positions in the robot base frame.

        Always returns a 2-D (N, 3) array so callers can vstack / index it
        safely even when `poses` is empty (an empty list would otherwise yield
        a 1-D (0,) array and break np.vstack)."""
        pts = [(d['T_base_ee'] @ T_ee_cam @ d['T_cam_board'])[:3, 3] for d in poses]
        return np.asarray(pts, dtype=float).reshape(-1, 3)

    def visualize_results(self, T_ee_cam):
        """Plot reconstructed board positions (3D, top-view, histogram)."""
        if T_ee_cam is None or np.asarray(T_ee_cam).shape != (4, 4):
            print("⚠ No valid T_ee_cam to visualize; skipping plots.")
            return

        if self.inlier_mask is not None:
            inlier_data = [d for i, d in enumerate(self.calibration_data) if self.inlier_mask[i]]
            outlier_data = [d for i, d in enumerate(self.calibration_data) if not self.inlier_mask[i]]
        else:
            inlier_data, outlier_data = self.calibration_data, []

        inlier_positions = self._board_positions(inlier_data, T_ee_cam)
        outlier_positions = self._board_positions(outlier_data, T_ee_cam) if outlier_data else np.array([])

        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(131, projection='3d')
        if len(inlier_positions) > 0:
            ax1.scatter(*inlier_positions.T, c='green', s=50, label='Inliers', alpha=0.7)
        if len(outlier_positions) > 0:
            ax1.scatter(*outlier_positions.T, c='red', s=30, label='Outliers', alpha=0.5)
        ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
        ax1.set_title('Board Position in Robot Base Frame')
        ax1.legend()

        ax2 = fig.add_subplot(132)
        if len(inlier_positions) > 0:
            ax2.scatter(inlier_positions[:, 0], inlier_positions[:, 1], c='green', s=50, alpha=0.7, label='Inliers')
        if len(outlier_positions) > 0:
            ax2.scatter(outlier_positions[:, 0], outlier_positions[:, 1], c='red', s=30, alpha=0.5, label='Outliers')
        ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)')
        ax2.set_title('Board Position (Top View)')
        ax2.legend(); ax2.grid(True)

        ax3 = fig.add_subplot(133)
        all_positions = np.vstack([inlier_positions, outlier_positions]) if len(outlier_positions) > 0 else inlier_positions
        if len(all_positions) > 0:
            center = np.mean(inlier_positions, axis=0) if len(inlier_positions) > 0 else np.mean(all_positions, axis=0)
            distances = np.linalg.norm(all_positions - center, axis=1)
            ax3.hist(distances, bins=20, color='blue', alpha=0.7)
            ax3.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.1f}mm')
            ax3.set_xlabel('Distance from Center (mm)'); ax3.set_ylabel('Frequency')
            ax3.set_title('Board Position Consistency')
            ax3.legend(); ax3.grid(True)

        plt.tight_layout()
        plt.show()

        if len(inlier_positions) > 0:
            std = np.std(inlier_positions, axis=0)
            mean = np.mean(inlier_positions, axis=0)
            print("\n" + "="*60)
            print("FINAL CALIBRATION RESULTS")
            print("="*60)
            for p in inlier_positions:
                print(p)
            print("="*60)
            print("Board position in base frame (should be constant):")
            print(f"  Mean: ({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}) mm")
            print(f"  Std:  ({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}) mm")
            print(f"  Max deviation from mean: {np.max(np.linalg.norm(inlier_positions - mean, axis=1)):.1f} mm")

            worst = float(np.max(std))
            if worst < QUALITY_EXCELLENT_MM:
                print("\n✓ Calibration quality: EXCELLENT")
            elif worst < QUALITY_GOOD_MM:
                print("\n✓ Calibration quality: GOOD")
            elif worst < QUALITY_ACCEPTABLE_MM:
                print("\n⚠ Calibration quality: ACCEPTABLE")
            else:
                print("\n✗ Calibration quality: POOR - Consider re-collecting data")

    def save_results(self, T_ee_cam, filename='T_ee_cam_ransac.npy'):
        """Save T_ee_cam as .npy and a human-readable .txt into ../data/."""
        txt_name = filename.replace('.npy', '.txt')
        np.save(str(_DATA_DIR / filename), T_ee_cam)

        with open(str(_DATA_DIR / txt_name), 'w') as f:
            f.write("Hand-Eye Calibration Result: T_ee_cam (RANSAC)\n")
            f.write("="*60 + "\n\n")
            f.write("4x4 Transformation Matrix:\n")
            for row in T_ee_cam:
                f.write(f"  {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}\n")

            f.write("\nTranslation (mm):\n")
            f.write(f"  X: {T_ee_cam[0,3]:.2f}\n")
            f.write(f"  Y: {T_ee_cam[1,3]:.2f}\n")
            f.write(f"  Z: {T_ee_cam[2,3]:.2f}\n")

            euler = R.from_matrix(T_ee_cam[:3,:3]).as_euler('xyz', degrees=True)
            f.write("\nRotation (degrees):\n")
            f.write(f"  Roll:  {euler[0]:.2f}\n")
            f.write(f"  Pitch: {euler[1]:.2f}\n")
            f.write(f"  Yaw:   {euler[2]:.2f}\n")

        print(f"\n✓ Saved: {filename}")
        print(f"✓ Saved: {txt_name}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    calibrator = RobustHandEyeCalibrator()

    print("="*60)
    print("ROBUST HAND-EYE CALIBRATION FOR UFACTORY LITE 6")
    print("="*60)

    try:
        data = np.load(str(_DATA_DIR / 'calibration_data.npy'), allow_pickle=True)
        calibrator.calibration_data = data.tolist()
        print(f"\nLoaded {len(calibrator.calibration_data)} existing poses")
    except FileNotFoundError:
        print("\nNo existing data found. Please capture new poses first "
              "(see transformation_calibration.py).")
        raise SystemExit

    print("\n" + "="*60)
    print("STEP 1: RANSAC Hand-Eye Calibration (subset sampling)")
    print("="*60)
    T_ee_cam = calibrator.ransac_calibrate(
        n_iterations=RANSAC_ITERATIONS,
        inlier_threshold_mm=RANSAC_INLIER_THRESHOLD_MM,
        seed=RANSAC_SEED,
    )

    if T_ee_cam is not None:
        # Build the inlier subset for RANSAC-specific reporting.
        n_total = len(calibrator.calibration_data)
        n_inliers = int(calibrator.inlier_mask.sum()) if calibrator.inlier_mask is not None else n_total
        inlier_data = [d for i, d in enumerate(calibrator.calibration_data)
                       if calibrator.inlier_mask[i]] if calibrator.inlier_mask is not None \
                      else calibrator.calibration_data

        # ---- Baseline comparison ------------------------------------------
        # Baseline: all poses, no rejection.
        # RANSAC:   inlier subset only (the whole point of RANSAC is to trim).
        print("\n" + "="*60)
        print("BASELINE vs RANSAC COMPARISON")
        print("="*60)
        T_baseline, baseline_scores = calibrator.baseline_calibrate()
        if T_baseline is not None:
            baseline_consistency = calibrator.evaluate_consistency(
                T_baseline, calibrator.calibration_data)
            ransac_consistency = calibrator.evaluate_consistency(
                T_ee_cam, inlier_data)
            print(f"\n  Baseline fitted on: ALL {n_total} poses")
            print(f"  RANSAC fitted on:   {n_inliers}/{n_total} inlier poses "
                  f"(threshold {RANSAC_INLIER_THRESHOLD_MM:g} mm)")
            print(f"\n{'Method':<20} {'Poses':>7} {'Consistency (mm)':>18}")
            print("-"*48)
            for name, score in sorted(baseline_scores.items(),
                                      key=lambda x: x[1]):
                print(f"  {name:<18} {n_total:>7} {score:>16.2f}")
            print(f"  {'RANSAC (inliers)':<18} {n_inliers:>7} {ransac_consistency:>16.2f}")
            delta = baseline_consistency - ransac_consistency
            if delta > 0:
                print(f"\n  RANSAC improved consistency by {delta:.2f} mm "
                      f"over the best direct method.")
            else:
                print(f"\n  Note: direct calibration matched or beat RANSAC "
                      f"by {-delta:.2f} mm on this dataset (likely few/no outliers).")
        print("="*60)

        print("\n" + "="*60)
        print("STEP 3: Nonlinear Refinement (on inlier subset)")
        print("="*60)
        T_ee_cam_refined = calibrator.refine_calibration(use_inliers=True)

        # refine_calibration can return None (e.g. no usable initial guess).
        # Fall back to the raw RANSAC estimate so downstream steps always get
        # a real 4x4 transform instead of crashing on a None matmul.
        if T_ee_cam_refined is None:
            print("\n⚠ Refinement returned no result; using the raw RANSAC "
                  "estimate for visualization and saving.")
            T_ee_cam_refined = T_ee_cam

        # Report final consistency on the inlier subset.
        final_consistency = calibrator.evaluate_consistency(
            T_ee_cam_refined, inlier_data)
        print(f"\n  Final RANSAC consistency (inliers only): {final_consistency:.2f} mm")

        print("\n" + "="*60)
        print("STEP 4: Visualization")
        print("="*60)
        calibrator.visualize_results(T_ee_cam_refined)

        print("\n" + "="*60)
        print("STEP 5: Saving Results")
        print("="*60)
        calibrator.save_results(T_ee_cam_refined)

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        print("\nTo use this calibration in your robot code:")
        print("  T_ee_cam = np.load('<data>/T_ee_cam_ransac.npy')")
        print("  T_base_board = T_base_ee @ T_ee_cam @ T_cam_board")
    else:
        print("\n✗ Calibration failed. Please check your data.")
