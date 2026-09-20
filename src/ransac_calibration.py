"""Robust eye-to-hand calibration with outlier rejection (RANSAC).

Setup: wrist-mounted camera observing a STATIC bench artifact (eye-to-hand). The
unknown is T_6_C (end-effector -> camera); a correct fit makes the reconstructed
artifact position in the base frame collapse to a single point.

Notation:
    base frame   : 0          end-effector frame : 6
    camera frame : C          ArUco artifact     : W

Loads captured (T_0_6, T_C_W) pairs from ../data/calibration_data.npy, filters
inconsistent poses via RANSAC over random pose subsets, refines by nonlinear
least squares, plots the result, and saves:
    ../data/T_6_C_ransac.npy             # the 4x4 transform
    ../data/T_6_C_ransac.txt             # quick matrix / translation / euler
    ../data/ransac_calibration_results.txt  # full report: params + consistency
                                            # + baseline comparison + refinement

Each subset hypothesis is solved with the shared self-contained eye-to-hand
solver (get_transform.solve_eye_to_hand), which uses the conjugation form A = X Bp X^-1
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from lab_config import (
    RANSAC_ITERATIONS, RANSAC_INLIER_THRESHOLD_MM, RANSAC_SEED,
    S_MAX_EXCELLENT_MM, S_MAX_GOOD_MM, S_MAX_ACCEPTABLE_MM,
)
from get_transform import resolve_T_0_6, solve_eye_to_hand

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'


class RobustHandEyeCalibrator:
    def __init__(self):
        self.calibration_data = []
        self.T_6_C = None
        self.inlier_mask = None

    # The eye-to-hand solver needs at least 4 poses: with only 3 poses there are
    # just 2 relative-rotation pairs, which cannot span 3D rotation space, so the
    # rotation stack is rank-deficient and the solve is rejected. Four poses give
    # 3 independent pairs -- the smallest well-posed set. Each RANSAC hypothesis
    # is built from a random 4-pose subset; sampling these small subsets and
    # scoring them globally against every pose is what makes this robust to
    # outliers.
    MIN_SAMPLE = 4

    def _solve_on_subset(self, subset, method_flag=None):
        """Solve T_6_C from a list of poses with the shared eye-to-hand solver.

        Returns (T_6_C, ok). Fails gracefully when the subset is degenerate
        (rank-deficient rotation stack), which is exactly what RANSAC must
        tolerate.
        """
        if len(subset) < self.MIN_SAMPLE:
            return None, False

        T_0_6_list = [resolve_T_0_6(d)[0] for d in subset]
        T_C_W_list = [np.asarray(d['T_C_W'], dtype=np.float64) for d in subset]
        return solve_eye_to_hand(T_0_6_list, T_C_W_list)

    def evaluate_consistency(self, T_6_C, data):
        """Mean std-dev of reconstructed artifact positions across poses."""
        art_positions = []
        for d in data:
            T_0_6, _src = resolve_T_0_6(d)
            T_0_W = T_0_6 @ T_6_C @ d['T_C_W']
            art_positions.append(T_0_W[:3, 3])

        art_positions = np.array(art_positions)
        return np.mean(np.std(art_positions, axis=0))

    def ransac_calibrate(self, n_iterations=2000, inlier_threshold_mm=10.0,
                         seed=None):
        """RANSAC over the captured pose pairs.

        Repeatedly samples a minimal subset of poses, solves a candidate T_6_C,
        and scores it by how tightly the reconstructed artifact position clusters
        across ALL poses. The lowest-scoring candidate wins; its inliers (poses
        whose reconstruction falls within inlier_threshold_mm of the consensus
        position) become the mask used for refinement.
        """
        rng = np.random.default_rng(seed)
        n = len(self.calibration_data)

        best_T = None
        best_score = float('inf')
        solved = 0

        for _ in range(n_iterations):
            idx = rng.choice(n, size=self.MIN_SAMPLE, replace=False)
            subset = [self.calibration_data[i] for i in idx]

            T_cand, ok = self._solve_on_subset(subset)
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

        # Assign inliers against the raw winning hypothesis; the nonlinear
        # refinement runs afterwards on the inlier subset (see STEP 3).
        positions = self._artifact_positions(self.calibration_data, best_T)
        center = np.median(positions, axis=0)
        dists = np.linalg.norm(positions - center, axis=1)
        self.inlier_mask = dists <= inlier_threshold_mm

        print(f"\n RANSAC iterations: {n_iterations}  (valid hypotheses: {solved})")
        print(f" Best consistency : {best_score:.2f} mm")
        print(f" Inliers          : {int(self.inlier_mask.sum())}/{n} "
              f"(threshold {inlier_threshold_mm:.1f} mm)")

        self.T_6_C = best_T
        self.best_score = best_score
        return best_T

    def baseline_calibrate(self):
        """Plain baseline: solve T_6_C on ALL poses, no RANSAC, no refinement.

        Uses the shared closed-form eye-to-hand solver on the full dataset and
        leaves the result untouched. This is the plain direct method the RANSAC
        route (closed form + outlier rejection + nonlinear refinement) is
        compared against. Returns (T_6_C, scores_dict).
        """
        all_data = self.calibration_data
        n = len(all_data)

        print(f"\n Solving on all {n} poses (no outlier rejection, no refinement)...")
        T_6_C, ok = self._solve_on_subset(all_data)
        if not ok or T_6_C is None:
            print("   Solver did not converge on the full dataset.")
            return None, {}

        score = self.evaluate_consistency(T_6_C, all_data)
        print(f"   Baseline (all poses, closed form only)  {score:7.2f} mm")

        return T_6_C, {'Baseline (all)': score}

    def _inlier_subset(self, use_inliers=True):
        """Poses passing the RANSAC inlier test (or all poses if none set)."""
        if use_inliers and self.inlier_mask is not None:
            return [d for i, d in enumerate(self.calibration_data) if self.inlier_mask[i]]
        return self.calibration_data

    def refine_calibration(self, initial_T_6_C=None, use_inliers=True):
        """Refine T_6_C by minimizing artifact-position spread via least squares."""
        data = self._inlier_subset(use_inliers)

        # Refining on a handful of inliers over-fits; if RANSAC kept fewer than 4
        # poses, widen to the full set so the optimizer has real leverage.
        if len(data) < 4:
            data = self.calibration_data

        if initial_T_6_C is None:
            initial_T_6_C = self.T_6_C
            if initial_T_6_C is None:
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
            T_6_C = param_to_transform(params)
            positions = np.array([
                (resolve_T_0_6(d)[0] @ T_6_C @ d['T_C_W'])[:3, 3] for d in data
            ])
            return (positions - np.mean(positions, axis=0)).flatten()

        initial_params = transform_to_param(initial_T_6_C)

        print("\n Refining calibration with nonlinear optimization...")
        result = least_squares(residuals, initial_params, method='trf', verbose=0)

        self.T_6_C_refined = param_to_transform(result.x)

        before_std = self.evaluate_consistency(initial_T_6_C, data)
        after_std = self.evaluate_consistency(self.T_6_C_refined, data)

        print(f"  Before refinement: {before_std:.2f} mm std")
        print(f"  After refinement:  {after_std:.2f} mm std")
        print(f"  Improvement: {before_std - after_std:.2f} mm")

        return self.T_6_C_refined

    def _artifact_positions(self, poses, T_6_C):
        """Reconstructed artifact positions in the robot base frame."""

        pts = [(resolve_T_0_6(d)[0] @ T_6_C @ d['T_C_W'])[:3, 3] for d in poses]
        return np.asarray(pts, dtype=float).reshape(-1, 3)

    def visualize_results(self, T_6_C):
        """Plot reconstructed artifact positions (3D, top-view, histogram)."""
        if T_6_C is None or np.asarray(T_6_C).shape != (4, 4):
            print("⚠ No valid T_6_C to visualize; skipping plots.")
            return

        if self.inlier_mask is not None:
            inlier_data = [d for i, d in enumerate(self.calibration_data) if self.inlier_mask[i]]
            outlier_data = [d for i, d in enumerate(self.calibration_data) if not self.inlier_mask[i]]
        else:
            inlier_data, outlier_data = self.calibration_data, []

        inlier_positions = self._artifact_positions(inlier_data, T_6_C)
        outlier_positions = self._artifact_positions(outlier_data, T_6_C) if outlier_data else np.array([])

        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(131, projection='3d')
        if len(inlier_positions) > 0:
            ax1.scatter(*inlier_positions.T, c='green', s=50, label='Inliers', alpha=0.7)
        if len(outlier_positions) > 0:
            ax1.scatter(*outlier_positions.T, c='red', s=30, label='Outliers', alpha=0.5)
        ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
        ax1.set_title('Artifact Position in Robot Base Frame')
        ax1.legend()

        ax2 = fig.add_subplot(132)
        if len(inlier_positions) > 0:
            ax2.scatter(inlier_positions[:, 0], inlier_positions[:, 1], c='green', s=50, alpha=0.7, label='Inliers')
        if len(outlier_positions) > 0:
            ax2.scatter(outlier_positions[:, 0], outlier_positions[:, 1], c='red', s=30, alpha=0.5, label='Outliers')
        ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)')
        ax2.set_title('Artifact Position (Top View)')
        ax2.legend(); ax2.grid(True)

        ax3 = fig.add_subplot(133)
        all_positions = np.vstack([inlier_positions, outlier_positions]) if len(outlier_positions) > 0 else inlier_positions
        if len(all_positions) > 0:
            center = np.mean(inlier_positions, axis=0) if len(inlier_positions) > 0 else np.mean(all_positions, axis=0)
            distances = np.linalg.norm(all_positions - center, axis=1)
            ax3.hist(distances, bins=20, color='blue', alpha=0.7)
            ax3.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.1f}mm')
            ax3.set_xlabel('Distance from Center (mm)'); ax3.set_ylabel('Frequency')
            ax3.set_title('Artifact Position Consistency')
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
            print("Artifact position in base frame (should be constant):")
            print(f"  Mean: ({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}) mm")
            print(f"  Std:  ({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}) mm")
            print(f"  Max deviation from mean: {np.max(np.linalg.norm(inlier_positions - mean, axis=1)):.1f} mm")

            s_max = float(np.max(std))
            if s_max < S_MAX_EXCELLENT_MM:
                print(f"\n✓ Calibration quality (s_max={s_max:.1f} mm): EXCELLENT")
            elif s_max < S_MAX_GOOD_MM:
                print(f"\n✓ Calibration quality (s_max={s_max:.1f} mm): GOOD")
            elif s_max < S_MAX_ACCEPTABLE_MM:
                print(f"\n⚠ Calibration quality (s_max={s_max:.1f} mm): ACCEPTABLE")
            else:
                print(f"\n✗ Calibration quality (s_max={s_max:.1f} mm): POOR - "
                      f"consider re-collecting data")

    def _transform_block(self, T_6_C):
        """Return the matrix / translation / euler lines shared by both files."""
        euler = R.from_matrix(T_6_C[:3, :3]).as_euler('xyz', degrees=True)
        lines = []
        lines.append("4x4 Transformation Matrix T_6_C (end-effector -> camera):\n")
        for row in T_6_C:
            lines.append(f"  {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}\n")
        lines.append("\nTranslation (mm):\n")
        lines.append(f"  X: {T_6_C[0,3]:.2f}\n")
        lines.append(f"  Y: {T_6_C[1,3]:.2f}\n")
        lines.append(f"  Z: {T_6_C[2,3]:.2f}\n")
        lines.append("\nRotation (degrees, xyz Euler):\n")
        lines.append(f"  Roll:  {euler[0]:.2f}\n")
        lines.append(f"  Pitch: {euler[1]:.2f}\n")
        lines.append(f"  Yaw:   {euler[2]:.2f}\n")
        return ''.join(lines)

    @staticmethod
    def _quality_grade(s_max_mm):
        if s_max_mm < S_MAX_EXCELLENT_MM:
            return f"EXCELLENT (s_max < {S_MAX_EXCELLENT_MM:g} mm)"
        if s_max_mm < S_MAX_GOOD_MM:
            return f"GOOD (s_max < {S_MAX_GOOD_MM:g} mm)"
        if s_max_mm < S_MAX_ACCEPTABLE_MM:
            return f"ACCEPTABLE (s_max < {S_MAX_ACCEPTABLE_MM:g} mm)"
        return f"POOR (s_max >= {S_MAX_ACCEPTABLE_MM:g} mm)"

    def save_results(self, T_6_C, filename='T_6_C_ransac.npy',
                     n_total=None, n_inliers=None, best_consistency=None,
                     baseline_scores=None, final_consistency=None,
                     refine_before=None, refine_after=None):
        """Save T_6_C plus a full results report into ../data/.

        Writes three files:
          - filename (.npy)                 : the raw 4x4 transform
          - filename with .txt              : quick matrix / translation / euler
          - ransac_calibration_results.txt  : full report with consistency,
                                               baseline comparison, refinement
        The optional keyword arguments carry the statistics computed during the
        run; any that are None are simply omitted from the report.
        """
        np.save(str(_DATA_DIR / filename), T_6_C)

        # --- Quick reference file (matrix only) -----------------------------
        txt_name = filename.replace('.npy', '.txt')
        with open(str(_DATA_DIR / txt_name), 'w') as f:
            f.write("Hand-Eye Calibration Result: T_6_C (RANSAC)\n")
            f.write("=" * 60 + "\n\n")
            f.write(self._transform_block(T_6_C))
        print(f"✓ Saved: {txt_name}")

        # --- Full results report --------------------------------------------
        report_path = str(_DATA_DIR / 'ransac_calibration_results.txt')
        with open(report_path, 'w') as f:
            f.write("ROBUST HAND-EYE CALIBRATION RESULTS (RANSAC)\n")
            f.write("=" * 60 + "\n\n")
            f.write("Worksheet notation: base=0, end-effector=6, camera=C,\n")
            f.write("ArUco artifact=W. Transform solved: T_6_C.\n\n")

            f.write("--- SOLVED TRANSFORM ---\n")
            f.write(self._transform_block(T_6_C))

            # Consistency metrics on the inlier subset.
            if final_consistency is not None:
                f.write("--- CONSISTENCY (reconstructed artifact position) ---\n")
                f.write(f"Final RANSAC consistency (inliers only): "
                        f"{final_consistency:.2f} mm (mean per-axis std)\n")
                if n_inliers is not None and n_total is not None:
                    f.write(f"Inliers used: {n_inliers}/{n_total} poses "
                            f"(threshold {RANSAC_INLIER_THRESHOLD_MM:g} mm)\n")
                if best_consistency is not None:
                    f.write(f"Best RANSAC hypothesis consistency (pre-refine): "
                            f"{best_consistency:.2f} mm\n")
                f.write(f"Quality grade: {self._quality_grade(final_consistency)}\n\n")

            # Baseline vs RANSAC comparison table.
            if baseline_scores:
                f.write("--- BASELINE (all poses, no rejection) vs RANSAC ---\n")
                f.write(f"{'Method':<14} {'Consistency (mm)':>18}\n")
                f.write("-" * 34 + "\n")
                for name, score in sorted(baseline_scores.items(),
                                          key=lambda x: x[1]):
                    f.write(f"{name:<14} {score:>18.2f}\n")
                if final_consistency is not None:
                    f.write(f"{'RANSAC (inl.)':<14} {final_consistency:>18.2f}\n")
                f.write("\n")

            # Nonlinear refinement detail.
            if refine_before is not None and refine_after is not None:
                f.write("--- NONLINEAR REFINEMENT (least squares) ---\n")
                f.write(f"Before refinement: {refine_before:.2f} mm std\n")
                f.write(f"After refinement:  {refine_after:.2f} mm std\n")
                f.write(f"Improvement:       {refine_before - refine_after:.2f} mm\n")

        print(f"✓ Saved: ransac_calibration_results.txt")


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
    T_6_C = calibrator.ransac_calibrate(
        n_iterations=RANSAC_ITERATIONS,
        inlier_threshold_mm=RANSAC_INLIER_THRESHOLD_MM,
        seed=RANSAC_SEED,
    )

    if T_6_C is not None:
        # Build the inlier subset for RANSAC-specific reporting.
        n_total = len(calibrator.calibration_data)
        n_inliers = int(calibrator.inlier_mask.sum()) if calibrator.inlier_mask is not None else n_total
        inlier_data = [d for i, d in enumerate(calibrator.calibration_data)
                       if calibrator.inlier_mask[i]] if calibrator.inlier_mask is not None \
                      else calibrator.calibration_data

        # ---- Baseline comparison ------------------------------------------
        # Baseline: all poses, no rejection.
        # RANSAC:   inlier subset only (the whole point of RANSAC is to trim).
        baseline_scores = {}
        print("\n" + "="*60)
        print("BASELINE vs RANSAC COMPARISON")
        print("="*60)
        T_baseline, baseline_scores = calibrator.baseline_calibrate()
        if T_baseline is not None:
            baseline_consistency = calibrator.evaluate_consistency(
                T_baseline, calibrator.calibration_data)
            ransac_consistency = calibrator.evaluate_consistency(
                T_6_C, inlier_data)
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
        # Consistency of the RAW RANSAC estimate on the inlier subset, captured
        # BEFORE refinement so the report can show how much the polish helped.
        refine_before = None
        try:
            refine_before = calibrator.evaluate_consistency(T_6_C, inlier_data)
        except Exception:
            pass
        T_6_C_refined = calibrator.refine_calibration(
            initial_T_6_C=T_6_C, use_inliers=True)

        # refine_calibration can return None (e.g. no usable initial guess).
        # Fall back to the raw RANSAC estimate so downstream steps always get
        # a real 4x4 transform instead of crashing on a None matmul.
        if T_6_C_refined is None:
            print("\n⚠ Refinement returned no result; using the raw RANSAC "
                  "estimate for visualization and saving.")
            T_6_C_refined = T_6_C

        # Report final consistency on the inlier subset.
        final_consistency = calibrator.evaluate_consistency(
            T_6_C_refined, inlier_data)
        refine_after = final_consistency
        print(f"\n  Final RANSAC consistency (inliers only): {final_consistency:.2f} mm")

        print("\n" + "="*60)
        print("STEP 4: Visualization")
        print("="*60)
        calibrator.visualize_results(T_6_C_refined)

        print("\n" + "="*60)
        print("STEP 5: Saving Results")
        print("="*60)
        calibrator.save_results(
            T_6_C_refined,
            n_total=n_total,
            n_inliers=n_inliers,
            best_consistency=getattr(calibrator, 'best_score', None),
            baseline_scores=baseline_scores,
            final_consistency=final_consistency,
            refine_before=refine_before,
            refine_after=refine_after,
        )

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        print("\nTo use this calibration in your robot code:")
        print("  T_6_C = np.load('<data>/T_6_C_ransac.npy')")
        print("  T_0_W = T_0_6 @ T_6_C @ T_C_W")
    else:
        print("\n✗ Calibration failed. Please check your data.")
