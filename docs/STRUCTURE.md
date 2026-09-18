# Lab 1 — Repository Structure

This file documents the canonical folder/module template used by this lab. Labs 2 and 3 follow the same convention, so a student who learns the layout here can navigate any of the three labs identically.

Lab 1 is **camera + hand-eye calibration** in Python: calibrate the wrist camera intrinsics from a chessboard, capture paired robot/artifact poses, then solve for the end-effector-to-camera transform `T_6_C`. The active pipeline follows the shared `src/` template below; earlier exploratory scripts and the MATLAB forward-kinematics part are frozen provenance under `reference/`.

## Notation

All transforms use the worksheet's frame labels:

| Symbol | Meaning |
|--------|---------|
| `0`    | Robot base frame |
| `6`    | End-effector (gripper) frame |
| `C`    | Wrist camera frame |
| `W`    | ArUco artifact frame |
| `board`| Chessboard (camera-calibration target) — distinct from the ArUco artifact |

Key transforms:
- `T_0_6` : base → end-effector (from FK on joint angles).
- `T_C_W` : camera → ArUco artifact (from PnP on detected markers).
- `T_6_C` : end-effector → camera (the unknown solved by hand-eye calibration).
- `T_0_W = T_0_6 @ T_6_C @ T_C_W` : reconstructed artifact position in the base frame; a correct calibration makes this constant across poses.

**Formulation: eye-to-hand.** The wrist camera observes a *static* bench artifact, so the loop closes as `T_0_W = T_0_6(i) @ X @ T_C_W(i)` with `X = T_6_C` constant. This is the **eye-to-hand** problem (not eye-in-hand). The solver exploits the conjugation relation `A_i = X·B_p,i·X⁻¹`, where `A_i` is the base-frame relative motion between consecutive poses and `B_p,i` the camera-frame relative motion — rotation solved by least squares on stacked rotvecs, translation by a linear system. See `solve_eye_to_hand()` in `get_transform.py`.

Two physical artifacts are used:
- **Chessboard** — printed checker pattern for camera intrinsics (`camera_calibration.py`).
- **ArUco artifact** — rigid plate carrying ArUco markers for hand-eye calibration (`transformation_calibration.py`).

## Top-level layout

```
Lab_1/
├── src/          # Active Python code. One flat directory of single-responsibility scripts.
├── data/         # Calibration artefacts (.npy/.npz/.txt) + captured images. Produced/consumed at runtime.
├── reference/    # Frozen provenance, never imported by src/:
│   ├── calib_3D.py                 # Superseded 3-D calibration attempt.
│   ├── hand_eye_solver.py          # Earlier hand-eye solver, replaced by get_transform.py.
│   ├── camera_calibration.pdf      # Original camera-calibration notes.
│   └── MATLAB_FK/                  # Part A forward kinematics (fk_lite6.m, position_ufactory_modifiedDH.mlx).
├── docs/         # This structure guide.
├── requirements.txt
└── commit-log.md                   # Human-readable version history.
```

Rules that make the template portable across labs:

1. **`src/` is the only place active Python code lives.** It is a *flat* directory — no nested packages, no `__init__.py`. Scripts may import shared helpers from sibling modules (`fk_lite6`, `lab_config`, `robot_io`, `get_transform.resolve_T_0_6`) but exchange calibration results through files in `data/`. This keeps each stage independently runnable.
2. **`data/` holds binary calibration output and captured images, not source.** Files here are produced by one stage and consumed read-only by the next. Paths into `data/` are resolved relative to the repo via `_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'` in each script, never hardcoded absolute paths — so a script runs correctly no matter what the current working directory is.
3. **`reference/` is for provenance, not execution.** Code archived here (the superseded 3-D calibration, the earlier hand-eye solver, the MATLAB FK part) is where a current design decision came from. It is intentionally *not* on the Python import path so students cannot accidentally depend on superseded logic.

## Module roles (Python pipeline)

The stages run in order; each reads the previous stage's output from `data/` and writes its own.

| Script | Owns | Produces in `data/` |
|--------|------|---------------------|
| `image_capture.py` | Grabs chessboard frames from the USB camera on demand (SPACE to capture, ESC to finish). | `calib_NNN.jpg` |
| `camera_calibration.py` | Intrinsics + distortion from the chessboard images (`findChessboardCorners` + `calibrateCamera`). | `camera_matrix.npy`, `dist_coeffs.npy`, `calibration_results.npz`, `calibration_parameters.txt` |
| `lab_config.py` | Single source of truth for every tunable: hardware settings (robot IP, camera index), chessboard geometry, ArUco dictionary/markers, and **all quality thresholds** — camera-intrinsics RMS-reprojection bands (`CAM_CALIB_RMS_*`), the hand-eye `s_max` grading bands (`S_MAX_*`), the ArUco capture reject limit, and RANSAC iterations/inlier tightness. Override hardware values via env vars on the bench. | *(none — config module)* |
| `fk_lite6.py` | Forward kinematics skeleton (student fills in DH params + link transform). Returns `T_0_6` from joint angles. | *(none — library module)* |
| `robot_io.py` | Thin xArm SDK wrapper: connect, read joints, move, home. | *(none — library module)* |
| `transformation_calibration.py` | Live ArUco-artifact detection loop; records `(T_0_6, T_C_W)` pose pairs (SPACE to capture, q to save, ESC to abort). | `calibration_data.npy` |
| `get_transform.py` | Direct **eye-to-hand** solve via a self-contained `AX = X·BX⁻¹` solver (`solve_eye_to_hand`) on the full dataset, scored by artifact-position consistency (`s_max` = max per-axis std). No `cv2.calibrateHandEye` dependency. Also exposes `resolve_T_0_6()` shared by other scripts. | `T_6_C_normal.npy`, `T_6_C_normal.txt` |
| `ransac_calibration.py` | Robust **eye-to-hand** solve: RANSAC sampling (min subset 4), outlier rejection using the same shared solver, baseline comparison (all poses, no rejection), nonlinear least-squares refinement, plotting. | `T_6_C_ransac.npy`, `T_6_C_ransac.txt`, `ransac_calibration_results.txt` |
| `validate_calibration.py` | Live keygated validation at a relocated artifact; loads both `T_6_C` results, reconstructs artifact position with each, prints + writes side-by-side comparison. | `validation_data.npy`, `validation_results.txt` |

`ransac_calibration.py` and `get_transform.py` are two independent routes to the same target (`T_6_C`) and both consume `calibration_data.npy`. `validate_calibration.py` evaluates both on the same fresh set of poses so the comparison is apples-to-apples.

## Units

Internal maths and all stored transforms use **millimetres + degrees**, matching the xArm SDK (`get_position()` returns `[x, y, z, roll, pitch, yaw]` in mm/deg) and the ArUco PnP solve (artifact translation in mm). Joint angles are handled internally in radians by `fk_lite6` and `robot_io.read_joints_rad`; the SDK boundary reports degrees. No metre conversion layer is needed because mm is the natural unit throughout.

## How to run

From anywhere (scripts resolve `data/` themselves):

```bash
python3 src/image_capture.py              # 1. grab chessboard frames
python3 src/camera_calibration.py         # 2. estimate intrinsics
python3 src/transformation_calibration.py # 3. record pose pairs (robot connected)
python3 src/get_transform.py              # 4a. direct hand-eye solve
python3 src/ransac_calibration.py         # 4b. robust (RANSAC) hand-eye solve
python3 src/validate_calibration.py       # 5. validate at a relocated artifact
```

Part A (MATLAB): open `reference/MATLAB_FK/fk_lite6.m` in MATLAB and run it. The Python equivalent skeleton is `src/fk_lite6.py`.

Requires: Python 3.12+, OpenCV 4.x, NumPy, SciPy, Matplotlib, xArm Python SDK, and a reachable UFACTORY Lite 6 at `192.168.1.153`.
