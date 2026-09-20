# Lab 1 — Repository Structure

This file documents the canonical folder/module template used by this lab. 

Lab 1 is **camera + hand-eye calibration** in Python: calibrate the wrist camera intrinsics from a chessboard, capture paired robot/artifact poses, then solve for the end-effector-to-camera transform `T_6_C`. The active pipeline follows the shared `src/` template below; 

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

**Formulation: eye-in-hand.** The wrist-mounted camera moves with the end-effector while observing a *static* bench artifact, so the loop closes as `T_0_W = T_0_6(i) @ X @ T_C_W(i)` with `X = T_6_C` constant. This is the classic **eye-in-hand** AX = XB problem. It is solved with OpenCV's `cv2.calibrateHandEye(..., method=cv2.CALIB_HAND_EYE_PARK)`: our pairs map directly onto its convention (`T_0_6` → gripper→base, `T_C_W` → target→camera), and the returned cam→gripper transform is exactly `T_6_C`. Wrapped in `solve_hand_eye_park()` in `get_transform.py` (with `solve_eye_to_hand` kept as a backwards-compatible alias).

Two physical artifacts are used:
- **Chessboard** — printed checker pattern for camera intrinsics (`camera_calibration.py`).
- **ArUco artifact** — rigid plate carrying ArUco markers for hand-eye calibration (`transformation_calibration.py`).

## Top-level layout

```
Lab_1/
├── src/          # Active Python code. One flat directory of single-responsibility scripts.
├── data/         # Calibration artifacts (.npy/.npz/.txt) + captured images. Produced/consumed at runtime.
├── docs/         # This structure guide.
├── requirements.txt
└── commit-log.md                   # Human-readable version history.
```

Rules that make the template portable across labs:

1. **`src/` is the only place active Python code lives.** It is a *flat* directory — no nested packages, no `__init__.py`. Scripts may import shared helpers from sibling modules (`fk_lite6`, `lab_config`, `robot_io`, `get_transform.resolve_T_0_6`) but exchange calibration results through files in `data/`. This keeps each stage independently runnable.
2. **`data/` holds binary calibration output, captured images and run results in human-readable format, not source.** Files here are produced by one stage and consumed by the next. Paths into `data/` are resolved relative to the repo via `_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'` in each script, never hardcoded absolute paths — so a script runs correctly no matter what the current working directory is.

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
| `get_transform.py` | Direct **hand-eye** solve via `cv2.calibrateHandEye` (Park method, `solve_hand_eye_park`) on the full dataset, scored by artifact-position consistency (`s_max` = max per-axis std). Also exposes `resolve_T_0_6()` shared by other scripts. | `T_6_C_normal.npy`, `T_6_C_normal.txt` |
| `ransac_calibration.py` | Robust **hand-eye** solve: RANSAC sampling (min subset 4) with the shared cv2 Park-method solver, inlier assignment on the raw winner, then nonlinear least-squares refinement on the inlier subset; compared against a plain baseline (Park solve on all poses, no rejection, no refinement). Plots and reports both results. | `T_6_C_ransac.npy`, `T_6_C_ransac.txt`, `ransac_calibration_results.txt` |
| `validate_calibration.py` | Live keygated validation at a relocated artifact; loads both `T_6_C` results, reconstructs artifact position with each, prints + writes side-by-side comparison. | `validation_data.npy`, `validation_results.txt` |

`ransac_calibration.py` and `get_transform.py` are two independent routes to the same target (`T_6_C`) and both consume `calibration_data.npy`: the former adds RANSAC outlier rejection plus a nonlinear polish on top of the shared cv2 Park-method solver, the latter runs that solver alone. `validate_calibration.py` evaluates both on the same fresh set of poses so the comparison is apples-to-apples.

## Units

Internal maths and all stored transforms use **millimetres + degrees**, matching the xArm SDK (`get_position()` returns `[x, y, z, roll, pitch, yaw]` in mm/deg) and the ArUco PnP solve (artifact translation in mm). Joint angles are handled internally in radians by `fk_lite6` and `robot_io.read_joints_rad`; the SDK boundary reports degrees. 

## How to run
Create a virtual environment in the main directory as follows
```bat
python.exe -m venv .venv
```
Activate the virtual environment
```bat
.\.venv\Scripts\activate
```
The command line should now have `(.venv)` preceding the prompt. If running the code for the first time, install all required Python libraries with the following command
```bat
pip install -r .\requirements.txt
```
Once the pip installer finishes, the code is ready to run.
Open `lab_config.py` and ensure the config parameters are correct. `ROBOT_IP` is found on the control box of the robot.
Run the stages in order from the repository root (grab frames, estimate intrinsics, record pose pairs, direct solve, robust solve, then validate — the scripts resolve `data/` themselves, so no need to change directory):

```bat
python.exe src\image_capture.py
python.exe src\camera_calibration.py
python.exe src\transformation_calibration.py
python.exe src\get_transform.py
python.exe src\ransac_calibration.py
python.exe src\validate_calibration.py
```

Requires: Python 3.12+, OpenCV 4.13.0.92, NumPy, SciPy, Matplotlib, xArm Python SDK, and a reachable UFACTORY Lite 6.
