# Lab 1 — Repository Structure

This file documents the canonical folder/module template used by this lab. Labs 2 and 3 follow the same convention, so a student who learns the layout here can navigate any of the three labs identically.

Lab 1 is **camera + hand-eye calibration** in Python: calibrate the wrist camera intrinsics from a chessboard, capture paired robot/board poses, then solve for the end-effector-to-camera transform `T_ee_cam`. The active pipeline follows the shared `src/` template below; earlier exploratory scripts and the MATLAB forward-kinematics part are frozen provenance under `reference/`.

## Top-level layout

```
Lab_1/
├── src/          # Active Python code. One flat directory of single-responsibility scripts.
├── data/         # Calibration artefacts (.npy/.npz/.txt) + captured images. Produced/consumed at runtime.
├── skeletons/    # Student starting point: identical module set to src/, every function body stubbed.
├── reference/    # Frozen provenance, never imported by src/:
│   ├── calib_3D.py                 # Superseded 3-D calibration attempt.
│   ├── hand_eye_solver.py          # Earlier hand-eye solver, replaced by get_tranform_3D.py.
│   ├── camera_calibration.pdf      # Original camera-calibration notes.
│   └── MATLAB_FK/                  # Part A forward kinematics (fk_lite6.m, position_ufactory_modifiedDH.mlx).
├── docs/         # Pipeline usage guide (README.md) and requirements.txt.
├── lab1_instructions.md            # Main instruction sheet.
└── commit-log.md                   # Human-readable version history.
```

Rules that make the template portable across labs:

1. **`src/` is the only place active Python code lives.** It is a *flat* directory — no nested packages, no `__init__.py`. Each script is standalone (no cross-imports between them); they exchange results through files in `data/`. This keeps each stage independently runnable and easy to open in isolation.
2. **`data/` holds binary calibration output and captured images, not source.** Files here are produced by one stage and consumed read-only by the next. Paths into `data/` are resolved relative to the repo via `_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'` in each script, never hardcoded absolute paths — so a script runs correctly no matter what the current working directory is.
3. **`skeletons/` mirrors `src/` one-to-one** so a student's submission compiles against the same import graph before they fill in a single body. Each stubbed function raises `NotImplementedError("TODO: <name>")`; signatures, docstrings, imports, and module constants are preserved verbatim from `src/`.
4. **`reference/` is for provenance, not execution.** Code archived here (the superseded 3-D calibration, the earlier hand-eye solver, the MATLAB FK part) is where a current design decision came from. It is intentionally *not* on the Python import path so students cannot accidentally depend on superseded logic.

## Module roles (Python pipeline)

The stages run in order; each reads the previous stage's output from `data/` and writes its own.

| Script | Owns | Produces in `data/` |
|--------|------|---------------------|
| `image_capture.py` | Grabs chessboard frames from the USB camera on demand. | `calib_NNN.jpg` |
| `camera_calibration.py` | Intrinsics + distortion from the chessboard images (`findChessboardCorners` + `calibrateCamera`). | `camera_matrix.npy`, `dist_coeffs.npy`, `calibration_results.npz`, `calibration_parameters.txt` |
| `transformation_calibration.py` | Live ArUco-board detection loop; records `(T_base_ee, T_cam_board)` pose pairs from the arm + wrist camera. | `calibration_data.npy` |
| `ransac_calibration.py` | Robust hand-eye solve: RANSAC sampling, outlier rejection, multi-method `calibrateHandEye`, baseline comparison (all poses, no rejection), nonlinear least-squares refinement, plotting. | `T_ee_cam_ransac.npy`, `T_ee_cam_ransac.txt` |
| `get_tranform_3D.py` | Direct hand-eye solve: tests all 5 OpenCV methods on the full dataset, scores by board-position consistency, keeps the best. | `T_ee_cam_normal.npy`, `T_ee_cam_normal.txt` |
| `validate_calibration.py` | Live keygated validation at a relocated board; loads both T_ee_cam results, reconstructs board position with each, prints side-by-side comparison. | `validation_data.npy` |

`ransac_calibration.py` and `get_tranform_3D.py` are two independent routes to the same target (`T_ee_cam`) and both consume `calibration_data.npy`. `validate_calibration.py` evaluates both on the same fresh set of poses so the comparison is apples-to-apples.

## Units

Internal maths and all stored transforms use **millimetres + degrees**, matching the xArm SDK (`get_position()` returns `[x, y, z, roll, pitch, yaw]` in mm/deg) and the ArUco PnP solve (board translation in mm). No metre/radian conversion layer is needed in this lab because the SDK boundary is the natural unit system throughout.

## How to run

From anywhere (scripts resolve `data/` themselves):

```bash
python3 Lab_1/src/image_capture.py             # 1. grab chessboard frames
python3 Lab_1/src/camera_calibration.py        # 2. estimate intrinsics
python3 Lab_1/src/transformation_calibration.py  # 3. record pose pairs (robot connected)
python3 Lab_1/src/ransac_calibration.py        # 4a. robust hand-eye solve
python3 Lab_1/src/get_tranform_3D.py           # 4b. brute-force hand-eye solve
```

Part A (MATLAB): open `reference/MATLAB_FK/fk_lite6.m` in MATLAB and run it.

Requires: Python 3.12+, OpenCV 4.x, NumPy, SciPy, Matplotlib, xArm Python SDK, and a reachable UFACTORY Lite 6 at `192.168.1.153`.
