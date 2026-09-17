"""Shared configuration for Lab 1 scripts.

Every tunable knob lives here so there is one place to change them, instead of
digging through individual scripts. Two kinds of value:

  * Hardware / connection settings (ROBOT_IP, CAMERA_INDEX, ...) can be
    overridden without editing code via environment variables — handy on the
    bench:

        export ROBOT_IP=192.168.1.153
        export CAMERA_INDEX=4
        python src/image_capture.py

  * Algorithm / geometry constants (chessboard size, RANSAC thresholds, quality
    bands, ...) are plain values you edit directly below.
"""
import os

# ---------------------------------------------------------------------------
# Robot & camera (overridable via environment variables)
# ---------------------------------------------------------------------------

# UFACTORY Lite 6 controller IP (same subnet as the PC; verify with ping).
ROBOT_IP = os.environ.get('ROBOT_IP', '192.168.1.153')

# USB device index of the wrist-mounted camera. Find it with:
#   Linux:  v4l2-ctl --list-devices
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '1'))

# Capture resolution used by the calibration pipelines.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# ---------------------------------------------------------------------------
# Camera (chessboard) intrinsic calibration
# ---------------------------------------------------------------------------

# Number of INNER chessboard corners per (row, column). For a board that shows
# 11x8 outer squares this is (10, 7). Count the dark-light intersections.
CHESSBOARD_SIZE = (10, 7)

# Physical side length of one chessboard square, in mm. Measure your printed
# board with a ruler — this sets the real-world scale of the calibration.
SQUARE_SIZE_MM = 15.0

# Glob pattern (relative to the data dir) matching the captured chessboard images.
CALIB_IMAGE_PATTERN = os.path.join("camera_calibration_images","calib_*.jpg")

# ---------------------------------------------------------------------------
# ArUco board detection (hand-eye capture)
# ---------------------------------------------------------------------------

# The board_config.json stores marker corners at a 30 mm reference scale; the
# physical markers are 40 mm, so detected corners are scaled up by this factor.
ARUCO_BOARD_SCALE = 4.0 / 3.0

# Frames whose ArUco reprojection error exceeds this (pixels) are rejected as
# unreliable during capture. Lower = stricter, fewer accepted poses.
ARUCO_REPROJ_REJECT_PX = 15.0

# ---------------------------------------------------------------------------
# Hand-eye calibration (RANSAC + refinement)
# ---------------------------------------------------------------------------

# Number of random 3-pose subsets RANSAC samples before declaring a winner.
# More iterations -> more thorough search, slower. 2000 is a good default.
RANSAC_ITERATIONS = 2000

# A pose is an "inlier" if its reconstructed board position lies within this
# many mm of the consensus. Laxer keeps more poses in the final fit; tighter
# rejects borderline frames. Applied after the initial nonlinear polish.
RANSAC_INLIER_THRESHOLD_MM = 25.0

# Random seed for reproducibility (None = unseeded).
RANSAC_SEED = 0

# ---------------------------------------------------------------------------
# Calibration quality grading (max std-dev of board position, in mm)
# Used by both ransac_calibration.py and validate_calibration.py.
# ---------------------------------------------------------------------------

QUALITY_EXCELLENT_MM = 5.0    # < this  -> EXCELLENT
QUALITY_GOOD_MM = 10.0        # < this  -> GOOD
QUALITY_ACCEPTABLE_MM = 20.0  # < this  -> ACCEPTABLE, else POOR

# ---------------------------------------------------------------------------
# Post-calibration validation (relocated-board check)
# ---------------------------------------------------------------------------

# How many poses to collect when validating against a moved board.
VALIDATION_MIN_POSES = 8
VALIDATION_MAX_POSES = 10
