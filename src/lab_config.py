r"""Shared configuration for Lab 1 scripts.

Every tunable knob lives here so there is one place to change them, instead of
digging through individual scripts. 
  * Algorithm / geometry constants (chessboard size, RANSAC thresholds, quality
    bands, ...) are plain values you edit directly below.
"""
import os

# ---------------------------------------------------------------------------
# Robot & camera (overridable via environment variables)
# ---------------------------------------------------------------------------

# UFACTORY Lite 6 controller IP (same subnet as the PC; verify with ping).
ROBOT_IP = os.environ.get('ROBOT_IP', '192.168.1.153')

# USB device index of the wrist-mounted camera (integer passed to
# cv2.VideoCapture). Which index your wrist cam gets varies by machine, so
# find it first:
#   Linux:       v4l2-ctl --list-devices          (or ls /dev/video*)
#   macOS:       system_profiler SPUSBDataType    (or just try indices 0..N)
#   Windows:     Device Manager > Imaging devices; usually index 0
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))

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
CALIB_IMAGE_PATTERN = "calib_*.jpg"

# Quality thresholds for the camera intrinsic calibration, expressed as the RMS
# reprojection error epsilon_rms in pixels (the value reported and entered in
# Table 2). Per the worksheet: it "should be below 0.5 px; above 1.0 px,
# recapture with more varied views." Values between the two are fine but not
# ideal.
CAM_CALIB_RMS_GOOD_TARGET_PX = 0.5   # at/below this -> EXCELLENT
CAM_CALIB_RMS_RECAPTURE_PX = 1.0     # above this -> POOR, recapture

# ---------------------------------------------------------------------------
# ArUco artifact detection (hand-eye capture)
# ---------------------------------------------------------------------------
# The ARUCO_ARTIFACT is the rigid plate carrying the ArUco markers used for
# hand-eye calibration. Its pose in the camera frame is T_C_W (camera C to
# artifact W). Distinct from the CHESSBOARD used for intrinsics above.

# ArUco dictionary and marker IDs for the artifact. The polyhedron carries five
# markers (IDs 0, 2, 3, 4, 5) printed from the DICT_6X6_250 set. ARUCO_DICT_NAME
# holds the attribute name looked up on cv2.aruco via getattr(), e.g.
# "DICT_6X6_250". The corner geometry lives in ../data/aruco_config.json
ARUCO_DICT_NAME = "DICT_6X6_250"
ARUCO_MARKER_IDS = [0, 2, 3, 4, 5]

# During hand-eye capture, discard any pose whose ArUco reprojection error
# exceeds this (pixels). Lower is stricter; 15 px is a lenient default that
# trims only clearly-bad frames while keeping plenty of usable poses.
ARUCO_REPROJ_REJECT_PX = 15.0

# ---------------------------------------------------------------------------
# Hand-eye calibration (RANSAC + refinement)
# ---------------------------------------------------------------------------

# Number of random 3-pose subsets RANSAC samples before declaring a winner.
# More iterations -> more thorough search, slower. 2000 is a good default.
RANSAC_ITERATIONS = 2000

# A pose is an "inlier" if its reconstructed artifact position lies within this
# many mm of the consensus. Laxer keeps more poses in the final fit; tighter
# rejects borderline frames. Applied after the initial nonlinear polish.
RANSAC_INLIER_THRESHOLD_MM = 15.0

# Random seed for reproducibility (None = unseeded).
RANSAC_SEED = 0

# ---------------------------------------------------------------------------
# Hand-eye calibration quality grading (s_max, in mm)
# The worksheet rates a hand-eye calibration on
#     s_max = max(s_x, s_y, s_z)
# where s_i is the sample standard deviation of the reconstructed artifact
# position along axis i across the captured poses. Bands per the worksheet:
# excellent if s_max < 5 mm, good from 5 to 10 mm, acceptable from 10 to 20 mm,
# poor above 20 mm. Used by both ransac_calibration.py and validate_calibration.py.
# ---------------------------------------------------------------------------

S_MAX_EXCELLENT_MM = 5.0      # < this  -> EXCELLENT
S_MAX_GOOD_MM = 10.0          # < this  -> GOOD
S_MAX_ACCEPTABLE_MM = 20.0    # < this  -> ACCEPTABLE, else POOR

# ---------------------------------------------------------------------------
# Post-calibration validation (relocated-artifact check)
# ---------------------------------------------------------------------------

# How many poses to collect when validating against a relocated artifact.
VALIDATION_MIN_POSES = 6
VALIDATION_MAX_POSES = 20
