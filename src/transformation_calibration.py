"""Capture paired robot + ArUco-board poses for hand-eye calibration.

Runs a live detection loop over the wrist-mounted camera; when the user hits
'c' it records the current T_base_ee alongside the detected T_cam_board, and
's' saves the collected pairs into ../data/calibration_data.npy.

The board geometry is read from ../data/board_config.json (a 3-D polyhedron of
markers), matching Labs 2 and 3. The corner coordinates are defined at 30 mm
marker scale in the config and scaled up by 4/3 to the physical 40 mm markers.
"""

import json

import numpy as np
import cv2
import cv2.aruco as aruco
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from lab_config import (
    ROBOT_IP, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    ARUCO_BOARD_SCALE, ARUCO_REPROJ_REJECT_PX,
)
from fk_lite6 import fk_lite6
import robot_io

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

# Per-marker corner winding correction (matches Labs 2/3 detector).
_CORNER_ROLL = {0: 0, 2: 0, 3: 0, 4: 0, 5: 0}


def get_robot_end_effector_pose(arm):
    """Return (T_base_ee, joints_deg, raw_pose) from the arm's current state.

    T_base_ee is derived by forward kinematics from the reported joint angles
    (fk_lite6), NOT taken from the controller's cartesian get_position(). This
    keeps the hand-eye inputs consistent with the DH model used everywhere else
    in Lab 1.

    Returns:
        T_base_ee   : 4x4 base->EE transform (translation in mm).
        joints_deg  : list of 6 joint angles in DEGREES (as reported by SDK).
        raw_pose    : controller's [x,y,z,roll,pitch,yaw] (mm,deg), kept only as
                      a cross-check against the FK result.
    """
    try:
        q_rad = robot_io.read_joints_rad(arm)   # (6,) radians
    except RuntimeError as e:
        print(f"Error reading joints: {e}")
        return None, None, None

    joints_deg = [float(v) for v in np.rad2deg(q_rad)]
    T_base_ee = fk_lite6(np.asarray(q_rad, dtype=np.float64))

    code_p, pose_data = arm.get_position()
    if code_p != 0:
        print(f"Warning: FK OK but get_position failed ({code_p}); skipping cross-check")
        raw_pose = None
    else:
        x, y, z, roll, pitch, yaw = pose_data
        raw_pose = [float(v) for v in (x, y, z, roll, pitch, yaw)]

    return T_base_ee, joints_deg, raw_pose


class ArucoBoardDetector:
    """3-D ArUco polyhedron detector driven by board_config.json."""

    def __init__(self, camera_index=None):
        if camera_index is None:
            camera_index = CAMERA_INDEX
        self.camera_matrix = np.load(str(_DATA_DIR / 'camera_matrix.npy'))
        self.dist_coeffs = np.load(str(_DATA_DIR / 'dist_coeffs.npy'))

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        self._build_board_model()

        self.arm = robot_io.connect_arm(ROBOT_IP)
        self.cap = None
        self.camera_index = camera_index
        self.calibration_data = []

    def _build_board_model(self):
        """Load marker corners from board_config.json, scale, and re-center."""
        config_path = str(_DATA_DIR / 'board_config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        board = cfg['toolList'][0]
        ids = board['marker_ids']
        corners = board['marker_corners_mm']

        scaled = {}
        for mid, raw in zip(ids, corners):
            pts = np.array(raw, dtype=np.float64) * ARUCO_BOARD_SCALE
            roll = _CORNER_ROLL.get(mid, 0)
            scaled[mid] = np.roll(pts, -roll, axis=0)

        all_pts = np.vstack([scaled[mid] for mid in ids])
        centroid = all_pts.mean(axis=0)

        self.marker_world = {mid: scaled[mid] - centroid for mid in ids}
        self.marker_ids_list = ids
        self.centroid_offset = centroid

        print(f"[Model] Loaded {len(ids)} markers  "
              f"scale={ARUCO_BOARD_SCALE:.4f}  "
              f"centroid=[{centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}]")

    def start_camera(self):
        """Open the camera; returns True on success."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print(f"✓ Camera {self.camera_index} opened")
        print(f"✓ OpenCV version: {cv2.__version__}")
        print(f"✓ Board: {len(self.marker_ids_list)} markers "
              f"(IDs {self.marker_ids_list}) from board_config.json")
        return True

    def detect_board(self, frame):
        """Detect the ArUco board and estimate its pose via solvePnP.

        Returns (success, T_cam_board, rvec, tvec, reproj_error); T_cam_board
        is 4x4. High reprojection error (>15 px) is treated as a failed frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return False, None, None, None, None

        ids_flat = ids.flatten()
        object_points = []
        image_points = []

        for i, mid in enumerate(ids_flat):
            if mid in self.marker_world:
                world_pts = self.marker_world[mid]
                for j in range(4):
                    object_points.append(world_pts[j])
                    image_points.append(corners[i][0][j])

        if len(object_points) < 4:
            return False, None, None, None, None

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
        if not success:
            return False, None, None, None, None

        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs, rvec, tvec
            )
        except Exception:
            pass

        proj, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        reproj_error = float(np.mean(
            np.linalg.norm(proj.reshape(-1, 2) - image_points, axis=1)
        ))
        if reproj_error > ARUCO_REPROJ_REJECT_PX:
            print(f"[REJECT] reproj_err={reproj_error:.2f}px "
                  f"> {ARUCO_REPROJ_REJECT_PX:g}px")
            return False, None, None, None, reproj_error

        R_mat, _ = cv2.Rodrigues(rvec)
        T_cam_board = np.eye(4, dtype=np.float64)
        T_cam_board[:3, :3] = R_mat
        T_cam_board[:3, 3] = tvec.flatten()

        return True, T_cam_board, rvec, tvec, reproj_error

    def draw_detection(self, frame, rvec, tvec, ids, corners, success):
        """Overlay detected markers and the board axes onto the frame."""
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        if success and rvec is not None:
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, 40.0
            )
            cv2.putText(frame, "Board Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            pos = tvec.flatten()
            cv2.putText(frame, f"X: {pos[0]:6.1f} mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Y: {pos[1]:6.1f} mm", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Z: {pos[2]:6.1f} mm", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "✗ Board not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def get_robot_end_effector_pose(self):
        """Fetch the current (T_base_ee, raw_pose) from the connected arm."""
        return get_robot_end_effector_pose(self.arm)

    def capture_calibration_pose(self):
        """Record one (T_base_ee, T_cam_board) pair for hand-eye calibration."""
        if self.cap is None:
            self.start_camera()

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return False

        success, T_cam_board, rvec, tvec, reproj = self.detect_board(frame)
        if not success:
            print("Failed to detect the ArUco board")
            return False

        T_base_ee, joints_deg, raw_pose = self.get_robot_end_effector_pose()
        if T_base_ee is None:
            print("Failed to read robot pose")
            return False

        self.calibration_data.append({
            'T_base_ee': T_base_ee,          # FK-derived (authoritative)
            'robot_joints': joints_deg,      # [q1..q6] degrees, from get_angle()
            'robot_pose_raw': raw_pose,      # cartesian cross-check (may be None)
            'T_cam_board': T_cam_board,
            'reproj_error_px': reproj,
            'timestamp': cv2.getTickCount()
        })

        ee_pos = T_base_ee[:3, 3]
        fk_rpy = R.from_matrix(T_base_ee[:3, :3]).as_euler('xyz', degrees=True)

        print(f"✓ Captured pose pair #{len(self.calibration_data)} "
              f"(reproj {reproj:.2f} px)")
        print(f"  Joints (deg): {[round(j, 2) for j in joints_deg]}")
        print(f"  EE via FK   : pos=({ee_pos[0]:.1f},{ee_pos[1]:.1f},{ee_pos[2]:.1f}) mm  "
              f"RPY=({fk_rpy[0]:.1f},{fk_rpy[1]:.1f},{fk_rpy[2]:.1f})°")
        if raw_pose is not None:
            x, y, z, *_ = raw_pose
            dx = ee_pos - np.array([x, y, z])
            print(f"  Cross-check   : ctrl pos=({x:.1f},{y:.1f},{z:.1f}) mm  "
                  f"|FK−ctrl|={np.linalg.norm(dx):.2f} mm")
        print("Cam to board transformation")
        for row in T_cam_board:
            print(f"{row[0]} {row[1]} {row[2]} {row[3]}")

        euler_cam_board = R.from_matrix(T_cam_board[:3, :3]).as_euler('zxy', degrees=True)
        print(f"  Board euler: ({euler_cam_board[0]:.1f}, {euler_cam_board[1]:.1f}, {euler_cam_board[2]:.1f}) deg")

        return True
    
    def run(self, mode='detect'):
        """Main loop."""
        if not self.start_camera():
            return
        
        self.calibration_data = []
        
        print("\n" + "="*60)
        print(f"ARUCO BOARD DETECTION - Mode: {mode.upper()}")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        
        if mode == 'calibrate':
            print("  'c' - Capture current pose (robot + marker)")
            print("  's' - Save captured data")
            print(f"  Currently captured: {len(self.calibration_data)} poses")
        else:
            print("  'c' - Print current transformation")
        
        print("="*60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            success, T_cam_board, rvec, tvec, _reproj = self.detect_board(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            frame = self.draw_detection(frame, rvec, tvec, ids, corners, success)
            frame = cv2.resize(frame, None, fx=0.5,fy=0.5)
            cv2.imshow('Aruco Board Detection', frame)
            
            if mode == 'calibrate':
                cv2.putText(frame, f"Calibration poses: {len(self.calibration_data)}", 
                           (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
            
            cv2.imshow('ArUco Board Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                if mode == 'calibrate':
                    self.capture_calibration_pose()
                elif success:
                    print("\n" + "="*40)
                    print("T_cam_board (Camera to Board):")
                    print("="*40)
                    print(T_cam_board)
                    print(f"\nPosition: ({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f}) mm")
            elif key == ord('s') and mode == 'calibrate':
                if len(self.calibration_data) > 0:
                    np.save(str(_DATA_DIR / 'calibration_data.npy'), self.calibration_data)
                    print(f"✓ Saved {len(self.calibration_data)} calibration poses")
                else:
                    print("No calibration data to save. Press 'c' to capture poses first.")
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(f"Connecting to UFACTORY Lite 6 at {ROBOT_IP}...")
    detector = ArucoBoardDetector()
    detector.run(mode='calibrate')
