"""Thin wrapper around the xArm SDK for the Lite 6.

Keeps every hardware call in one place so the calibration scripts never touch
the SDK directly. Ported from Lab 3's robot_io.py, made self-contained: the
joint-limit table is defined here (rather than imported from ik_solver) and the
controller IP comes from lab_config so it stays overridable via ROBOT_IP.

Units: joint angles are returned as (6,) numpy arrays in RADIANS; cartesian
positions as (6,) vectors [x, y, z, roll, pitch, yaw] with position in MM and
rpy in DEGREES (the SDK's native units).
"""

import time

import numpy as np

from lab_config import ROBOT_IP as _DEFAULT_ROBOT_IP


SERVO_SPEED = 30          # % speed for each servo move
POSITION_MODE = 0         # position control mode (programmed motion)

# Factory Lite 6 joint limits (degrees), per axis [min, max].
LITE6_JOINT_LIMITS_DEG = np.array([
    [-360.0,  360.0],   # J1
    [-150.0,  150.0],   # J2
    [  -3.5,  300.0],   # J3 (asymmetric minimum)
    [-360.0,  360.0],   # J4
    [-124.0,  124.0],   # J5
    [-360.0,  360.0],   # J6
])


def connect_arm(ip=_DEFAULT_ROBOT_IP):
    """Connect + enable motion; returns an XArmAPI. No reset() so we don't wipe
    the pose left from a previous run. Raises RuntimeError on failure."""
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip)
    try:
        # NOTE: no arm.reset() here — it drives the arm home on every connect,
        # which wipes out whatever pose you left it in between runs. We only
        # clear faults and enable motion so the arm stays exactly where it is.
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(POSITION_MODE)
        arm.set_state(0)
        time.sleep(2)
    except Exception as e:
        arm.disconnect()
        raise RuntimeError("Cannot connect to robot. Check IP and/or enable "
                           "robot in web app: %s" % e)
    return arm


def read_joints_rad(arm):
    """Current joint angles as a (6,) array in radians."""
    code, res = arm.get_servo_angle(is_radian=True)
    if code != 0:
        raise RuntimeError("Failed to read joint angles (code %d)" % code)
    return np.asarray(res[:6], dtype=float)


def within_joint_limits(q_deg):
    """True if a (6,) degree vector is inside the Lite 6 joint limits."""
    q_deg = np.asarray(q_deg, dtype=float)
    return bool(np.all((q_deg >= LITE6_JOINT_LIMITS_DEG[:, 0]) &
                       (q_deg <= LITE6_JOINT_LIMITS_DEG[:, 1])))


def move_to_joints(arm, q_target_rad, speed=SERVO_SPEED, wait=True):
    """Drive servos to q_target_rad (radians). wait=True blocks on arrival;
    wait=False queues for smooth trajectory streaming. Refuses out-of-limit
    targets. Returns (ok, msg)."""
    q_target_rad = np.asarray(q_target_rad, dtype=float)
    if not within_joint_limits(np.rad2deg(q_target_rad)):
        return False, 'IK solution outside joint limits'
    try:
        code = arm.set_servo_angle(angle=list(np.rad2deg(q_target_rad)),
                                   speed=speed, is_radian=False, wait=wait)
        if code == 0:
            return True, 'moved'
        return False, 'servo command rejected (code %d)' % code
    except Exception as e:  # network/timeout errors surface here
        return False, 'servo command error: %s' % e


HOME_JOINTS_DEG = [0, 0, 20, 0, 0, 0]


def go_home(arm):
    """Move the arm back to its home position (blocking). Returns (ok, msg)."""
    return move_to_joints(arm, np.deg2rad(HOME_JOINTS_DEG))
