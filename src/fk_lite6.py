"""Forward kinematics for the UFACTORY Lite 6 (modified / Craig D-H).

Python port of reference/MATLAB_FK/fk_lite6.m. This is the single source of
truth for turning joint angles into a base->end-effector transform in Lab 1, so
the hand-eye solver derives T_base_ee from the arm's reported joints rather than
trusting the controller's cartesian get_position() output.

Units:
    q      : joint angles [q1..q6] in RADIANS
    d, a   : link offset / length in MM
    alpha  : twist angle in DEGREES (converted internally)
    theta  : DH column offset in DEGREES (converted internally), added to q

Returns a 4x4 homogeneous transform whose translation column is in MM.
"""

import numpy as np

# Modified-DH columns: [theta_offset_deg, d_mm, alpha_deg, a_mm]
_DH = np.array([
    [0.0,   243.3,   0.0,   0.0],
    [-90.0,   0.0, -90.0,   0.0],
    [-90.0,   0.0, 180.0, 200.0],
    [0.0,   227.6,  90.0,  87.0],
    [0.0,     0.0,  90.0,   0.0],
    [0.0,    61.5, -90.0,   0.0],
])


def _link_transform(theta_rad, d_mm, alpha_rad, a_mm):
    """One modified (Craig) DH link transform."""
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [ct,       -st,      0.0,  a_mm],
        [st * ca,  ct * ca, -sa,  -sa * d_mm],
        [st * sa,  ct * sa,  ca,   ca * d_mm],
        [0.0,      0.0,      0.0,  1.0],
    ])


def fk_lite6(q):
    """Base -> end-effector transform.

    Args:
        q: iterable of 6 joint angles in RADIANS.

    Returns:
        T_base_ee: 4x4 float64 homogeneous matrix, translation in MM.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError(f"fk_lite6 expects 6 joint angles, got shape {q.shape}")

    d = _DH[:, 1]
    a = _DH[:, 3]
    alpha = np.deg2rad(_DH[:, 2])
    theta_offset = np.deg2rad(_DH[:, 0])

    T = np.eye(4)
    for i in range(6):
        theta = q[i] + theta_offset[i]
        T = T @ _link_transform(theta, d[i], alpha[i], a[i])
    return T


if __name__ == "__main__":
    # Sanity check: at zero pose the EE should sit ~+154 mm along Z.
    t0 = fk_lite6(np.zeros(6))[:3, 3]
    print("q=0 EE position (mm):", t0.round(3))
    assert abs(t0[2] - 154.0) < 5.0, f"unexpected q=0 Z: {t0[2]}"
    print("OK")
