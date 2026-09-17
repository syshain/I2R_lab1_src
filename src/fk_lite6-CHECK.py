"""Forward kinematics for the UFACTORY Lite 6 (modified / Craig D-H).

Skeleton: fill in _link_transform() so fk_lite6() chains the six links into a
base->end-effector transform. See reference/MATLAB_FK/fk_lite6.m for the worked
version you are porting.

Units:
    q      : joint angles [q1..q6] in RADIANS
    d, a   : link offset / length in MM
    alpha  : twist angle in DEGREES (convert internally)
    theta  : DH column offset in DEGREES (convert internally), added to q

Returns a 4x4 homogeneous transform whose translation column is in MM.
"""

import numpy as np

# Modified-DH columns: [theta_offset_deg, d_mm, alpha_deg, a_mm]
_DH = np.array([
"""
Refer UFACTORY Lite 6 documentation
Fill in the modified DH parameters as 2 dimensional array
"""
])


def _link_transform(theta_rad, d_mm, alpha_rad, a_mm):
    # <TO_FILL>
    """One modified (Craig) DH link transform.

    TODO: build the 4x4 matrix from theta_rad, d_mm, alpha_rad, a_mm using the
    Craig convention:
        [ ct,  -st,       0,  a ]
        [ st*ca, ct*ca, -sa, -sa*d ]
        [ st*sa, ct*sa,  ca,  ca*d ]
        [ 0,    0,        0,  1 ]
    where c*/s* are cos/sin of the respective angles.
    t and a are angle offset and twist
    """
    raise NotImplementedError('TODO: _link_transform')


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

    # TODO: chain the six _link_transform calls, adding theta_offset[i] to q[i].
    T = np.eye(4)
    for i in range(6):
        # theta = <TO_FILL> # How do theta and q relate to each other?
        # T =  <TO_FILL> # Use the _link_transform function filled in above to generate the link-wise transformation matrix. Multiply all to get the final transform matrix

    return T


if __name__ == "__main__":
    # Sanity check once implemented: at zero pose the EE should sit ~+154 mm along Z.
    t0 = fk_lite6(np.zeros(6))[:3, 3]
    print("q=0 EE position (mm):", t0.round(3))
