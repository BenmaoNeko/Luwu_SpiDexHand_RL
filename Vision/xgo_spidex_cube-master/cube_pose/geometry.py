"""All lengths are millimetres; T_camera_cube maps cube coordinates to camera."""

from itertools import permutations, product

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def basis(normal):
    n = np.asarray(normal, float)
    u = np.eye(3)[(int(np.argmax(abs(n))) + 1) % 3]
    return u, np.cross(n, u)


def face_quad(normal, size):
    n = np.asarray(normal, float)
    u, v = basis(n)
    return np.array([n + u + v, n - u + v, n - u - v, n + u - v]) * size / 2


def axis_rotation(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, float) * angle).as_matrix()


def project(points, rvec, tvec, K, dist=None):
    return cv2.projectPoints(
        np.asarray(points, float).reshape(-1, 3),
        np.asarray(rvec, float),
        np.asarray(tvec, float),
        K,
        dist,
    )[0].reshape(-1, 2)


def wrap_angle_deg(angle):
    return (float(angle) + 45.0) % 90.0 - 45.0


def unwrap_angle_deg(angle, reference):
    return float(angle + 90 * np.round((reference - angle) / 90))


def cube_symmetries():
    out = []
    for perm in permutations(range(3)):
        for signs in product([-1, 1], repeat=3):
            R = np.eye(3)[:, perm] * signs
            if np.linalg.det(R) > 0.5:
                out.append(R)
    return out


def geometry(size, layer_axis=None, angle=0.0, samples=9):
    """Grid segments of a cube or two independently rigid slabs.

    Returns sampled points, tangent directions, face normals, component labels,
    and line segments. Interface/internal faces are deliberately omitted.
    """
    half = size / 2
    intervals = np.linspace(-half, half, 4)
    components = [(np.full(3, -half), np.full(3, half), np.eye(3), 0)]
    if layer_axis is not None:
        axis = np.asarray(layer_axis, float)
        k = int(np.argmax(abs(axis)))
        low_b, high_b = np.full(3, -half), np.full(3, half)
        low_l, high_l = low_b.copy(), high_b.copy()
        if axis[k] > 0:
            high_b[k] = low_l[k] = half / 3
        else:
            low_b[k] = high_l[k] = -half / 3
        components = [(low_b, high_b, np.eye(3), 0), (low_l, high_l, axis_rotation(axis, angle), 1)]
    pts, tangents, normals, labels, lines = [], [], [], [], []
    for lo, hi, Q, label in components:
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            for sign in [-1, 1]:
                plane = lo[a] if sign < 0 else hi[a]
                if abs(abs(plane) - half) > 1e-5:
                    continue
                normal = np.eye(3)[a] * sign
                for fixed, varying in [(b, c), (c, b)]:
                    for level in intervals:
                        if not lo[fixed] - 1e-5 <= level <= hi[fixed] + 1e-5:
                            continue
                        p = np.zeros(3)
                        p[a], p[fixed], p[varying] = plane, level, lo[varying]
                        q = p.copy()
                        q[varying] = hi[varying]
                        p, q = Q @ p, Q @ q
                        tangent = q - p
                        ts = np.linspace(0.07, 0.93, samples)
                        pts.extend(p[None, :] + ts[:, None] * tangent)
                        tangents.extend([tangent / np.linalg.norm(tangent)] * samples)
                        normals.extend([Q @ normal] * samples)
                        labels.extend([label] * samples)
                        lines.append((p, q, Q @ normal, label))
    return tuple(np.asarray(x) for x in (pts, tangents, normals, labels)), lines, components


def visible_samples(points, normals, labels, R, t, components):
    camera = -R.T @ t
    visible = np.sum(normals * (camera - points), axis=1) > 1e-5
    if len(components) > 1:
        for lo, hi, Q, label in components:
            origin = Q.T @ camera
            dirs = (points - camera) @ Q
            safe = np.where(abs(dirs) < 1e-10, 1e-10, dirs)
            near = np.minimum((lo - origin) / safe, (hi - origin) / safe).max(axis=1)
            far = np.maximum((lo - origin) / safe, (hi - origin) / safe).min(axis=1)
            hidden = (far >= np.maximum(near, 0)) & (near < 0.995) & (far > 1e-5)
            visible &= ~((labels != label) & hidden)
    return visible


def pose_dict(rvec, tvec):
    rotation = Rotation.from_rotvec(rvec)
    R = rotation.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return {
        "translation_mm": np.asarray(tvec).tolist(),
        "rpy_xyz_deg": rotation.as_euler("xyz", degrees=True).tolist(),
        "rotation_vector_rad": np.asarray(rvec).tolist(),
        "quaternion_xyzw": rotation.as_quat().tolist(),
        "rotation_matrix": R.tolist(),
        "T_camera_cube_mm": T.tolist(),
    }
