import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import camera_for_image, load_config
from .features import Features
from .geometry import (
    axis_rotation,
    cube_symmetries,
    face_quad,
    geometry,
    pose_dict,
    project,
    unwrap_angle_deg,
    visible_samples,
    wrap_angle_deg,
)

SYMMETRIES = cube_symmetries()
NORMALS = np.vstack([np.eye(3), -np.eye(3)])


class CubePoseEstimator:
    def __init__(self, config=None):
        self.cfg = load_config() if config is None else config

    def estimate(self, image, task=1, angle_reference_deg=None):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a nonempty BGR image")
        if task not in (1, 2):
            raise ValueError("task must be 1 or 2")
        oh, ow = image.shape[:2]
        K, dist = camera_for_image(self.cfg, ow, oh)
        scale = min(1.0, self.cfg["detection"]["max_image_side"] / max(oh, ow))
        work = cv2.resize(image, (round(ow * scale), round(oh * scale)))
        # Rectify before line fitting; retain original intrinsics in output.
        Ks = K.copy()
        Ks[0] *= scale
        Ks[1] *= scale
        Ks[0, 2] += 0.5 * scale - 0.5
        Ks[1, 2] += 0.5 * scale - 0.5
        work = cv2.undistort(work, Ks, dist)
        f = Features(work, self.cfg)
        result = {
            "schema_version": 1,
            "task": task,
            "status": "no_detection",
            "pose": None,
            "layer": None,
            "camera": {
                "matrix": K.tolist(),
                "distortion": dist.tolist(),
                "calibrated": bool(self.cfg["camera"].get("calibrated", False)),
            },
            "coordinate_convention": "p_camera = R_camera_cube @ p_cube + t_mm; camera x right, y down, z forward",
            "warnings": [],
        }
        if not result["camera"]["calibrated"]:
            result["warnings"].append(
                "Placeholder intrinsics: metric pose is provisional, not calibrated accuracy."
            )
        if not f.valid or not f.quads:
            result["warnings"].append("Insufficient geometric corner proposals.")
            return result, {"work_image": work, "scale": scale}
        size = self.cfg["cube_size_mm"]
        geom_cache = {}

        def model(angle=0):
            if task == 1:
                if 0 not in geom_cache:
                    geom_cache[0] = geometry(size, samples=8)
                return geom_cache[0]
            key = round(float(angle), 8)
            if key not in geom_cache:
                geom_cache[key] = geometry(size, [0, 0, 1], angle, samples=8)
            return geom_cache[key]

        def center_assignment(params):
            r, t = params[:3], params[3:6]
            R = Rotation.from_rotvec(r).as_matrix()
            pos = NORMALS * size / 2
            vis = np.sum((pos @ R.T + t) * (NORMALS @ R.T), axis=1) < -size * 0.06
            if task == 2:
                _, _, components = model(params[6])
                center_labels = np.array([0, 0, 1, 0, 0, 0])
                vis &= visible_samples(pos, NORMALS, center_labels, R, t, components)
            pixels = project(pos, r, t, Ks)
            observed = [f.center_color(p)[0] if v else None for p, v in zip(pixels, vis)]
            best = None
            for S in SYMMETRIES:
                hits = []
                mismatch = 0
                for n, p, name in zip(NORMALS, pixels, observed):
                    if name is None:
                        continue
                    expected = np.array(self.cfg["colors"][name]["axis"])
                    if np.linalg.norm(S @ expected - n) < 0.1:
                        hits.append(
                            {
                                "color": name,
                                "axis": expected.tolist(),
                                "pixel": ((p + 0.5) / scale - 0.5).tolist(),
                            }
                        )
                    else:
                        mismatch += 1
                score = len(hits) - 1.5 * mismatch
                if best is None or score > best[0]:
                    best = (score, S, hits, mismatch)
            return best

        def measure(params, details=False):
            r, t = params[:3], params[3:6]
            if t[2] < size * 0.8:
                return (1e4, 0.0, None) if details else 1e4
            (pts, tangents, normals, labels), _, components = model(params[6] if task == 2 else 0)
            R = Rotation.from_rotvec(r).as_matrix()
            vis = visible_samples(pts, normals, labels, R, t, components)
            if vis.sum() < 40:
                return (1e4, 0.0, None) if details else 1e4
            uv = project(pts[vis], r, t, Ks)
            dirs = project(pts[vis] + tangents[vis] * 2, r, t, Ks) - uv
            ds = f.sample_distance(uv, dirs)
            trim = self.cfg["detection"]["occlusion_trim_fraction"]
            core = np.sort(ds)[: max(1, int(len(ds) * (1 - trim)))]
            edge = float(np.mean(np.minimum(core, 16)))
            if task == 2:
                component_costs = []
                for lab in (0, 1):
                    dcomp = ds[labels[vis] == lab]
                    if len(dcomp) < 12:
                        component_costs.append(16.0)
                        continue
                    c = np.sort(dcomp)[: max(1, int(len(dcomp) * (1 - trim)))]
                    component_costs.append(float(np.mean(np.minimum(c, 16))))
                edge = 0.65 * component_costs[0] + 0.35 * component_costs[1]
            x, y, w, h = f.object_bounds
            # Chromatic evidence need not cover white faces, but cannot lie far outside the model.
            pmin, pmax = uv.min(axis=0), uv.max(axis=0)
            deficit = np.maximum(pmin - [x + 7, y + 7], 0) + np.maximum(
                [x + w - 7, y + h - 7] - pmax, 0
            )
            oversize = np.maximum((pmax - pmin) / [w, h] - 1.65, 0)
            hull = cv2.convexHull(uv.astype(np.float32))
            area_ratio = cv2.contourArea(hull) / max(1.0, f.foreground_area)
            fg_dist = map_coordinates(
                f.foreground_distance, [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=30.0
            )
            fg_core = np.sort(fg_dist)[: max(1, int(len(fg_dist) * 0.78))]
            score = (
                edge
                + 0.07 * deficit.sum()
                + 4 * oversize.sum()
                + 6 * max(0, area_ratio - 1.35)
                + 0.25 * np.minimum(fg_core, 30).mean()
            )
            if details:
                return (
                    score,
                    float(np.mean(ds < self.cfg["detection"]["edge_tolerance_px"])),
                    (pts, vis, uv, ds),
                )
            return score

        candidates = []
        x, y, w, h = f.object_bounds
        square = face_quad([0, 0, 1], size)
        for q in f.quads:
            # Seed either a full geometric face or an individual 1/3-width cell.
            for cells in (3, 1):
                obj = square.copy()
                obj[:, :2] *= cells / 3
                try:
                    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                        obj, q.astype(float), Ks, None, flags=cv2.SOLVEPNP_IPPE
                    )
                except cv2.error:
                    continue
                for rv, tv in zip(rvecs, tvecs):
                    rv, tv = rv.ravel(), tv.ravel()
                    if tv[2] < size or tv[2] > size * 70:
                        continue
                    R = Rotation.from_rotvec(rv).as_matrix()
                    if np.dot(R[:, 2], tv + R[:, 2] * size / 2) >= 0:
                        continue
                    offsets = (
                        [(0, 0)] if cells == 3 else [(a, b) for a in [-1, 0, 1] for b in [-1, 0, 1]]
                    )
                    for a, b in offsets:
                        t = tv - R @ np.array([a * size / 3, b * size / 3, 0.0])
                        # Reject scale/translation hypotheses unable to contain the colored cube.
                        if cells == 1:
                            vertices = (
                                np.array(
                                    [
                                        [i, j, k]
                                        for i in [-0.5, 0.5]
                                        for j in [-0.5, 0.5]
                                        for k in [-0.5, 0.5]
                                    ]
                                )
                                * size
                            )
                            pp = project(vertices, rv, t, Ks)
                            lo, hi = pp.min(axis=0), pp.max(axis=0)
                            if (
                                np.any((hi - lo) / [w, h] > 2.0)
                                or np.any(lo > [x + w * 0.3, y + h * 0.3])
                                or np.any(hi < [x + w * 0.7, y + h * 0.7])
                            ):
                                continue
                        if task == 1:
                            p = np.r_[rv, t]
                            score = measure(p)
                            candidates.append((score, p))
                        else:
                            # The cap defines the rotation axis; the two-layer body is free to rotate about it.
                            for angle in np.deg2rad(np.arange(-40, 41, 10)):
                                body_R = R @ axis_rotation([0, 0, 1], -angle)
                                p = np.r_[Rotation.from_matrix(body_R).as_rotvec(), t, angle]
                                candidates.append((measure(p), p))
        if not candidates:
            result["warnings"].append("No positive-depth pose hypothesis.")
            return result, {"work_image": work, "scale": scale}
        candidates.sort(key=lambda c: c[0])
        # Geometric duplicates are merged before nonlinear fitting.
        shortlist = []
        for score, p in candidates:
            if any(
                np.linalg.norm(p[3:6] - prev[1][3:6]) < size * 0.035
                and abs(score - prev[0]) < 0.12
                and (task == 1 or abs(p[6] - prev[1][6]) < 0.15)
                for prev in shortlist
            ):
                continue
            shortlist.append((score, p))
            if len(shortlist) >= self.cfg["detection"]["max_candidates"]:
                break

        optimized = []
        for _, initial in shortlist:
            p = initial.copy()
            for _ in range(2):
                (points, tangents, normals, labels), _, components = model(
                    p[6] if task == 2 else 0.0
                )
                R = Rotation.from_rotvec(p[:3]).as_matrix()
                vis = visible_samples(points, normals, labels, R, p[3:6], components)
                p0 = points.copy()
                tan0 = tangents.copy()

                # Freeze visibility per outer iteration, never manufacture residuals for hidden edges.
                def residuals(v, p0=p0, tan0=tan0, p=p, labels=labels, vis=vis):
                    pts = p0.copy()
                    tans = tan0.copy()
                    if task == 2:
                        Q = axis_rotation([0, 0, 1], v[6] - p[6])
                        pts[labels == 1] = pts[labels == 1] @ Q.T
                        tans[labels == 1] = tans[labels == 1] @ Q.T
                    uv = project(pts[vis], v[:3], v[3:6], Ks)
                    dirs = project(pts[vis] + tans[vis] * 2, v[:3], v[3:6], Ks) - uv
                    ds = f.sample_distance(uv, dirs)
                    return ds

                lower = (
                    p
                    - np.r_[
                        [0.20] * 3,
                        [size * 0.22, size * 0.22, size * 0.35],
                        ([0.28] if task == 2 else []),
                    ]
                )
                upper = (
                    p
                    + np.r_[
                        [0.20] * 3,
                        [size * 0.22, size * 0.22, size * 0.35],
                        ([0.28] if task == 2 else []),
                    ]
                )
                lower[5] = max(lower[5], size)
                fit = least_squares(
                    residuals,
                    p,
                    bounds=(lower, upper),
                    loss="soft_l1",
                    f_scale=1.7,
                    diff_step=1e-4,
                    x_scale="jac",
                    max_nfev=45,
                )
                p = fit.x
            if task == 2:
                # Hold the layer's camera-frame orientation fixed while refining the
                # six body coordinates on the two stationary layers alone. Then fit
                # just the articulation angle with the body pose frozen.
                for stage in ("body", "layer"):
                    (pts, tans, normals, labels), _, components = model(p[6])
                    R0 = Rotation.from_rotvec(p[:3]).as_matrix()
                    vis = visible_samples(pts, normals, labels, R0, p[3:6], components)
                    vis &= labels == (0 if stage == "body" else 1)
                    if vis.sum() < 24:
                        continue
                    fixed = p.copy()

                    def component_residual(
                        v, stage=stage, pts=pts, tans=tans, vis=vis, fixed=fixed
                    ):
                        if stage == "body":
                            rr, tt = v[:3], v[3:6]
                            pp = pts[vis]
                            tangent = tans[vis]
                        else:
                            rr, tt = fixed[:3], fixed[3:6]
                            Q = axis_rotation([0, 0, 1], float(v[0]) - fixed[6])
                            pp = pts[vis] @ Q.T
                            tangent = tans[vis] @ Q.T
                        uv = project(pp, rr, tt, Ks)
                        dirs = project(pp + tangent * 2, rr, tt, Ks) - uv
                        return f.sample_distance(uv, dirs)

                    start = p[:6] if stage == "body" else p[6:7]
                    delta = (
                        np.r_[[0.10] * 3, [size * 0.09, size * 0.09, size * 0.15]]
                        if stage == "body"
                        else np.array([0.22])
                    )
                    fit = least_squares(
                        component_residual,
                        start,
                        bounds=(start - delta, start + delta),
                        loss="soft_l1",
                        f_scale=1.5,
                        diff_step=1e-4,
                        x_scale="jac",
                        max_nfev=28,
                    )
                    if stage == "body":
                        p[:6] = fit.x
                        # Keep the cap approximately fixed while body orientation changes.
                        R1 = Rotation.from_rotvec(p[:3]).as_matrix()
                        rel = R1.T @ R0 @ axis_rotation([0, 0, 1], fixed[6])
                        p[6] = np.arctan2(rel[1, 0], rel[0, 0])
                    else:
                        p[6] = fit.x[0]
            geo_score, support, extra = measure(p, True)
            assignment = center_assignment(p)
            # Axis labels are checked only at the fitted geometric face centers.
            score = geo_score + 2.5 * max(0, 2 - len(assignment[2])) + 2.0 * assignment[3]
            if task == 2:
                # Task 2 asserts a displaced layer. An almost rigid explanation is
                # weak evidence for ANY particular moving axis; prefer a similarly
                # good displaced-layer explanation, while retaining near-zero ambiguity.
                angle_mod = wrap_angle_deg(np.rad2deg(p[6]))
                score += 0.8 * np.exp(-((angle_mod / 3.0) ** 2))
            optimized.append((score, p, geo_score, support, assignment, extra))
        optimized.sort(key=lambda x: x[0])
        _, p, geo_score, support, assignment, extra = optimized[0]
        _, S, center_hits, mismatches = assignment
        R = Rotation.from_rotvec(p[:3]).as_matrix() @ S
        rv = Rotation.from_matrix(R).as_rotvec()
        status = "ok"
        if len(center_hits) < self.cfg["detection"]["minimum_center_colors"] or mismatches:
            status = "ambiguous"
            result["warnings"].append(
                "Fewer than two consistent nonparallel center colors; absolute cube axes are unconfirmed."
            )
        if support < self.cfg["detection"]["minimum_edge_support"] or geo_score > 4.0:
            status = "low_confidence"
            result["warnings"].append(
                "Insufficient geometric support or significant occlusion; inspect overlay."
            )
        result["status"] = status
        # An unconfirmed coordinate frame is returned only as a candidate, not as a valid pose.
        pose = pose_dict(rv, p[3:6])
        result["pose"] = pose if status == "ok" else None
        if status != "ok":
            result["candidate_pose"] = pose
        result["quality"] = {
            "edge_support_fraction": support,
            "geometric_fit_cost": geo_score,
            "median_directional_edge_error_px_work": float(np.median(extra[3])),
            "center_matches": center_hits,
            "center_mismatches": mismatches,
            "working_image_size": [f.w, f.h],
            "geometric_hypotheses": len(candidates),
        }
        if task == 2:
            ax = np.rint(S.T @ np.array([0, 0, 1])).astype(int)
            angle = wrap_angle_deg(np.rad2deg(p[6]))
            if abs(angle) < 3.0:
                status = "ambiguous"
                result["status"] = status
                result["candidate_pose"] = pose
                result["pose"] = None
                result["warnings"].append(
                    "Layer is almost aligned: the moving axis/angle cannot be reliably identified from this frame."
                )
            layer = {
                "axis_cube": ax.tolist(),
                "angle_deg_mod90": angle,
                "period_deg": 90,
                "sign_convention": "right-hand rule about axis_cube",
                "absolute_angle_observable": False,
            }
            _, vis, _, ds = extra
            sample_labels = model(p[6])[0][3][vis]
            for lab, name in [(0, "body"), (1, "layer")]:
                errors = ds[sample_labels == lab]
                result["quality"][name + "_edge_support_fraction"] = (
                    float(np.mean(errors < self.cfg["detection"]["edge_tolerance_px"]))
                    if len(errors)
                    else 0.0
                )
            if (
                result["quality"]["body_edge_support_fraction"] < 0.45
                or result["quality"]["layer_edge_support_fraction"] < 0.55
            ):
                status = "low_confidence"
                result["status"] = status
                result["candidate_pose"] = pose
                result["pose"] = None
                result["warnings"].append(
                    "The body or the moving layer lacks independent geometric support."
                )
            if angle_reference_deg is not None:
                layer["angle_deg_unwrapped"] = unwrap_angle_deg(angle, angle_reference_deg)
                layer["unwrap_reference_deg"] = float(angle_reference_deg)
            result["layer"] = layer if status == "ok" else None
            if status != "ok":
                result["candidate_layer"] = layer
            result["warnings"].append(
                "Single-frame layer angle is observable only modulo 90 degrees without facelet identities/history."
            )
        debug = {
            "work_image": work,
            "scale": scale,
            "K": Ks,
            "parameters": p,
            "geometry": model(p[6] if task == 2 else 0.0),
            "symmetry": S,
        }
        return result, debug


def draw_overlay(result, debug):
    image = debug["work_image"].copy()
    if "parameters" not in debug:
        return image
    p = debug["parameters"]
    K = debug["K"]
    _, lines, components = debug["geometry"]
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    for a, b, n, label in lines:
        mid = (a + b) / 2
        if not visible_samples(mid[None, :], n[None, :], np.array([label]), R, p[3:6], components)[
            0
        ]:
            continue
        uv = np.rint(project([a, b], p[:3], p[3:6], K)).astype(int)
        color = (255, 180, 0) if label == 0 else (255, 0, 255)
        cv2.line(image, tuple(uv[0]), tuple(uv[1]), color, 1, cv2.LINE_AA)
    Rc = R @ debug["symmetry"]
    rv = Rotation.from_matrix(Rc).as_rotvec()
    axes = project(np.vstack([np.zeros(3), np.eye(3) * 38]), rv, p[3:6], K)
    origin = tuple(np.rint(axes[0]).astype(int))
    for i, (color, name) in enumerate(zip([(0, 0, 255), (0, 220, 0), (255, 0, 0)], "XYZ")):
        end = tuple(np.rint(axes[i + 1]).astype(int))
        cv2.arrowedLine(image, origin, end, color, 2, cv2.LINE_AA, tipLength=0.12)
        cv2.putText(image, name, end, 0, 0.5, color, 1, cv2.LINE_AA)
    scale = debug["scale"]
    for match in result["quality"]["center_matches"]:
        p = (np.asarray(match["pixel"]) + 0.5) * scale - 0.5
        cv2.circle(image, tuple(np.rint(p).astype(int)), 5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 52), (25, 25, 25), -1)
    cv2.putText(
        image,
        f"{result['status']}  support {result['quality']['edge_support_fraction']:.2f}",
        (8, 20),
        0,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    layer = result.get("layer") or result.get("candidate_layer")
    if layer:
        cv2.putText(
            image,
            f"layer {layer['axis_cube']} : {layer['angle_deg_mod90']:.1f} deg (mod 90)",
            (8, 42),
            0,
            0.45,
            (255, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return image
