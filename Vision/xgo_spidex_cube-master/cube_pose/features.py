from itertools import combinations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment


def color_labels(image, cfg):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
    h, s, v = cv2.split(hsv)
    names = list(cfg["colors"])
    costs = np.full((len(names), *h.shape), np.inf)
    for i, name in enumerate(names):
        c = cfg["colors"][name]
        if "max_saturation" in c:
            valid = (s <= c["max_saturation"]) & (v >= c["min_value"])
            costs[i][valid] = 0.6
        else:
            rgb = np.array([[c["rgb"]]], np.uint8)
            hue = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0, 0])
            dh = np.minimum(abs(h - hue), 180 - abs(h - hue))
            valid = (dh <= c["hue_tolerance"]) & (s >= c["min_saturation"]) & (v >= c["min_value"])
            costs[i][valid] = dh[valid] / c["hue_tolerance"]
    labels = np.argmin(costs, axis=0).astype(np.int16)
    labels[~np.isfinite(costs.min(axis=0))] = -1
    return labels


def ordered_quad(points):
    q = np.asarray(points, np.float32).reshape(4, 2)
    if cv2.contourArea(q, oriented=True) > 0:
        q = q[::-1]
    return q.copy()


def quad_proposals(contour):
    """Four long support lines recover ideal sharp corners behind rounded tips."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    ap = cv2.approxPolyDP(hull, 0.018 * perimeter, True)[:, 0].astype(float)
    proposals = []
    if len(ap) == 4:
        proposals.append(ap)
    # Remove a bevel or an occluded corner by intersecting its adjacent long edges.
    if 4 <= len(ap) <= 8:
        lengths = np.linalg.norm(np.roll(ap, -1, axis=0) - ap, axis=1)
        for inds in combinations(range(len(ap)), 4):
            if min(lengths[list(inds)]) < perimeter * 0.075:
                continue
            lines = []
            for idx in inds:
                p, q = ap[idx], ap[(idx + 1) % len(ap)]
                line = np.cross(np.r_[p, 1], np.r_[q, 1])
                line /= np.linalg.norm(line[:2])
                lines.append(line)
            corners = []
            for j in range(4):
                p = np.cross(lines[j - 1], lines[j])
                if abs(p[2]) < 1e-6:
                    break
                corners.append(p[:2] / p[2])
            if len(corners) != 4:
                continue
            q = np.array(corners, np.float32)
            if not cv2.isContourConvex(q):
                continue
            area = cv2.contourArea(q)
            if not 0.8 * cv2.contourArea(hull) < area < 1.7 * cv2.contourArea(hull):
                continue
            if np.max(np.linalg.norm(q - ap.mean(axis=0), axis=1)) > perimeter * 0.65:
                continue
            proposals.append(q)
    return [ordered_quad(q) for q in proposals]


class Features:
    def __init__(self, image, cfg):
        self.h, self.w = image.shape[:2]
        self.labels = color_labels(image, cfg)
        self.names = list(cfg["colors"])
        chromatic = np.zeros((self.h, self.w), np.uint8)
        for i, c in enumerate(cfg["colors"].values()):
            if "max_saturation" not in c:
                chromatic[self.labels == i] = 255
        count, cc, stats, _ = cv2.connectedComponentsWithStats(chromatic)
        good = np.zeros_like(chromatic)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] >= cfg["detection"]["minimum_region_area"] * 0.4:
                good[cc == i] = 255
        merged = cv2.dilate(good, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.valid = False
            return
        co = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(co)
        if w * h < cfg["detection"]["minimum_region_area"] * 3:
            self.valid = False
            return
        pad = int(max(w, h) * 0.42)
        component_mask = np.zeros_like(good)
        cv2.drawContours(component_mask, [co], -1, 255, -1)
        good &= component_mask
        self.roi = (
            max(0, x - pad),
            max(0, y - pad),
            min(self.w, x + w + pad),
            min(self.h, y + h + pad),
        )
        self.object_bounds = (x, y, w, h)
        self.valid = True
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        self.edges = np.zeros((self.h, self.w), np.uint8)
        for channel in cv2.split(lab):
            self.edges |= cv2.Canny(channel, 30, 75)
        x0, y0, x1, y1 = self.roi
        roi_mask = np.zeros_like(self.edges)
        roi_mask[y0:y1, x0:x1] = 255
        self.edges &= roi_mask
        self.distance = cv2.distanceTransform(255 - self.edges, cv2.DIST_L2, 5)
        # Directional chamfer rejects a projected grid that merely crosses many unrelated edges.
        self.orientation_bins = 12
        strength = np.zeros((self.h, self.w), np.float32)
        orientation = np.zeros_like(strength)
        for channel in cv2.split(lab):
            gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
            mag = gx * gx + gy * gy
            use = mag > strength
            orientation[use] = (np.arctan2(gy, gx)[use] + np.pi / 2) % np.pi
            strength = np.maximum(strength, mag)
        self.oriented_distance = []
        for k in range(self.orientation_bins):
            angle = k * np.pi / self.orientation_bins
            delta = abs(orientation - angle)
            delta = np.minimum(delta, np.pi - delta)
            edge = ((self.edges > 0) & (delta < np.pi / self.orientation_bins * 1.4)).astype(
                np.uint8
            ) * 255
            self.oriented_distance.append(cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5))
        self.oriented_distance = np.stack(self.oriented_distance)
        self.quads = []
        foreground = good.copy()
        # These are only geometric proposals. Region color never assigns an axis.
        for idx in range(len(self.names)):
            mask = (self.labels == idx).astype(np.uint8) * 255
            mask &= roi_mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                area = cv2.contourArea(c)
                if area < 90 or area > w * h * 1.1:
                    continue
                if "max_saturation" in cfg["colors"][self.names[idx]]:
                    center = c[:, 0].mean(axis=0)
                    if (
                        x - 0.15 * w < center[0] < x + 1.15 * w
                        and y - 0.4 * h < center[1] < y + 1.1 * h
                        and area < w * h * 0.8
                    ):
                        cv2.drawContours(foreground, [c], -1, 255, -1)
                for q in quad_proposals(c):
                    self._add_quad(q)
        # Closed edge cells work for arbitrary facelet permutations.
        edge_barriers = cv2.dilate(self.edges, np.ones((2, 2), np.uint8))
        cells = 255 - edge_barriers
        cells &= roi_mask
        cs, _ = cv2.findContours(cells, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            area = cv2.contourArea(c)
            if not 70 < area < w * h * 0.55:
                continue
            for q in quad_proposals(c):
                self._add_quad(q)
        self.quads.sort(key=lambda q: cv2.contourArea(q), reverse=True)
        self.quads = self.quads[:65]
        self._recover_grids()
        fc, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vertices = np.vstack(fc)
        self.foreground_area = cv2.contourArea(cv2.convexHull(vertices))
        self.foreground_distance = cv2.distanceTransform(255 - foreground, cv2.DIST_L2, 5)

    def _recover_grids(self):
        """Join coplanar cells by integer lattice geometry, without reading cell colors."""
        raw = list(self.quads)
        if len(raw) < 3:
            return
        ref = np.array([[0.47, 0.47], [-0.47, 0.47], [-0.47, -0.47], [0.47, -0.47]], np.float32)
        all_points = np.array(raw, np.float32).reshape(-1, 1, 2)
        recovered = []
        for q in raw:
            H = cv2.getPerspectiveTransform(ref, q.astype(np.float32))
            if abs(np.linalg.det(H)) < 1e-9:
                continue
            local = cv2.perspectiveTransform(all_points, np.linalg.inv(H)).reshape(-1, 4, 2)
            matches = {}
            for idx, cell in enumerate(local):
                center = cell.mean(axis=0)
                grid = np.rint(center).astype(int)
                if np.max(abs(grid)) > 2 or np.linalg.norm(center - grid) > 0.26:
                    continue
                target = ref + grid
                dist = np.linalg.norm(cell[:, None, :] - target[None, :, :], axis=2)
                rr, cc = linear_sum_assignment(dist)
                error = float(dist[rr, cc].mean())
                if error > 0.23:
                    continue
                key = tuple(grid)
                if key not in matches or error < matches[key][0]:
                    matches[key] = (error, target[cc], raw[idx][rr])
            if len(matches) < 3:
                continue
            coords = np.array(list(matches))
            if np.linalg.matrix_rank(coords - coords.mean(axis=0)) < 2:
                continue
            if np.any(coords.max(axis=0) - coords.min(axis=0) > 2):
                continue
            obj = np.vstack([v[1] for v in matches.values()]).astype(np.float32)
            img = np.vstack([v[2] for v in matches.values()]).astype(np.float32)
            refined, inliers = cv2.findHomography(obj, img, cv2.RANSAC, 2.0)
            if refined is None or inliers.sum() < len(obj) * 0.75:
                continue
            for a in range(int(coords[:, 0].max()) - 1, int(coords[:, 0].min()) + 2):
                for b in range(int(coords[:, 1].max()) - 1, int(coords[:, 1].min()) + 2):
                    corners = np.array(
                        [
                            [
                                [a + 1.5, b + 1.5],
                                [a - 1.5, b + 1.5],
                                [a - 1.5, b - 1.5],
                                [a + 1.5, b - 1.5],
                            ]
                        ],
                        np.float32,
                    )
                    full = cv2.perspectiveTransform(corners, refined)[0]
                    recovered.append(ordered_quad(full))
        for q in recovered:
            self._add_quad(q)

    def _add_quad(self, q):
        center = q.mean(axis=0)
        x, y, w, h = self.object_bounds
        if not x - 0.3 * w < center[0] < x + 1.3 * w or not y - 0.3 * h < center[1] < y + 1.3 * h:
            return
        if (
            (q[:, 0] < 1).any()
            or (q[:, 0] > self.w - 2).any()
            or (q[:, 1] < 1).any()
            or (q[:, 1] > self.h - 2).any()
        ):
            return
        lengths = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        if min(lengths) < 7 or max(lengths) / min(lengths) > 7:
            return
        # A proposal must contain some observed cube color, including white.
        cx, cy = np.rint(center).astype(int)
        if self.labels[cy, cx] < 0:
            return
        if any(
            np.linalg.norm(old.mean(axis=0) - center) < 2
            and abs(cv2.contourArea(old) - cv2.contourArea(q)) < 20
            for old in self.quads
        ):
            return
        self.quads.append(q)

    def sample_distance(self, pixels, directions=None):
        plain = map_coordinates(
            self.distance, [pixels[:, 1], pixels[:, 0]], order=1, mode="constant", cval=30.0
        )
        if directions is None:
            return plain
        a = (np.arctan2(directions[:, 1], directions[:, 0]) % np.pi) * self.orientation_bins / np.pi
        bins = np.floor(a).astype(int) % self.orientation_bins
        alpha = a - np.floor(a)
        dist = np.zeros(len(pixels))
        for k in range(self.orientation_bins):
            for b, weight in [(bins, 1 - alpha), ((bins + 1) % self.orientation_bins, alpha)]:
                use = b == k
                if use.any():
                    values = map_coordinates(
                        self.oriented_distance[k],
                        [pixels[use, 1], pixels[use, 0]],
                        order=1,
                        mode="constant",
                        cval=30.0,
                    )
                    dist[use] += weight[use] * values
        return 0.8 * np.minimum(dist, plain + 10) + 0.2 * plain

    def center_color(self, pixel, radius=3):
        x, y = np.rint(pixel).astype(int)
        if not radius <= x < self.w - radius or not radius <= y < self.h - radius:
            return None, 0.0
        patch = self.labels[y - radius : y + radius + 1, x - radius : x + radius + 1]
        counts = np.bincount(patch[patch >= 0], minlength=len(self.names))
        if counts.max() < patch.size * 0.55:
            return None, float(counts.max() / patch.size)
        return self.names[int(counts.argmax())], float(counts.max() / patch.size)
