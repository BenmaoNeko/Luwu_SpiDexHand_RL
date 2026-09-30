#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""
Camera Calibration Script - calibrate camera using 11x8 chessboard (high-quality auto-capture)

Features:
- Covers different image regions (center, four edges)
- Different distances (near, middle, far)
- Different angles (frontal, tilted)
- Strict capture conditions

How to use:
1. Run this script
2. Follow the on-screen prompts to move the chessboard to the required position/angle
3. Capture is automatic once conditions are met
4. After completing all conditions, press 's' to calibrate
5. Press 'q' to quit
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from ctypes import *
import time

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_ROOT = SCRIPT_DIR.parent  # deploy/reorient/
CONFIG_DIR = DEPLOY_ROOT / "config"
DEPLOY_REPO_ROOT = DEPLOY_ROOT.parent.parent  # worktree root
# Make deploy.reorient.* imports work when script is run directly.
sys.path.insert(0, str(DEPLOY_REPO_ROOT))

# MvImport: Hikvision MVS SDK Python bindings.
# System-level dependency (NOT vendored in this repo). Default install path is
# /opt/MVS; override with MVS_PYTHON_PATH env var if installed elsewhere.
_mvs_python_path = os.environ.get("MVS_PYTHON_PATH", "/opt/MVS/Samples/64/Python")
if not os.path.isdir(os.path.join(_mvs_python_path, "MvImport")):
    raise RuntimeError(
        f"MvImport not found at {_mvs_python_path}/MvImport. "
        "Install Hikvision MVS SDK (https://www.hikrobotics.com) or set "
        "MVS_PYTHON_PATH env var to the dir containing MvImport/."
    )
sys.path.insert(0, _mvs_python_path)

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from MvImport.MvCameraControl_class import *

# Camera config loader
from deploy.reorient.lib.camera_config import (
    load_camera_config, get_roi, setup_camera_roi
)

# Chessboard parameters
CHESSBOARD_SIZE = (11, 8)  # Inner corner count (cols, rows)
SQUARE_SIZE = 0.020  # Square side length (meters) — 2cm

# Capture parameters
CAPTURE_INTERVAL = 0.8  # Minimum capture interval (seconds)
MIN_STABLE_FRAMES = 5   # Required stable frame count
MIN_QUALITY_SCORE = 60  # Minimum quality score required


def get_chessboard_metrics(corners, img_shape, objp, K_approx=None):
    """
    Compute various metrics for the chessboard.

    Uses PnP to compute the actual tilt angle (degrees), which is more intuitive
    and stable than an edge-length ratio.
    """
    h, w = img_shape[:2]

    # Center position (normalized 0-1)
    center = corners.mean(axis=0).flatten()
    center_norm = (center[0] / w, center[1] / h)

    # Size (diagonal length, used to estimate distance)
    min_pt = corners.min(axis=0).flatten()
    max_pt = corners.max(axis=0).flatten()
    diagonal = np.sqrt((max_pt[0] - min_pt[0])**2 + (max_pt[1] - min_pt[1])**2)
    size_ratio = diagonal / np.sqrt(w**2 + h**2)  # Relative to image diagonal

    # Use PnP to compute tilt angle (more accurate, more intuitive)
    # Use approximate intrinsics (a reasonable estimate when no calibration is available)
    if K_approx is None:
        # Approximate intrinsics for a short-focal-length lens at close range (~53 deg HFOV).
        # Used for the PnP solve that estimates chessboard tilt; does not affect actual
        # calibration result, only the tilt_angle in the HUD, so a wide-ish midrange is fine.
        fx = fy = w * 1.0
        cx, cy = w / 2, h / 2
        K_approx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Use solvePnP to estimate chessboard pose
    success, rvec, tvec = cv2.solvePnP(
        objp, corners.reshape(-1, 2), K_approx, None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    tilt_angle = 0.0
    if success:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        # Chessboard normal in chessboard frame is [0, 0, 1].
        # After transforming to camera frame: R @ [0, 0, 1] = third column of R.
        normal_in_camera = R[:, 2]
        # Camera optical axis is [0, 0, 1]
        # Compute the angle between them
        cos_angle = abs(normal_in_camera[2])  # Absolute value, treat front/back the same
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        tilt_angle = np.degrees(np.arccos(cos_angle))

    return {
        'center': center_norm,
        'size': size_ratio,
        'tilt': tilt_angle  # Now angle (degrees) rather than ratio
    }


def calculate_image_quality(gray, corners):
    """Compute image quality score (0-100)."""
    # 1. Sharpness — use Laplacian variance
    # Only compute over the chessboard region
    x_min = int(corners[:, :, 0].min())
    x_max = int(corners[:, :, 0].max())
    y_min = int(corners[:, :, 1].min())
    y_max = int(corners[:, :, 1].max())

    # Extend the boundary a bit
    pad = 20
    x_min = max(0, x_min - pad)
    x_max = min(gray.shape[1], x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(gray.shape[0], y_max + pad)

    roi = gray[y_min:y_max, x_min:x_max]
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    sharpness = laplacian.var()

    # Normalized sharpness score (looser thresholds)
    # 200+ excellent, 100-200 good, <100 blurry
    sharpness_score = min(100, sharpness / 5)

    # 2. Contrast — use standard deviation
    contrast = roi.std()
    # std of 40+ scores full marks (looser)
    contrast_score = min(100, contrast / 0.4)

    # 3. Corner quality — gradient magnitude around each corner
    corner_quality = 0
    for pt in corners.reshape(-1, 2):
        x, y = int(pt[0]), int(pt[1])
        if 5 <= x < gray.shape[1]-5 and 5 <= y < gray.shape[0]-5:
            patch = gray[y-5:y+5, x-5:x+5].astype(np.float32)
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            corner_quality += np.sqrt(gx**2 + gy**2).mean()
    corner_quality /= len(corners.reshape(-1, 2))
    # Looser corner score
    corner_score = min(100, corner_quality / 0.3)

    # Overall score (weighted average)
    total_score = sharpness_score * 0.5 + contrast_score * 0.25 + corner_score * 0.25

    return {
        'total': total_score,
        'sharpness': sharpness_score,
        'contrast': contrast_score,
        'corner': corner_score
    }


def check_region(center, region):
    """Check whether the center point lies in the specified region."""
    cx, cy = center
    regions = {
        # Center 60%
        'center': (0.20, 0.80, 0.20, 0.80),
        # Four "edge-biased" regions: the board center must enter the corresponding
        # half, and the board edges should not spill out. Tailored for a large board
        # at short working distance: a strict-corner (quadrant) constraint is
        # unreachable, so we use looser position constraints like "left half" while
        # still providing positional diversity.
        'left':   (0.00, 0.40, 0.20, 0.80),  # Left middle band
        'right':  (0.60, 1.00, 0.20, 0.80),
        'top':    (0.20, 0.80, 0.00, 0.40),  # Top middle band
        'bottom': (0.20, 0.80, 0.60, 1.00),
    }
    x1, x2, y1, y2 = regions[region]
    return x1 <= cx <= x2 and y1 <= cy <= y2


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    print("=" * 60)
    print("Camera Calibration - high-quality auto-capture mode")
    print("=" * 60)
    print(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE*1000:.1f}mm")
    print("=" * 60)

    # Initialize camera
    print("\nInitializing camera...")
    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

    if deviceList.nDeviceNum == 0:
        print("No camera found!")
        return

    cam = MvCamera()
    stDevice = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam.MV_CC_CreateHandle(stDevice)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetFloatValue("ExposureTime", 15000)
    cam.MV_CC_SetFloatValue("Gain", 8.0)
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGB8)

    # Load ROI from config
    _cam_cfg = load_camera_config()
    width, height = setup_camera_roi(cam, _cam_cfg)
    cam.MV_CC_StartGrabbing()
    print(f"Camera ready! Resolution: {width}x{height} (ROI mode)")

    # Prepare calibration data
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    obj_points = []
    img_points = []
    captured_images = []

    # Capture task definitions — use angles (degrees) rather than ratios for clarity.
    # (region, distance range, angle range (degrees), description)
    # Working distance ~20cm (cube reorient task calibration). Board 11x8 inner
    # corners x 20mm squares -> diagonal ~244mm. size_ratio = board diagonal /
    # image diagonal. Each bin tolerates fx uncertainty in 800-1300, with a wide
    # span per bin.
    SIZE_NEAR = (0.70, 0.95)   # ~15cm  (board nearly fills the frame)
    SIZE_MID  = (0.50, 0.75)   # ~20cm  (target working distance)
    SIZE_FAR  = (0.35, 0.55)   # ~25-30cm
    SIZE_ANY  = (0.35, 0.95)   # any distance

    # Angle threshold notes:
    # - Frontal: 0-20 deg (for principal point and distortion center)
    # - Tilted: 20-50 deg (for focal length; too steep and corner detection becomes
    #   unreliable)
    # Task list is tuned for "large board + short working distance (~20cm)":
    # board occupies a large portion of the frame, strict-corner is unreachable,
    # so we use 4 edge-biased regions + center + lots of tilt variation. Tilt
    # diversity is the main constraint on fx/fy/distortion, compensating for the
    # limited image-position coverage.
    tasks = [
        # === Group 1: center + multiple tilts (focal length + distortion center) ===
        ('center', SIZE_MID, (0, 15),  'center-frontal'),
        ('center', SIZE_MID, (15, 30), 'center-slight tilt'),
        ('center', SIZE_MID, (30, 45), 'center-large tilt 1'),
        ('center', SIZE_MID, (30, 45), 'center-large tilt 2 (change direction)'),

        # === Group 2: 4 edge-biased regions (image-position diversity) ===
        ('left',   SIZE_MID, (0, 25),  'left-biased frontal'),
        ('right',  SIZE_MID, (0, 25),  'right-biased frontal'),
        ('top',    SIZE_MID, (0, 25),  'top-biased frontal'),
        ('bottom', SIZE_MID, (0, 25),  'bottom-biased frontal'),

        # === Group 3: 4 edge-biased regions + tilt (combined constraints) ===
        ('left',   SIZE_MID, (15, 40), 'left-biased tilted'),
        ('right',  SIZE_MID, (15, 40), 'right-biased tilted'),
        ('top',    SIZE_MID, (15, 40), 'top-biased tilted'),
        ('bottom', SIZE_MID, (15, 40), 'bottom-biased tilted'),

        # === Group 4: distance variation (distortion constraints) ===
        ('center', SIZE_NEAR, (0, 25),  'center-near'),
        ('center', SIZE_FAR,  (0, 25),  'center-far'),
    ]

    task_completed = [False] * len(tasks)
    current_task = 0

    # State
    last_capture_time = 0
    stable_count = 0
    last_corners = None

    print("\nInstructions:")
    print("  Move the chessboard following on-screen prompts")
    print("  Auto-capture once conditions are met")
    print("  c: force capture current frame")
    print("  n: skip current task")
    print("  s: start calibration (at least 12 tasks completed)")
    print("  q: quit")
    print("")

    stOutFrame = MV_FRAME_OUT()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 100)
        if ret != 0:
            continue

        nH = stOutFrame.stFrameInfo.nHeight
        nW = stOutFrame.stFrameInfo.nWidth
        data = string_at(stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
        bayer = np.frombuffer(data, dtype=np.uint8).reshape(nH, nW)
        gray = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2GRAY)
        color = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)

        # Detect chessboard
        # 1) First try findChessboardCornersSB (sector-based, OpenCV 4+): much more
        #    robust to noise, low contrast, and shadows; includes subpixel refinement;
        #    well-suited to large boards (11x8).
        # 2) On failure, fall back to the standard algorithm, dropping CALIB_CB_FAST_CHECK
        #    — that flag rejects slightly noisy images outright and is a common cause
        #    of "unstable detection".
        # 3) SB occasionally returns found=True with corner count < N*M when the board
        #    is near the image edge (partial detection). Downstream solvePnP will
        #    assert/crash on this, so we validate the corner count here and treat an
        #    incomplete detection as not-found so the fallback path takes over.
        n_expected = CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]

        def _good(found_, corners_):
            return (found_
                    and corners_ is not None
                    and corners_.shape[0] == n_expected)

        found, corners = False, None
        if hasattr(cv2, 'findChessboardCornersSB'):
            try:
                found, corners = cv2.findChessboardCornersSB(
                    gray, CHESSBOARD_SIZE,
                    flags=cv2.CALIB_CB_NORMALIZE_IMAGE
                          + cv2.CALIB_CB_EXHAUSTIVE
                          + cv2.CALIB_CB_LARGER,
                )
            except cv2.error:
                found = False
            if not _good(found, corners):
                found, corners = False, None
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray, CHESSBOARD_SIZE,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not _good(found, corners):
                found, corners = False, None

        display = color.copy()
        corners_refined = None
        metrics = None
        quality = None
        task_match = False

        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners_refined, found)

            metrics = get_chessboard_metrics(corners_refined, gray.shape, objp)
            quality = calculate_image_quality(gray, corners_refined)

            # Check stability
            if last_corners is not None:
                movement = np.mean(np.abs(corners_refined - last_corners))
                if movement < 2.0:  # pixels
                    stable_count += 1
                else:
                    stable_count = 0
            last_corners = corners_refined.copy()

            # Check whether the current task matches
            if current_task < len(tasks):
                region, size_range, tilt_range, desc = tasks[current_task]

                region_ok = check_region(metrics['center'], region)
                size_ok = size_range[0] <= metrics['size'] <= size_range[1]
                tilt_ok = tilt_range[0] <= metrics['tilt'] <= tilt_range[1]
                stable_ok = stable_count >= MIN_STABLE_FRAMES
                time_ok = (time.time() - last_capture_time) > CAPTURE_INTERVAL
                quality_ok = quality['total'] >= MIN_QUALITY_SCORE

                task_match = region_ok and size_ok and tilt_ok

                # Auto-capture — must meet quality requirements
                if task_match and stable_ok and time_ok and quality_ok:
                    obj_points.append(objp)
                    img_points.append(corners_refined)
                    captured_images.append(gray.copy())
                    task_completed[current_task] = True
                    last_capture_time = time.time()
                    stable_count = 0
                    print(f"  [OK] Captured #{len(img_points)}: {desc} (quality: {quality['total']:.0f})")

                    # Move to the next incomplete task
                    while current_task < len(tasks) and task_completed[current_task]:
                        current_task += 1

        # Draw the UI
        # Current task prompt
        if current_task < len(tasks):
            region, size_range, tilt_range, desc = tasks[current_task]
            cv2.putText(display, f"Task {current_task+1}/{len(tasks)}: {desc}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Condition indicators
            if metrics and quality:
                # Region
                region_ok = check_region(metrics['center'], region)
                color_ok = (0, 255, 0) if region_ok else (0, 0, 255)
                cv2.putText(display, f"Region: {'OK' if region_ok else 'Move to ' + region}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 1)

                # Distance
                size_ok = size_range[0] <= metrics['size'] <= size_range[1]
                if metrics['size'] < size_range[0]:
                    size_hint = "Move CLOSER"
                elif metrics['size'] > size_range[1]:
                    size_hint = "Move FARTHER"
                else:
                    size_hint = "OK"
                color_ok = (0, 255, 0) if size_ok else (0, 0, 255)
                cv2.putText(display, f"Distance: {size_hint} ({metrics['size']:.2f})",
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 1)

                # Tilt (now displayed in degrees)
                tilt_ok = tilt_range[0] <= metrics['tilt'] <= tilt_range[1]
                if metrics['tilt'] < tilt_range[0]:
                    tilt_hint = "Tilt MORE"
                elif metrics['tilt'] > tilt_range[1]:
                    tilt_hint = "Tilt LESS"
                else:
                    tilt_hint = "OK"
                color_ok = (0, 255, 0) if tilt_ok else (0, 0, 255)
                cv2.putText(display, f"Tilt: {tilt_hint} ({metrics['tilt']:.0f} deg, need {tilt_range[0]:.0f}-{tilt_range[1]:.0f})",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 1)

                # Stability
                stable_ok = stable_count >= MIN_STABLE_FRAMES
                color_ok = (0, 255, 0) if stable_ok else (255, 165, 0)
                cv2.putText(display, f"Stable: {stable_count}/{MIN_STABLE_FRAMES}",
                           (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 1)

                # Quality score
                quality_ok = quality['total'] >= MIN_QUALITY_SCORE
                if quality['total'] >= 70:
                    q_color = (0, 255, 0)  # green - excellent
                elif quality['total'] >= MIN_QUALITY_SCORE:
                    q_color = (0, 255, 255)  # yellow - acceptable
                else:
                    q_color = (0, 0, 255)  # red - unacceptable
                cv2.putText(display, f"Quality: {quality['total']:.0f} (S:{quality['sharpness']:.0f} C:{quality['contrast']:.0f})",
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 1)

                # Overall status — show which conditions are satisfied
                region_ok = check_region(metrics['center'], region)
                size_ok = size_range[0] <= metrics['size'] <= size_range[1]
                tilt_ok = tilt_range[0] <= metrics['tilt'] <= tilt_range[1]
                stable_ok = stable_count >= MIN_STABLE_FRAMES
                time_ok = (time.time() - last_capture_time) > CAPTURE_INTERVAL

                status = []
                if region_ok: status.append("R")
                if size_ok: status.append("D")
                if tilt_ok: status.append("T")
                if stable_ok: status.append("S")
                if quality_ok: status.append("Q")

                all_ok = region_ok and size_ok and tilt_ok and stable_ok and quality_ok and time_ok
                status_color = (0, 255, 0) if all_ok else (0, 165, 255)
                cv2.putText(display, f"Ready: {'/'.join(status)} {'-> CAPTURE!' if all_ok else ''}",
                           (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            else:
                cv2.putText(display, "Chessboard NOT found",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display, f"ALL TASKS DONE! Press 's' to calibrate",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Progress bar
        completed = sum(task_completed)
        progress = completed / len(tasks)
        bar_width = 300
        cv2.rectangle(display, (10, nH-40), (10 + bar_width, nH-20), (100, 100, 100), -1)
        cv2.rectangle(display, (10, nH-40), (10 + int(bar_width * progress), nH-20), (0, 255, 0), -1)
        cv2.putText(display, f"{completed}/{len(tasks)} tasks",
                   (10 + bar_width + 10, nH-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw region indicator (kept consistent with check_region)
        regions_coords = {
            'center': (int(0.20*nW), int(0.20*nH), int(0.80*nW), int(0.80*nH)),
            'left':   (0,            int(0.20*nH), int(0.40*nW), int(0.80*nH)),
            'right':  (int(0.60*nW), int(0.20*nH), nW,           int(0.80*nH)),
            'top':    (int(0.20*nW), 0,            int(0.80*nW), int(0.40*nH)),
            'bottom': (int(0.20*nW), int(0.60*nH), int(0.80*nW), nH),
        }
        if current_task < len(tasks):
            region = tasks[current_task][0]
            x1, y1, x2, y2 = regions_coords[region]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Show
        cv2.imshow('Camera Calibration (Quality Mode)', display)
        cam.MV_CC_FreeImageBuffer(stOutFrame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and found and corners_refined is not None:
            # Force capture
            obj_points.append(objp)
            img_points.append(corners_refined)
            captured_images.append(gray.copy())
            if current_task < len(tasks):
                task_completed[current_task] = True
                print(f"  [Manual] Captured #{len(img_points)}")
                while current_task < len(tasks) and task_completed[current_task]:
                    current_task += 1
            last_capture_time = time.time()
        elif key == ord('n') and current_task < len(tasks):
            # Skip the current task
            print(f"  Skipped: {tasks[current_task][3]}")
            task_completed[current_task] = True
            while current_task < len(tasks) and task_completed[current_task]:
                current_task += 1
        elif key == ord('s') and len(img_points) >= 12:
            # Start calibration
            print("\n" + "=" * 60)
            print(f"Starting calibration... (using {len(img_points)} images)")
            print("=" * 60)

            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, gray.shape[::-1], None, None
            )

            print(f"\nReprojection error (RMS): {ret:.4f} pixels")

            if ret > 1.0:
                print("WARNING: RMS > 1.0, recommend recalibrating")
            elif ret > 0.5:
                print("Acceptable: RMS in 0.5-1.0")
            else:
                print("Excellent: RMS < 0.5")

            print("\nCamera intrinsics matrix K:")
            print(K)
            print("\nDistortion coefficients dist:")
            print(dist.flatten())

            # Save results to deploy/reorient/config/camera_calibration.npz
            CONFIG_DIR.mkdir(exist_ok=True, parents=True)
            output_file = CONFIG_DIR / "camera_calibration.npz"
            np.savez(str(output_file), K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, rms=ret)
            print(f"\nSaved to: {output_file}")

            # Generate copy-pasteable code
            print("\n" + "=" * 60)
            print("Code to paste into cube_observer.py:")
            print("=" * 60)
            print(f"self.K = np.array([")
            print(f"    [{K[0,0]:.2f}, {K[0,1]:.2f}, {K[0,2]:.2f}],")
            print(f"    [{K[1,0]:.2f}, {K[1,1]:.2f}, {K[1,2]:.2f}],")
            print(f"    [{K[2,0]:.2f}, {K[2,1]:.2f}, {K[2,2]:.2f}]")
            print(f"], dtype=np.float64)")
            d = dist.flatten()
            print(f"self.dist = np.array([{d[0]:.6f}, {d[1]:.6f}, {d[2]:.6f}, {d[3]:.6f}, {d[4]:.6f}])")
            print("=" * 60)
            break

    # Cleanup
    cv2.destroyAllWindows()
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    print("\nDone.")


if __name__ == "__main__":
    main()
