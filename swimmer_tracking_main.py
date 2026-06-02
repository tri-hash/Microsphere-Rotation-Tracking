#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swimmer_tracking_main.py
========================
Main script: detect and track a two-sphere microswimmer across all video frames.

Outputs:
  - tracking_results.csv   : frame-by-frame center positions, pair angle,
                              and per-bead rotation angles
  - debug_frame_NNN.png    : annotated frames (written every DEBUG_INTERVAL frames)
  - angle_timeseries.png   : plot of pair angle and bead rotation over time

Coordinate convention:
  - Image coordinates: x = right, y = down
  - Angles: measured in degrees, atan2(dy, dx) convention
             0° = rightward, positive = clockwise (image coords)
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from ReadOnly_swimmer_functions import (
    find_swimmer_contour,
    crop_around_centroid,
    find_bead_centers,
    assign_big_small,
    extract_bead_patch,
    compute_rotation_angle,
    compute_feature_angle,
    crop_to_full,
    full_to_um,
    pair_angle_deg,
    draw_detection,
)


# =============================================================================
# USER SETTINGS — edit these for each experiment
# =============================================================================

VIDEO_PATH = (
    "/Users/Taryn/Library/CloudStorage/Box-Box/"
    "PhD_Projects/LinkageStiffness/CV_tracking/"
    "ICRA2025_Dataset/TAEPull_MS19/"
    "TAEPull_MS19_constDirRotSine_vid1.mp4"
)

OUTPUT_DIR = "tracking_output"

# Physical calibration
PX_PER_UM = 2.31673        # pixels per micron

# Detection parameters (tuned for TAEPull_MS19 video type)
IGNORE_TOP_PX   = 120      # timestamp bar at top of frame
IGNORE_EDGE_PX  = 30       # black corners at edges
SWIMMER_AREA_MIN = 350     # min contour area for swimmer detection (pixels^2)
SWIMMER_AREA_MAX = 1000    # max contour area for swimmer detection
DARK_THRESHOLD   = 80      # intensity threshold for ring body (full-frame scan)
CROP_THRESHOLD   = 60      # intensity threshold inside crop (finer, for holes)
CROP_RADIUS      = 55      # half-size of swimmer crop (pixels)

# Rotation tracking
PATCH_RADIUS      = 18     # half-size of bead patch for rotation tracking (pixels)
BRIGHT_THRESHOLD  = 200    # threshold for bright feature detection in patch
USE_NCC_ROTATION  = False  # True = slow but precise NCC; False = fast feature angle

# Debug output
DEBUG_INTERVAL    = 50     # write a debug image every N frames (0 = disable)
FRAME_LIMIT       = None   # set to int to process only first N frames (None = all)


# =============================================================================
# Setup
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps        = cap.get(cv2.CAP_PROP_FPS)
n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {frame_w}x{frame_h}, {fps:.2f} fps, {n_frames} frames")
print(f"Output directory: {OUTPUT_DIR}/")

if FRAME_LIMIT:
    n_frames = min(n_frames, FRAME_LIMIT)

# =============================================================================
# Per-frame tracking
# =============================================================================

records = []

# State carried across frames
prev_big_crop   = None   # previous big bead center in crop coords
prev_small_crop = None   # previous small bead center in crop coords
ref_patch_big   = None   # reference patch for big bead rotation (frame 0)
ref_patch_small = None   # reference patch for small bead rotation (frame 0)
ref_angle_big   = None   # feature angle in reference frame (big bead)
ref_angle_small = None   # feature angle in reference frame (small bead)

n_detected  = 0
n_failed    = 0

for frame_idx in range(n_frames):

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Step 1: Find swimmer
    # ------------------------------------------------------------------
    swimmer_contour, swimmer_center = find_swimmer_contour(
        gray,
        ignore_top_px=IGNORE_TOP_PX,
        ignore_edge_px=IGNORE_EDGE_PX,
        area_min=SWIMMER_AREA_MIN,
        area_max=SWIMMER_AREA_MAX,
        dark_threshold=DARK_THRESHOLD,
    )

    if swimmer_center is None:
        records.append({
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "detected": False,
        })
        n_failed += 1
        continue

    # ------------------------------------------------------------------
    # Step 2: Crop around swimmer
    # ------------------------------------------------------------------
    crop, origin = crop_around_centroid(gray, swimmer_center, crop_radius=CROP_RADIUS)

    # ------------------------------------------------------------------
    # Step 3: Find hole centroids (bead centers in crop coords)
    # ------------------------------------------------------------------
    holes = find_bead_centers(crop, dark_threshold=CROP_THRESHOLD)

    if len(holes) < 2:
        records.append({
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "detected": False,
        })
        n_failed += 1
        continue

    # ------------------------------------------------------------------
    # Step 4: Assign big / small bead (with cross-frame continuity)
    # ------------------------------------------------------------------
    big_crop, small_crop = assign_big_small(
        holes,
        prev_big=prev_big_crop,
        prev_small=prev_small_crop,
    )

    if big_crop is None:
        records.append({
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "detected": False,
        })
        n_failed += 1
        continue

    # Convert to full-frame coordinates
    big_full_x,   big_full_y   = crop_to_full(big_crop[0],   big_crop[1],   origin)
    small_full_x, small_full_y = crop_to_full(small_crop[0], small_crop[1], origin)

    # Update state (use the same centers for next-frame matching)
    prev_big_crop   = big_crop
    prev_small_crop = small_crop

    # Convert to microns
    big_um_x,   big_um_y   = full_to_um(big_full_x,   big_full_y,   PX_PER_UM)
    small_um_x, small_um_y = full_to_um(small_full_x, small_full_y, PX_PER_UM)

    # Pair angle (big -> small)
    angle = pair_angle_deg(
        (big_full_x, big_full_y),
        (small_full_x, small_full_y)
    )

    # ------------------------------------------------------------------
    # Step 5 & 6: Rotation tracking
    # ------------------------------------------------------------------
    big_patch,   _ = extract_bead_patch(gray, (big_full_x,   big_full_y),   PATCH_RADIUS)
    small_patch, _ = extract_bead_patch(gray, (small_full_x, small_full_y), PATCH_RADIUS)

    big_rot_angle   = None
    small_rot_angle = None
    big_feat_angle   = None
    small_feat_angle = None

    # Store reference patches from first successfully detected frame
    if ref_patch_big is None and big_patch is not None:
        ref_patch_big   = big_patch.copy()
    if ref_patch_small is None and small_patch is not None:
        ref_patch_small = small_patch.copy()

    # Feature angle method (fast): track bright hole centroid angle
    if small_patch is not None:
        feat_angle_s, feat_center_s = compute_feature_angle(
            small_patch, bright_threshold=BRIGHT_THRESHOLD
        )
        small_feat_angle = feat_angle_s

        # Store reference angle
        if ref_angle_small is None and feat_angle_s is not None:
            ref_angle_small = feat_angle_s

        # Rotation = change in feature angle from reference
        if ref_angle_small is not None and feat_angle_s is not None:
            small_rot_angle = feat_angle_s - ref_angle_small

    if big_patch is not None:
        feat_angle_b, feat_center_b = compute_feature_angle(
            big_patch, bright_threshold=BRIGHT_THRESHOLD
        )
        big_feat_angle = feat_angle_b

        if ref_angle_big is None and feat_angle_b is not None:
            ref_angle_big = feat_angle_b

        if ref_angle_big is not None and feat_angle_b is not None:
            big_rot_angle = feat_angle_b - ref_angle_big

    # NCC rotation method (slow, more precise) — optional
    if USE_NCC_ROTATION:
        if big_patch is not None and ref_patch_big is not None:
            big_rot_angle, _ = compute_rotation_angle(big_patch, ref_patch_big)
        if small_patch is not None and ref_patch_small is not None:
            small_rot_angle, _ = compute_rotation_angle(small_patch, ref_patch_small)

    # ------------------------------------------------------------------
    # Record result
    # ------------------------------------------------------------------
    records.append({
        "frame":            frame_idx,
        "time_s":           frame_idx / fps,
        "detected":         True,
        "big_x_px":         big_full_x,
        "big_y_px":         big_full_y,
        "big_x_um":         big_um_x,
        "big_y_um":         big_um_y,
        "small_x_px":       small_full_x,
        "small_y_px":       small_full_y,
        "small_x_um":       small_um_x,
        "small_y_um":       small_um_y,
        "pair_angle_deg":   angle,
        "big_rot_deg":      big_rot_angle,
        "small_rot_deg":    small_rot_angle,
        "big_feat_angle":   big_feat_angle,
        "small_feat_angle": small_feat_angle,
    })
    n_detected += 1

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------
    if DEBUG_INTERVAL > 0 and frame_idx % DEBUG_INTERVAL == 0:
        display = draw_detection(
            frame,
            (big_full_x, big_full_y),
            (small_full_x, small_full_y),
            pair_angle=angle,
            extra_text=f"frame {frame_idx}",
        )
        debug_path = os.path.join(OUTPUT_DIR, f"debug_frame_{frame_idx:05d}.png")
        cv2.imwrite(debug_path, display)

    if frame_idx % 100 == 0:
        print(f"  Frame {frame_idx}/{n_frames}  detected={n_detected}  failed={n_failed}")

cap.release()
print(f"\nDone. Detected: {n_detected}/{n_frames} frames.")


# =============================================================================
# Save CSV
# =============================================================================

df = pd.DataFrame(records)
csv_path = os.path.join(OUTPUT_DIR, "tracking_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")


# =============================================================================
# Plot results
# =============================================================================

df_ok = df[df["detected"] == True].copy()

if len(df_ok) == 0:
    print("No frames detected — check thresholds and video path.")
else:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: pair angle
    axes[0].plot(df_ok["time_s"], df_ok["pair_angle_deg"], lw=1, color="steelblue")
    axes[0].set_ylabel("Pair angle (°)")
    axes[0].set_title("Big→Small bead linkage angle over time")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: bead rotation (relative to frame 0)
    if df_ok["big_rot_deg"].notna().any():
        axes[1].plot(
            df_ok["time_s"], df_ok["big_rot_deg"],
            lw=1, color="darkorange", label="Big bead (magnetic)"
        )
    if df_ok["small_rot_deg"].notna().any():
        axes[1].plot(
            df_ok["time_s"], df_ok["small_rot_deg"],
            lw=1, color="royalblue", label="Small bead (non-magnetic)"
        )
    axes[1].set_ylabel("Rotation (°)")
    axes[1].set_title("Bead rotation relative to frame 0")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: bead trajectories
    axes[2].plot(
        df_ok["big_x_um"], df_ok["big_y_um"],
        "o-", ms=2, lw=0.5, color="darkorange", label="Big bead"
    )
    axes[2].plot(
        df_ok["small_x_um"], df_ok["small_y_um"],
        "o-", ms=2, lw=0.5, color="royalblue", label="Small bead"
    )
    axes[2].set_xlabel("x (µm)")
    axes[2].set_ylabel("y (µm)")
    axes[2].set_title("Bead center trajectories")
    axes[2].set_aspect("equal")
    axes[2].invert_yaxis()
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "angle_timeseries.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")
    print(f"\nSummary:")
    print(f"  Pair angle range: {df_ok['pair_angle_deg'].min():.1f}° to {df_ok['pair_angle_deg'].max():.1f}°")
    if df_ok["small_rot_deg"].notna().any():
        print(f"  Small bead rotation range: {df_ok['small_rot_deg'].min():.1f}° to {df_ok['small_rot_deg'].max():.1f}°")
    if df_ok["big_rot_deg"].notna().any():
        print(f"  Big bead rotation range: {df_ok['big_rot_deg'].min():.1f}° to {df_ok['big_rot_deg'].max():.1f}°")
