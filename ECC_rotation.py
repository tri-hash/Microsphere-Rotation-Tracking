#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:33:21 2026

@author: Taryn
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ReadOnlySphereRotationTracking import *

# =============================================================================
# Load video
# =============================================================================

video_path = (
    "/Users/Taryn/Library/CloudStorage/Box-Box/"
    "PhD_Projects/LinkageStiffness/CV_tracking/"
    "ICRA2025_Dataset/TAEPull_MS19/"
    "TAEPull_MS19_constDirRotSine_vid1.mp4"
)

cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Could not open video")
    raise SystemExit


# =============================================================================
# Video diagnostics
# =============================================================================

fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps:.2f}")
print(f"Frames: {n_frames}")
print(f"Size: {width} x {height}")


# =============================================================================
# Read first frame
# =============================================================================

ret, frame = cap.read()

if not ret:
    print("Could not read first frame")
    cap.release()
    raise SystemExit

# Convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# =============================================================================
# Detect attached bead pair
#
# Returns:
#   big_x, big_y      -> center of large bead
#   big_r             -> radius of large bead
#   small_x, small_y  -> center of attached small bead
#   marker_angle      -> angle from big bead to small bead
# =============================================================================

# Find center of bead pair
pair_center = find_particle_center(gray)

if pair_center is None:
    print("No bead pair found")

else:
    # Crop around bead-pair COM
    pair_crop, crop_info = crop_around_center(
        gray,
        pair_center,
        crop_radius=60
    )

    # Threshold crop so beads are white and background is black
    crop_thresh = threshold_bead_crop(pair_crop)

    # Split attached pair into big and small bead centers
    big_crop, small_crop = split_pair_with_kmeans(crop_thresh)

    if big_crop is None:
        print("Could not split bead pair")

    else:
        # Convert crop coordinates to full-frame coordinates
        bx, by, big_r = big_crop
        sx, sy, small_r = small_crop

        big_x = int(bx + crop_info["x0"])
        big_y = int(by + crop_info["y0"])

        small_x = int(sx + crop_info["x0"])
        small_y = int(sy + crop_info["y0"])

        print("Big bead center:", big_x, big_y)
        print("Small bead center:", small_x, small_y)

        # Visualize result
        display = frame.copy()

        cv2.circle(display, (big_x, big_y), 4, (0, 0, 255), -1)
        cv2.circle(display, (big_x, big_y), int(big_r), (0, 255, 0), 1)

        cv2.circle(display, (small_x, small_y), 4, (255, 0, 0), -1)
        cv2.circle(display, (small_x, small_y), int(small_r), (255, 0, 0), 1)

        cv2.line(
            display,
            (big_x, big_y),
            (small_x, small_y),
            (255, 255, 0),
            2
        )

        plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        plt.title("Big bead and attached small bead")
        plt.axis("off")
        plt.show()
        
        
# Convert from BGR to RGB for plotting
display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

# Save annotated image
cv2.imwrite(
    "debug_bead_detection.png",
    display
)

print("Saved: debug_bead_detection.png")
#Big sphere frame 1 coors: x = 370, y = 560

# =============================================================================
# Clean up
# =============================================================================

cap.release()