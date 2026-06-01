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

pair_center = find_particle_center(gray)

if pair_center is not None:
    pair_crop, crop_info = crop_around_center(gray, pair_center, crop_radius=60)

    circles_crop = find_circles_in_crop(pair_crop)

    circles_full = convert_crop_circles_to_full_frame(circles_crop, crop_info)

    big_bead, small_bead = choose_big_small_beads(circles_full)

    if big_bead is not None:
        big_x, big_y, big_r = big_bead
        small_x, small_y, small_r = small_bead

        print("Big bead:", big_x, big_y, big_r)
        print("Small bead:", small_x, small_y, small_r)
    else:
        print("Could not split bead pair into big/small circles.")
else:
    print("No bead pair found.")
    
display = frame.copy()

cv2.circle(display, (big_x, big_y), big_r, (0, 255, 0), 2)
cv2.circle(display, (big_x, big_y), 4, (0, 0, 255), -1)

cv2.circle(display, (small_x, small_y), small_r, (255, 0, 0), 2)
cv2.circle(display, (small_x, small_y), 4, (255, 0, 0), -1)

plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

pair_crop, crop_info = crop_around_center(gray, pair_center, crop_radius=60)

plt.figure()
plt.imshow(pair_crop, cmap="gray")
plt.title("Raw crop")
plt.axis("off")

_, crop_thresh = cv2.threshold(
    pair_crop,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

plt.figure()
plt.imshow(crop_thresh, cmap="gray")
plt.title("Thresholded crop")
plt.axis("off")
plt.show()

#Big sphere frame 1 coors: x = 370, y = 560

# =============================================================================
# Clean up
# =============================================================================

cap.release()