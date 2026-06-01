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
# Set Variables
# =============================================================================

# Pixels per micron
convert = 2.31673

# Known bead radii in microns
BIG_R_UM = 10.4      # replace with your actual big bead radius
SMALL_R_UM = 6.7   # replace with your actual small bead radius

# Convert bead radii to pixels for image processing
BIG_R_PX = BIG_R_UM * convert
SMALL_R_PX = SMALL_R_UM * convert

# Crop radius in microns
CROP_RADIUS_UM = 35

# Convert crop radius to pixels
CROP_RADIUS_PX = int(CROP_RADIUS_UM * convert)

print(f"Big bead radius: {BIG_R_UM} um = {BIG_R_PX:.2f} px")
print(f"Small bead radius: {SMALL_R_UM} um = {SMALL_R_PX:.2f} px")
print(f"Crop radius: {CROP_RADIUS_UM} um = {CROP_RADIUS_PX} px")


video_path = (
    "/Users/Taryn/Library/CloudStorage/Box-Box/"
    "PhD_Projects/LinkageStiffness/CV_tracking/"
    "ICRA2025_Dataset/TAEPull_MS19/"
    "TAEPull_MS19_constDirRotSine_vid1.mp4"
)


# =============================================================================
# Load video
# =============================================================================


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
    pair_crop, crop_info = crop_around_center(
    gray,
    pair_center,
    crop_radius=CROP_RADIUS_PX
    )

    crop_thresh = threshold_bead_crop(pair_crop)
    
    # ---------------------------------------------------
    # Display raw crop
    # ---------------------------------------------------
    
    plt.figure(figsize=(5,5))
    plt.imshow(pair_crop, cmap="gray")
    plt.title("Raw crop")
    plt.axis("off")
    plt.show()
    
    # ---------------------------------------------------
    # Display thresholded crop
    # ---------------------------------------------------
    
    plt.figure(figsize=(5,5))
    plt.imshow(crop_thresh, cmap="gray")
    plt.title("Thresholded crop")
    plt.axis("off")
    plt.show()

big_crop, small_crop = get_centers_from_contours(crop_thresh)

if big_crop is not None and small_crop is not None:
    big_crop = refine_big_center_away_from_small(
        big_crop,
        small_crop,
        shift_px=5
    )

if big_crop is None:
    print("Could not split bead pair")
    
else:
    bx, by, big_r = big_crop
    sx, sy, small_r = small_crop
    
    big_x = int(bx + crop_info["x0"])
    big_y = int(by + crop_info["y0"])
    small_x = int(sx + crop_info["x0"])
    small_y = int(sy + crop_info["y0"])
    
    print("Big bead center:")
    print(f"  pixels : ({big_x}, {big_y})")
    print(f"  microns: ({big_x / convert:.2f}, {big_y / convert:.2f})")
    
    print("Small bead center:")
    print(f"  pixels : ({small_x}, {small_y})")
    print(f"  microns: ({small_x / convert:.2f}, {small_y / convert:.2f})")
    
    # ---------------------------------------------------
    # Display centers only
    # ---------------------------------------------------
    
    display = frame.copy()
    
    # Red = big bead center
    cv2.circle(display, (big_x, big_y), 5, (0, 0, 255), 1)

    # Blue = small bead center
    cv2.circle(display, (small_x, small_y), 5, (255, 0, 0), 1)
    
    # Cyan line = marker direction
    cv2.line(display, (big_x, big_y), (small_x, small_y), (255, 255, 0), 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.title("Detected bead centers only")
    plt.axis("off")
    plt.show()
    
    cv2.imwrite("debug_bead_detection.png", display)

#dist = show_distance_transform(crop_thresh)

# #contours, hierarchy = cv2.findContours(
#     crop_thresh,
#     cv2.RETR_TREE,
#     cv2.CHAIN_APPROX_SIMPLE
# )

# print("Number of contours:", len(contours))

# for i, c in enumerate(contours):
#     area = cv2.contourArea(c)
#     print(i, area)
    
# debug = cv2.cvtColor(crop_thresh, cv2.COLOR_GRAY2BGR)

# for i, c in enumerate(contours):

#     (x, y), r = cv2.minEnclosingCircle(c)

#     cv2.circle(debug, (int(x), int(y)), int(r), (0,255,0), 1)

#     cv2.putText(
#         debug,
#         str(i),
#         (int(x), int(y)),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.4,
#         (255,255,255),
#         1
#     )

# plt.figure(figsize=(6,6))
# plt.imshow(debug)
# plt.axis("off")
# plt.show()

# =============================================================================
# Clean up
# =============================================================================

cap.release()