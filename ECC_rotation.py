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

video_path = "/Users/Taryn/Library/CloudStorage/Box-Box/PhD_Projects/LinkageStiffness/CV_tracking/ICRA2025_Dataset/TAEPull_MS19/TAEPull_MS19_constDirRotSine_vid1.mp4"

cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Could not open video")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Frames: {n_frames}")
print(f"Size: {width} x {height}")



ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

center = find_particle_center(gray)

if center is not None:
    x, y = center
    print("Particle center:", x, y)

    r = 100
    roi = gray[y-r:y+r, x-r:x+r]

    plt.imshow(roi, cmap="gray")
    plt.title("Centered particle ROI")
    plt.axis("off")
    plt.show()
else:
    print("No particle found")
    
display = frame.copy()

if center is not None:
    x, y = center
    cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
    cv2.circle(display, (x, y), 100, (0, 255, 0), 2)

display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

plt.imshow(display_rgb)
plt.axis("off")
plt.show()

result = find_big_bead_center(gray)

if result is not None:
    x, y, bead_radius = result

    print("Big bead center:", x, y)
    print("Radius:", bead_radius)
    
display = frame.copy()

cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
cv2.circle(display, (x, y), int(bead_radius), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

#x = 370
#y = 560

#r = 100

#roi = gray[y-r:y+r, x-r:x+r]

#plt.imshow(roi, cmap='gray')
#plt.show()

cap.release()