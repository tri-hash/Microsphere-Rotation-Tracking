#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:33:21 2026

@author: Taryn
"""

import cv2
import matplotlib.pyplot as plt

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

if not ret:
    print("Could not read frame")
    cap.release()
    raise SystemExit

print("Frame shape:", frame.shape)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(gray, cmap="gray")
plt.title("First frame")
plt.axis("off")
plt.show()

x = 370
y = 560

r = 100

roi = gray[y-r:y+r, x-r:x+r]

plt.imshow(roi, cmap='gray')
plt.show()

cap.release()