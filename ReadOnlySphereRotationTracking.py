#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:02:52 2026

@author: Taryn
"""

from __future__ import print_function
import sys
import cv2
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import csv
from PIL import Image
import matplotlib.cm as cm








#Finds COM of microsphere structures

def find_particle_center(gray):
    
    gray_work = gray.copy()

    # Ignore top timestamp/info bar
    gray_work[0:120, :] = 255   # make top 120 pixels white
    
    # blur to reduce noise
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # threshold: adjust depending on whether particle is dark or bright
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # find blobs
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Filter out objects that are too big or too small
    good_contours = []

    for c in contours:
        area = cv2.contourArea(c)

        if 20 < area < 900:
            good_contours.append(c)

    if len(good_contours) == 0:
        return None

    largest = max(good_contours, key=cv2.contourArea)

    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None

    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    return x, y