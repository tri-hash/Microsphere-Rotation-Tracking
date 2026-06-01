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



import cv2
import numpy as np


# =============================================================================
# Bead-pair tracking helper functions
#
# Workflow:
#   1. find_particle_center()
#        Finds the center of mass of an attached bead pair.
#
#   2. crop_around_center()
#        Crops a small image around that bead-pair center.
#
#   3. find_circles_in_crop()
#        Uses Hough circles to identify circular bead components inside the crop.
#
#   4. convert_crop_circles_to_full_frame()
#        Converts circle coordinates from crop coordinates to full-image coordinates.
#
#   5. choose_big_small_beads()
#        Chooses the larger bead and smaller attached bead.
#
#   6. find_big_bead_center()
#        Older contour-based method for finding one large circular bead.
# =============================================================================


def find_particle_center(gray, ignore_top_px=120, min_area=20, max_area=900):
    """
    Find the center of mass of a dark multi-sphere structure.

    This is useful for locating the attached bead pair as one object.
    It does NOT separate the big bead from the small bead.

    Parameters
    ----------
    gray : 2D numpy array
        Grayscale video frame.

    ignore_top_px : int
        Number of pixels at the top of the image to ignore.

    min_area, max_area : float
        Area limits for keeping dark contours.

    Returns
    -------
    center : tuple or None
        (x, y) center of mass of the largest valid dark object.
        Returns None if no valid object is found.
    """

    gray_work = gray.copy()

    # Ignore timestamp/info bar by setting top region to white background.
    gray_work[0:ignore_top_px, :] = 255

    # Blur to reduce small pixel noise before thresholding.
    blur = cv2.GaussianBlur(gray_work, (9, 9), 0)

    # Convert dark objects to white blobs on black background.
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find connected dark objects.
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    good_contours = []

    # Keep only contours in the expected bead-pair size range.
    for c in contours:
        area = cv2.contourArea(c)

        if min_area < area < max_area:
            good_contours.append(c)

    if len(good_contours) == 0:
        return None

    # Use largest valid object as the bead pair.
    largest = max(good_contours, key=cv2.contourArea)

    # Compute center of mass of bead-pair contour.
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None

    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    return x, y


def crop_around_center(gray, center, crop_radius=60):
    """
    Crop a square region around a known center.

    Used after finding the bead-pair center of mass. The crop is then passed
    to Hough circle detection to split the big and small bead.

    Returns
    -------
    crop : 2D numpy array
        Cropped grayscale image.

    crop_info : dict
        Stores crop boundaries so crop coordinates can be converted back
        to full-frame coordinates.
    """

    x, y = center

    x0 = max(x - crop_radius, 0)
    y0 = max(y - crop_radius, 0)
    x1 = min(x + crop_radius, gray.shape[1])
    y1 = min(y + crop_radius, gray.shape[0])

    crop = gray[y0:y1, x0:x1]

    crop_info = {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1
    }

    return crop, crop_info


def find_circles_in_crop(
    crop,
    dp=1.2,
    min_dist=8,
    param1=50,
    param2=8,
    min_radius=3,
    max_radius=30
):
    """
    Detect circular bead components inside a cropped bead-pair image.

    This is where HoughCircles is used. Running HoughCircles on the small crop
    is more reliable than running it on the entire frame.

    Returns
    -------
    circles : list of tuples
        Each circle is returned as (x, y, r) in CROP coordinates.
        Returns an empty list if no circles are found.
    """

    # Median blur preserves edges better than Gaussian blur for circle detection.
    blur = cv2.medianBlur(crop, 5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype("int")

    return [tuple(c) for c in circles]


def convert_crop_circles_to_full_frame(circles, crop_info):
    """
    Convert circles from crop coordinates back to full-frame coordinates.

    Parameters
    ----------
    circles : list of tuples
        Circles as (x, y, r) in crop coordinates.

    crop_info : dict
        Contains x0 and y0 location of crop in original image.

    Returns
    -------
    circles_full : list of tuples
        Circles as (x, y, r) in full-frame coordinates.
    """

    circles_full = []

    for x, y, r in circles:
        full_x = int(x + crop_info["x0"])
        full_y = int(y + crop_info["y0"])

        circles_full.append((full_x, full_y, int(r)))

    return circles_full


def choose_big_small_beads(circles_full):
    """
    Choose the big bead and small bead from detected circles.

    Currently this assumes:
        - the largest detected circle is the big bead
        - the second largest detected circle is the attached small bead

    Returns
    -------
    big_bead : tuple or None
        (big_x, big_y, big_r)

    small_bead : tuple or None
        (small_x, small_y, small_r)
    """

    if len(circles_full) < 2:
        return None, None

    # Sort circles by radius from largest to smallest.
    circles_sorted = sorted(
        circles_full,
        key=lambda c: c[2],
        reverse=True
    )

    big_bead = circles_sorted[0]
    small_bead = circles_sorted[1]

    return big_bead, small_bead


def find_big_bead_center(gray, ignore_top_px=120):
    """
    Older contour-based method for finding a single large circular bead.

    This can be useful as a fallback, but for attached bead pairs it may fail
    because the big bead and small bead can be detected as one merged object.
    """

    gray_work = gray.copy()

    # Ignore timestamp/info bar.
    gray_work[0:ignore_top_px, :] = 255

    blur = cv2.GaussianBlur(gray_work, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 20 or area > 2000:
            continue

        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity < 0.4:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)

        candidates.append((area, circularity, x, y, radius))

    if len(candidates) == 0:
        return None

    best = max(candidates, key=lambda item: item[0])

    area, circularity, x, y, radius = best

    return int(x), int(y), radius




#Finds COM of multi-sphere structures

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