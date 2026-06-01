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

def threshold_bead_crop(pair_crop):
    """
    Convert bead crop into a binary image.

    Output:
        White = bead material
        Black = background
    """

    _, crop_thresh = cv2.threshold(
        pair_crop,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return crop_thresh


def split_pair_with_kmeans_and_circle_fit(crop_thresh):
    """
    Split attached bead pair with K-means, then fit a circle to each lobe.

    Returns:
        big_bead_crop, small_bead_crop
        each as (x_center, y_center, radius)
    """

    ys, xs = np.where(crop_thresh > 0)

    if len(xs) < 10:
        return None, None

    points = np.column_stack((xs, ys)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        points,
        2,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    beads = []

    for cluster_id in [0, 1]:

        cluster_points = points[labels.ravel() == cluster_id]

        if len(cluster_points) < 10:
            continue

        # Make a binary mask for this lobe
        mask = np.zeros_like(crop_thresh, dtype=np.uint8)

        for px, py in cluster_points.astype(int):
            mask[py, px] = 255

        # Find the boundary of this lobe
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)

        boundary_points = contour[:, 0, :].astype(float)

        # Fit circle to lobe boundary
        cx, cy, r = fit_circle_to_points(boundary_points)

        area = len(cluster_points)

        beads.append((cx, cy, r, area))

    if len(beads) < 2:
        return None, None

    # Larger lobe is big bead
    beads = sorted(beads, key=lambda b: b[3], reverse=True)

    big = beads[0]
    small = beads[1]

    big_bead_crop = (int(big[0]), int(big[1]), float(big[2]))
    small_bead_crop = (int(small[0]), int(small[1]), float(small[2]))

    return big_bead_crop, small_bead_crop


def fit_circle_to_points(points):
    """
    Fit a circle to x,y boundary points using least squares.

    Returns:
        x_center, y_center, radius
    """

    x = points[:, 0]
    y = points[:, 1]

    A = np.column_stack((2*x, 2*y, np.ones(len(points))))
    b = x**2 + y**2

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    x_center = c[0]
    y_center = c[1]
    radius = np.sqrt(c[2] + x_center**2 + y_center**2)

    return x_center, y_center, radius



def split_pair_from_threshold(crop_thresh):
    """
    Split an attached bead pair using distance transform.

    Returns:
        big_bead_crop   = (x, y, radius_estimate)
        small_bead_crop = (x, y, radius_estimate)

    Coordinates are in CROP coordinates, not full-frame coordinates.
    """

    dist = cv2.distanceTransform(crop_thresh, cv2.DIST_L2, 5)

    dist_norm = cv2.normalize(
        dist,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    _, peaks = cv2.threshold(
        dist_norm,
        int(0.05 * dist_norm.max()),
        255,
        cv2.THRESH_BINARY
    )

    peak_contours, _ = cv2.findContours(
        peaks,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    print("Number of distance peaks:", len(peak_contours))

    centers = []

    for p in peak_contours:
        M = cv2.moments(p)

        if M["m00"] == 0:
            continue

        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        radius_estimate = dist[y, x]

        centers.append((x, y, radius_estimate))

    if len(centers) < 2:
        return None, None

    centers = sorted(centers, key=lambda c: c[2], reverse=True)

    big_bead_crop = centers[0]
    small_bead_crop = centers[1]

    return big_bead_crop, small_bead_crop


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


def circle_mask(shape, x, y, r):
    """
    Create a filled circular mask.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
    return mask


def score_two_circle_model(crop_thresh, big_x, big_y, big_r, small_r, angle):
    """
    Score how well a two-circle bead model matches the thresholded crop.

    Higher score = better overlap between model and thresholded bead pixels.
    """

    # Small bead center is predicted from big bead center, radii, and angle.
    distance = 0.75 * (big_r + small_r)

    small_x = big_x + distance * np.cos(angle)
    small_y = big_y + distance * np.sin(angle)

    model = np.zeros_like(crop_thresh, dtype=np.uint8)

    cv2.circle(model, (int(big_x), int(big_y)), int(big_r), 255, -1)
    cv2.circle(model, (int(small_x), int(small_y)), int(small_r), 255, -1)

    overlap = np.logical_and(model > 0, crop_thresh > 0).sum()
    union = np.logical_or(model > 0, crop_thresh > 0).sum()

    if union == 0:
        return 0

    return overlap / union


def fit_fixed_two_circle_model(
    crop_thresh,
    big_r,
    small_r,
    center_guess=None,
    search_radius=12,
    angle_steps=180
    ):
    """
    Fit a fixed-radius two-circle model to an attached bead pair.

    This assumes bead sizes are known and approximately constant.

    Parameters
    ----------
    crop_thresh : 2D binary image
        Thresholded crop. Beads should be white, background black.

    big_r : float
        Big bead radius in pixels.

    small_r : float
        Small bead radius in pixels.

    center_guess : tuple or None
        Approximate big bead center in crop coordinates.
        If None, the crop center is used.

    search_radius : int
        Number of pixels around center_guess to search.

    angle_steps : int
        Number of angles tested from 0 to 2*pi.

    Returns
    -------
    big_bead_crop, small_bead_crop, best_score

    big_bead_crop = (big_x, big_y, big_r)
    small_bead_crop = (small_x, small_y, small_r)
    """

    h, w = crop_thresh.shape

    if center_guess is None:
        cx0 = w // 2
        cy0 = h // 2
    else:
        cx0, cy0 = center_guess

    best_score = -1
    best_result = None

    angles = np.linspace(0, 2 * np.pi, angle_steps, endpoint=False)

    for big_x in range(int(cx0 - search_radius), int(cx0 + search_radius + 1)):
        for big_y in range(int(cy0 - search_radius), int(cy0 + search_radius + 1)):

            # Skip impossible centers near crop edge.
            if big_x < big_r or big_x > w - big_r:
                continue
            if big_y < big_r or big_y > h - big_r:
                continue

            for angle in angles:

                distance = 0.75 * (big_r + small_r)

                small_x = big_x + distance * np.cos(angle)
                small_y = big_y + distance * np.sin(angle)

                # Skip if small bead would leave the crop.
                if small_x < small_r or small_x > w - small_r:
                    continue
                if small_y < small_r or small_y > h - small_r:
                    continue

                score = score_two_circle_model(
                    crop_thresh,
                    big_x,
                    big_y,
                    big_r,
                    small_r,
                    angle
                )

                if score > best_score:
                    best_score = score

                    best_result = (
                        (int(big_x), int(big_y), float(big_r)),
                        (int(small_x), int(small_y), float(small_r)),
                        float(best_score)
                    )

    if best_result is None:
        return None, None, None

    return best_result

def debug_kmeans_masks(crop_thresh):
    ys, xs = np.where(crop_thresh > 0)
    points = np.column_stack((xs, ys)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        points, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    for cluster_id in [0, 1]:
        mask = np.zeros_like(crop_thresh, dtype=np.uint8)
        cluster_points = points[labels.ravel() == cluster_id]

        for px, py in cluster_points.astype(int):
            mask[py, px] = 255

        plt.figure()
        plt.imshow(mask, cmap="gray")
        plt.title(f"K-means cluster {cluster_id}")
        plt.axis("off")
        plt.show()
        
        
def show_distance_transform(crop_thresh):

    dist = cv2.distanceTransform(
        crop_thresh,
        cv2.DIST_L2,
        5
    )

    plt.figure(figsize=(5,5))
    plt.imshow(dist)
    plt.colorbar()
    plt.title("Distance transform")
    plt.axis("off")
    plt.show()

    return dist

def get_centers_from_contours(crop_thresh):
    """
    Uses contours from the thresholded crop to estimate:
    - big bead center from largest filled contour centroid
    - small bead center from small ring contour centroid
    """

    contours, _ = cv2.findContours(
        crop_thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) < 2:
        return None, None

    contour_info = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 5:
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        contour_info.append((area, x, y, c))

    if len(contour_info) < 2:
        return None, None

    # Sort by area largest to smallest
    contour_info = sorted(contour_info, key=lambda item: item[0], reverse=True)

    # Largest contour = big bead region
    big_area, big_x, big_y, big_contour = contour_info[0]

    # Second largest contour = small bead ring
    small_area, small_x, small_y, small_contour = contour_info[1]

    # Radius estimates from area
    big_r = np.sqrt(big_area / np.pi)
    small_r = np.sqrt(small_area / np.pi)

    big_bead_crop = (big_x, big_y, big_r)
    small_bead_crop = (small_x, small_y, small_r)

    return big_bead_crop, small_bead_crop

def get_big_center_from_lower_lobe(crop_thresh):
    """
    Estimate big bead center while ignoring the neck/small-bead connection.

    Uses only pixels below the vertical midpoint of the largest contour.
    """

    contours, _ = cv2.findContours(
        crop_thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(crop_thresh, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    # Keep lower half of the object only
    y_cut = np.percentile(ys, 55)

    keep = ys > y_cut

    xs_keep = xs[keep]
    ys_keep = ys[keep]

    if len(xs_keep) < 10:
        return None

    big_x = int(np.mean(xs_keep))
    big_y = int(np.mean(ys_keep))

    area = len(xs_keep)
    big_r = np.sqrt(area / np.pi) * 1.4

    return big_x, big_y, big_r


def refine_big_center_away_from_small(big_crop, small_crop, shift_px=6):
    """
    Move big-bead center away from the small bead along the bead-pair axis.
    This works for any orientation.
    """

    bx, by, br = big_crop
    sx, sy, sr = small_crop

    dx = bx - sx
    dy = by - sy

    length = np.sqrt(dx**2 + dy**2)

    if length == 0:
        return big_crop

    ux = dx / length
    uy = dy / length

    bx_refined = int(bx + shift_px * ux)
    by_refined = int(by + shift_px * uy)

    return bx_refined, by_refined, br