#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swimmer_functions.py
====================
Helper functions for detecting and tracking a two-sphere microswimmer
in brightfield microscopy video.

Bead appearance in this video type:
  - Each bead appears as a dark ring with a bright central hole.
  - The magnetic (big) bead is more opaque: smaller bright center.
  - The non-magnetic (small) bead is transparent: larger bright center.
  - The bright hole centroid = geometric bead center (sub-pixel stable).
  - The two rings merge into one outer contour at threshold; the two
    interior holes are retrieved as child contours via RETR_TREE.

Detection pipeline (per frame):
  1. find_swimmer_contour()   -- locate the figure-8 blob in the full frame
  2. crop_around_centroid()   -- isolate it with padding
  3. find_bead_centers()      -- threshold + RETR_TREE -> two hole centroids
  4. assign_big_small()       -- label the two centers by contour area

Rotation tracking (per frame, after centers are found):
  5. extract_bead_patch()     -- tight crop around each bead center
  6. compute_patch_angle()    -- track angle of the bright feature inside each patch
                                 relative to a reference patch (frame 0)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants (set these in your main script, not here)
# ---------------------------------------------------------------------------
# IGNORE_TOP_PX  = 120   # timestamp bar height
# IGNORE_EDGE_PX = 30    # black border width
# SWIMMER_AREA_MIN = 350
# SWIMMER_AREA_MAX = 1000
# DARK_THRESHOLD   = 60   # intensity threshold for ring body detection
# CROP_RADIUS      = 55   # half-size of crop around swimmer center (pixels)
# PATCH_RADIUS     = 18   # half-size of rotation patch around each bead center


# ---------------------------------------------------------------------------
# Step 1: Find the swimmer in the full frame
# ---------------------------------------------------------------------------

def find_swimmer_contour(
    gray,
    ignore_top_px=120,
    ignore_edge_px=30,
    area_min=350,
    area_max=1000,
    dark_threshold=80,
):
    """
    Locate the microswimmer (figure-8 dark blob) in a grayscale frame.

    The swimmer is identified as the largest dark contour whose area falls
    within the expected range for a touching bead pair.

    Parameters
    ----------
    gray : 2D ndarray
        Grayscale video frame.
    ignore_top_px : int
        Pixels to blank at the top (timestamp bar).
    ignore_edge_px : int
        Pixels to blank at left/right/bottom edges (black borders).
    area_min, area_max : float
        Contour area filter for the figure-8 object.
    dark_threshold : int
        Intensity threshold. Pixels below this are considered "dark ring".

    Returns
    -------
    contour : ndarray or None
        The swimmer contour, or None if not found.
    centroid : tuple or None
        (cx, cy) centroid of the swimmer contour in full-frame coordinates.
    """
    h, w = gray.shape
    work = gray.copy()

    # Blank out timestamp bar and black border regions
    work[0:ignore_top_px, :] = 255
    work[:, 0:ignore_edge_px] = 255
    work[:, w - ignore_edge_px:] = 255
    work[h - ignore_top_px:, :] = 255

    blur = cv2.GaussianBlur(work, (5, 5), 0)
    binary = (blur < dark_threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area_min < area < area_max:
            x, y, rw, rh = cv2.boundingRect(c)
            aspect = rh / rw if rw > 0 else 0
            # Figure-8 is taller than wide (two beads stacked)
            if 0.7 < aspect < 3.5:
                if area > best_area:
                    best = c
                    best_area = area

    if best is None:
        return None, None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return best, (cx, cy)


# ---------------------------------------------------------------------------
# Step 2: Crop around swimmer centroid
# ---------------------------------------------------------------------------

def crop_around_centroid(gray, centroid, crop_radius=55):
    """
    Return a square crop around the swimmer centroid.

    Parameters
    ----------
    gray : 2D ndarray
        Grayscale frame.
    centroid : tuple
        (cx, cy) center of the swimmer.
    crop_radius : int
        Half-size of the crop in pixels.

    Returns
    -------
    crop : 2D ndarray
        Cropped grayscale image.
    origin : tuple
        (x0, y0) top-left corner of the crop in full-frame coordinates.
        Use this to convert crop coordinates back to full-frame coordinates.
    """
    cx, cy = centroid
    h, w = gray.shape

    x0 = max(cx - crop_radius, 0)
    y0 = max(cy - crop_radius, 0)
    x1 = min(cx + crop_radius, w)
    y1 = min(cy + crop_radius, h)

    crop = gray[y0:y1, x0:x1]
    origin = (x0, y0)

    return crop, origin


# ---------------------------------------------------------------------------
# Step 3: Find bead centers from hole contours
# ---------------------------------------------------------------------------

def find_bead_centers(crop, dark_threshold=60, local_ring_radius=22):
    """
    Find the two bead centers inside a swimmer crop using RETR_TREE.

    Each bead appears as a dark ring. The two rings share one outer contour
    (they touch), but each ring has its own bright interior hole detected as
    a child contour.

    For each detected hole we compute two center estimates:
      - hole centroid: centroid of the bright interior (good for small bead)
      - local ring centroid: centroid of dark pixels within local_ring_radius
        of the hole — this isolates each bead's ring body independently of
        the shared outer contour (good for the big bead)

    Parameters
    ----------
    crop : 2D ndarray
        Grayscale crop containing the swimmer.
    dark_threshold : int
        Intensity threshold for the ring bodies.
    local_ring_radius : int
        Radius (pixels) of the neighborhood used to compute each bead's
        local ring centroid.

    Returns
    -------
    holes : list of dicts with keys:
        'hole_cx', 'hole_cy'   -- centroid of the bright hole (crop coords)
        'local_cx', 'local_cy' -- centroid of dark pixels near this hole
        'hole_area'            -- area of the bright hole
    Returns empty list if fewer than 2 holes are found.
    """
    binary = (crop < dark_threshold).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return []

    dark_mask = binary  # 255 where dark ring pixels are
    h_img, w_img = crop.shape
    yy, xx = np.mgrid[0:h_img, 0:w_img]

    holes = []
    for c, h in zip(contours, hierarchy[0]):
        if h[3] < 0:
            continue  # not a child contour (not a hole)

        hole_area = cv2.contourArea(c)
        if hole_area < 5:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        hcx = M["m10"] / M["m00"]
        hcy = M["m01"] / M["m00"]

        # Local ring centroid: centroid of dark pixels within local_ring_radius
        dist2 = (xx - hcx) ** 2 + (yy - hcy) ** 2
        local = (dist2 <= local_ring_radius ** 2) & (dark_mask > 0)
        n_local = local.sum()
        if n_local > 0:
            lcx = float(xx[local].mean())
            lcy = float(yy[local].mean())
        else:
            lcx, lcy = hcx, hcy

        holes.append({
            "hole_cx": hcx, "hole_cy": hcy, "hole_area": hole_area,
            "local_cx": lcx, "local_cy": lcy,
        })

    return holes


# ---------------------------------------------------------------------------
# Step 4: Assign big and small bead
# ---------------------------------------------------------------------------

def assign_big_small(holes, prev_big=None, prev_small=None):
    """
    Label which hole is the big (magnetic) bead and which is the small bead.

    The big (opaque) bead has a smaller bright hole and a larger ring body.
    The small (transparent) bead has a larger bright hole.

    Centers returned:
      - big bead:   ring body centroid (geometric center, more reliable than
                    its tiny/asymmetric bright hole)
      - small bead: bright hole centroid (large and well-centered)

    Parameters
    ----------
    holes : list of dicts from find_bead_centers()
    prev_big : tuple or None
        (cx, cy) of big bead center from previous frame (crop coords).
    prev_small : tuple or None
        (cx, cy) of small bead center from previous frame (crop coords).

    Returns
    -------
    big_center : (cx, cy) or None   -- ring centroid of big bead
    small_center : (cx, cy) or None -- hole centroid of small bead
    """
    if len(holes) < 2:
        return None, None

    if prev_big is None or prev_small is None:
        # First frame: smaller hole area = big bead (more opaque)
        holes_sorted = sorted(holes, key=lambda h: h["hole_area"])
        big_h   = holes_sorted[0]
        small_h = holes_sorted[1]
    else:
        # Subsequent frames: nearest-neighbor matching to previous centers
        def dist2(h, prev):
            return (h["local_cx"] - prev[0]) ** 2 + (h["local_cy"] - prev[1]) ** 2

        h0, h1 = holes[0], holes[1]
        if dist2(h0, prev_big) + dist2(h1, prev_small) <= dist2(h1, prev_big) + dist2(h0, prev_small):
            big_h, small_h = h0, h1
        else:
            big_h, small_h = h1, h0

    # Big bead: local ring centroid (geometric center, unaffected by tiny hole)
    big_center   = (big_h["local_cx"],  big_h["local_cy"])
    # Small bead: hole centroid (large bright center is reliable)
    small_center = (small_h["hole_cx"], small_h["hole_cy"])

    return big_center, small_center


# ---------------------------------------------------------------------------
# Step 5: Extract a tight patch around each bead for rotation tracking
# ---------------------------------------------------------------------------

def extract_bead_patch(gray, center_full, patch_radius=18):
    """
    Extract a tight grayscale patch around a bead center.

    Parameters
    ----------
    gray : 2D ndarray
        Full grayscale frame.
    center_full : tuple
        (cx, cy) bead center in full-frame coordinates.
    patch_radius : int
        Half-size of the patch.

    Returns
    -------
    patch : 2D ndarray or None
        Grayscale patch, or None if the center is too close to the edge.
    patch_origin : tuple
        (x0, y0) top-left corner of the patch in full-frame coordinates.
    """
    cx, cy = int(round(center_full[0])), int(round(center_full[1]))
    h, w = gray.shape

    x0 = cx - patch_radius
    y0 = cy - patch_radius
    x1 = cx + patch_radius
    y1 = cy + patch_radius

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None, (x0, y0)

    patch = gray[y0:y1, x0:x1]
    return patch, (x0, y0)


# ---------------------------------------------------------------------------
# Step 6: Compute bead rotation angle from patch
# ---------------------------------------------------------------------------

def compute_rotation_angle(patch, ref_patch, search_degrees=180):
    """
    Estimate the rotation of a bead relative to a reference patch.

    Uses normalized cross-correlation between the current patch and
    rotated versions of the reference patch to find the best-fit angle.

    Parameters
    ----------
    patch : 2D ndarray
        Current frame's bead patch.
    ref_patch : 2D ndarray
        Reference bead patch (frame 0).
    search_degrees : float
        Angular search range in degrees (searches ±search_degrees/2).

    Returns
    -------
    best_angle : float
        Rotation angle in degrees (positive = counterclockwise).
    best_score : float
        NCC score at best angle (1.0 = perfect match).
    """
    h, w = ref_patch.shape
    cx, cy = w // 2, h // 2

    ref_f = ref_patch.astype(np.float32)
    patch_f = patch.astype(np.float32)

    best_score = -np.inf
    best_angle = 0.0

    angles = np.linspace(
        -search_degrees / 2, search_degrees / 2, int(search_degrees * 2)
    )

    for angle in angles:
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            ref_f, M_rot, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # Normalized cross-correlation
        num = np.sum((rotated - rotated.mean()) * (patch_f - patch_f.mean()))
        denom = (np.std(rotated) * np.std(patch_f) * rotated.size)
        if denom > 0:
            score = num / denom
            if score > best_score:
                best_score = score
                best_angle = angle

    return best_angle, best_score


def compute_feature_angle(patch, bright_threshold=200):
    """
    Compute the angle of the bright internal feature (reflection/hole)
    relative to the patch center.

    This is fast and works well for the small bead's large bright hole.
    The angle changes as the bead rotates, giving a direct rotation signal.

    Parameters
    ----------
    patch : 2D ndarray
        Grayscale bead patch.
    bright_threshold : int
        Pixels above this intensity are treated as the bright feature.

    Returns
    -------
    angle : float or None
        Angle in degrees from patch center to bright feature centroid.
        None if no bright feature is detected.
    feature_center : tuple or None
        (cx, cy) centroid of the bright feature within the patch.
    """
    bright_mask = (patch > bright_threshold).astype(np.uint8) * 255

    M = cv2.moments(bright_mask)
    if M["m00"] < 10:  # too few bright pixels
        return None, None

    fcx = M["m10"] / M["m00"]
    fcy = M["m01"] / M["m00"]

    patch_cx = patch.shape[1] / 2.0
    patch_cy = patch.shape[0] / 2.0

    dx = fcx - patch_cx
    dy = fcy - patch_cy

    angle = np.degrees(np.arctan2(dy, dx))

    return angle, (fcx, fcy)


# ---------------------------------------------------------------------------
# Coordinate conversion utilities
# ---------------------------------------------------------------------------

def crop_to_full(cx_crop, cy_crop, origin):
    """Convert crop coordinates to full-frame coordinates."""
    return cx_crop + origin[0], cy_crop + origin[1]


def full_to_um(x_px, y_px, px_per_um):
    """Convert pixel coordinates to microns."""
    return x_px / px_per_um, y_px / px_per_um


def pair_angle_deg(big_center, small_center):
    """
    Angle from big bead to small bead in degrees.
    0° = rightward, 90° = downward (image coordinates).
    """
    dx = small_center[0] - big_center[0]
    dy = small_center[1] - big_center[1]
    return np.degrees(np.arctan2(dy, dx))


# ---------------------------------------------------------------------------
# Debug visualisation helper
# ---------------------------------------------------------------------------

def draw_detection(frame, big_full, small_full, pair_angle=None, extra_text=None):
    """
    Draw bead centers and linkage line on a copy of frame.

    Parameters
    ----------
    frame : 3D ndarray
        BGR video frame.
    big_full, small_full : tuple
        (cx, cy) bead centers in full-frame coordinates.
    pair_angle : float or None
        Linkage angle in degrees (drawn as text).
    extra_text : str or None
        Additional label drawn on the frame.

    Returns
    -------
    display : 3D ndarray
        Annotated BGR frame.
    """
    display = frame.copy()

    bx, by = int(round(big_full[0])), int(round(big_full[1]))
    sx, sy = int(round(small_full[0])), int(round(small_full[1]))

    # Big bead: orange dot
    cv2.circle(display, (bx, by), 5, (0, 140, 255), -1)
    cv2.circle(display, (bx, by), 6, (0, 0, 0), 1)

    # Small bead: blue dot
    cv2.circle(display, (sx, sy), 5, (255, 80, 0), -1)
    cv2.circle(display, (sx, sy), 6, (0, 0, 0), 1)

    # Linkage line
    cv2.line(display, (bx, by), (sx, sy), (0, 220, 100), 1)

    if pair_angle is not None:
        cv2.putText(
            display,
            f"angle: {pair_angle:.1f}deg",
            (bx + 8, by - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            display,
            f"angle: {pair_angle:.1f}deg",
            (bx + 8, by - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    if extra_text:
        cv2.putText(
            display, extra_text, (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )
        cv2.putText(
            display, extra_text, (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return display
