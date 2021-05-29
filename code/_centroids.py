"""
This file contains internal, supporting functions for detect.py. They are used to compute window centroids on the form: (left_x, right_x, y)
"""


import numpy as np
import cv2
from collections import deque

# Custom imports
from code._math import _fit_lanes, _fit_point

# Max value in a uint8 image
IMAGE_MAX = 255

# Window size used to detect lane lines
WINDOW_WIDTH = 30
WINDOW_HEIGHT = 80

# The number of columns on each side of the image to ignore while searching for lanes
WINDOW_MARGIN = 35

# An estimate of the maximum value of the convolution signal
WINDOW_MAX_SINGAL = WINDOW_WIDTH * WINDOW_HEIGHT * IMAGE_MAX

# The minimum number of points required required for a _lane_fit()
MIN_POINTS_FIT = 4

# A minimum value for the convolution signal, discarded as noise if below this threhsold
NOISE_THRESHOLD = 1000

def _compute_signal_center(signal, offset, scale_confidence = False):
    """
    Compute signal center (defined below)

    Inputs
    ----------
    signal : numpy.ndarray
        Convolution signal
    offset: int
        Ensures that the center is relative to the image and not the window
    scale_confidence: bool
        If True and a center is found, scale the confidence relative to WINDOW_MAX_SINGAL
        If False and a center is found, complete confidence is assumed

    Outputs
    -------
    center: (int | None)
        An x-value of a center
        None if not found
    confidence: float
        A float on the interval [0.0, 1.0] describing how confident we are that a lane was found
    """

    signal_max = np.max(signal)

    if (not scale_confidence) or signal_max > NOISE_THRESHOLD:

        # Center = (index containing the biggest value in signal) + (offset to shift center relative to image) - (half window width)
        center = np.argmax(signal) + offset - (WINDOW_WIDTH / 2)

        if scale_confidence:
            confidence = signal_max / WINDOW_MAX_SINGAL
        else:
            confidence = 1.0
    else:
        center = None
        confidence = 0.0
    
    return center, confidence


def _estimate_first_centroid(gray_warped, n_rows, n_cols, window_signal, centroids_buffer):
    """
    Estimate the first centroids in an image

    Inputs
    ----------
    gray_warped : numpy.ndarray
        A single grayscale image that has undergone pre-processing (see detect.py)
    n_rows, n_cols: int, int
        Number of image rows and columns
    window_signal: numpy.ndarray
        1D numpy array of length WINDOW_WIDTH containing ones
    centroids_buffer: numpy.ndarray
        Centroids found in the previous N frams

    Outputs
    -------
    left_x, right_x,  y: int, int, int
        Three integers forming a single centroid, i.e centroid = (left_x, right_x, y)
        Or put differently, three integers representing two points, i.e (left_x, y) and (right_x, y)

    left_confidence, right_confidence: float, float
        Floats on the interval [0.0, 1.0] describing the confidence in left_x and right_x
        See _compute_signal_center
    """
 

    if len(centroids_buffer) > 0:

        prev_centroids = np.array(centroids_buffer)

        prev_left_centroids = prev_centroids[:,:,0]
        prev_right_centroids = prev_centroids[:,:,1]

        left_min_index = int(max(np.min(prev_left_centroids) - WINDOW_MARGIN, 0))
        left_max_index = int(min(np.max(prev_left_centroids) + WINDOW_MARGIN, n_cols))

        right_min_index = int(max(np.min(prev_right_centroids) - WINDOW_MARGIN, 0))
        right_max_index = int(min(np.max(prev_right_centroids) + WINDOW_MARGIN, n_cols))
    else:
        left_min_index = 0
        left_max_index = int(n_cols / 2)

        right_min_index = int(n_cols / 2)
        right_max_index = n_cols

    window_top = int(n_rows * 0.75)
    y = int(n_rows - WINDOW_HEIGHT / 2)

    # Set scale_confidence = False here, since we simply have assume that the first centroids found in an image are good
    
    left_sum = np.sum(gray_warped[window_top:, left_min_index:left_max_index], axis=0)
    left_signal = np.convolve(window_signal, left_sum)
    left_x, left_confidence = _compute_signal_center(left_signal, left_min_index, scale_confidence = False)
    
    right_sum = np.sum(gray_warped[window_top:, right_min_index:right_max_index], axis=0)
    right_signal = np.convolve(window_signal, right_sum)
    right_x, right_confidence = _compute_signal_center(right_signal, right_min_index, scale_confidence = False)

    return left_x, right_x,  y, left_confidence, right_confidence

def _compute_window_center(signal, n_cols, prev_center):

    """
    Compute the center in a window given a previous center

    Inputs
    ----------
    signal : numpy.ndarray
        Convolution signal
    n_cols: int
        Number of columns in the image
    prev_center: int
        The previous x value in a center point, i.e (left_x,y) OR (right_x, y)

    Outputs
    -------
    center: (int | None)
        An x-value of a center
        None if not found
    confidence: float
        A float on the interval [0.0, 1.0] describing how confident we are that a lane was found
        See _compute_signal_center()
    """

    offset = WINDOW_WIDTH / 2

    min_index = int(max(prev_center + offset - WINDOW_MARGIN, 0))
    max_index = int(min(prev_center + offset + WINDOW_MARGIN, n_cols))

    conv_window = signal[min_index:max_index]

    center, confidence = _compute_signal_center(conv_window, min_index, scale_confidence = True)

    return center, confidence

def _estimate_centroids(gray_warped, n_rows, n_cols, window_signal, prev_left_x, prev_right_x, centroids, layer):
    """
    Estimate the first centroids in an image

    Inputs
    ----------
    gray_warped: numpy.ndarray
        A single grayscale image that has undergone pre-processing (see detect.py)
    n_rows, n_cols: int, int
        Number of image rows and columns
    window_signal: numpy.ndarray
        1D numpy array of length WINDOW_WIDTH containing ones
    prev_left_x, prev_right_x: int, int
        Centroid x values found in the previous layer
    centroids: numpy.ndarray
        All centroids found so far
    layer: int
        Iterator on the interval [1, (n_rows / WINDOW_HEIGHT))]
        See detect.py

    Outputs
    -------
    left_x, right_x,  y: int, int, int
        Three integers forming a single centroid, i.e centroid = (left_x, right_x, y)
        Or put differently, three integers representing two points, i.e (left_x, y) and (right_x, y)

    left_confidence, right_confidence: float, float
        Floats on the interval [0.0, 1.0] describing the confidence in left_x and right_x
        See _compute_signal_center()
    """

    window_top = int(n_rows - (layer + 1) * WINDOW_HEIGHT)
    window_bottom = int(n_rows - layer * WINDOW_HEIGHT)
    y = int(window_bottom - WINDOW_HEIGHT / 2)

    window_sum = np.sum(gray_warped[window_top:window_bottom, :], axis=0)

    conv_signal = np.convolve(window_signal, window_sum)

    left_x, left_confidence = _compute_window_center(conv_signal, n_cols, prev_left_x)
    right_x, right_confidence = _compute_window_center(conv_signal, n_cols, prev_right_x)

    # Attempt to handle missing values

    # This method assumes that prev_right_x and prev_right_y is not None, 
    # which should never happen unless _estimate_first_centroid() fails.

    if (left_x is None) or (right_x is None):

        if len(centroids) > MIN_POINTS_FIT:
            
            left_fit, right_fit = _fit_lanes(np.array(centroids), 1, 1)

            if left_x is None:
                left_x = _fit_point(left_fit, y, n_cols)

            if right_x is None:
                right_x = _fit_point(right_fit, y, n_cols)
        else:
            if left_x is None and (right_x is not None):
                left_x = right_x - (prev_right_x - prev_left_x)
            else:
                left_x = prev_left_x

            if right_x is None and (left_x is not None):
                right_x = left_x + (prev_right_x - prev_left_x)
            else:
                right_x = prev_right_x

    return left_x, right_x, y, left_confidence, right_confidence

