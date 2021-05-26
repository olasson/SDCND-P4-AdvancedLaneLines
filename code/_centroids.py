import numpy as np
import cv2
from collections import deque

from code._math import _fit_lanes, _fit_point

#from code._config import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_MARGIN, IMAGE_MAX

IMAGE_MAX = 255

WINDOW_WIDTH = 30
WINDOW_HEIGHT = 80
WINDOW_MARGIN = 35

WINDOW_MAX_SINGAL = WINDOW_WIDTH * WINDOW_HEIGHT * IMAGE_MAX

MIN_POINTS_FIT = 4

NOISE_THRESHOLD = 1000


def _compute_signal_center(signal, offset, scale_confidence = False):

    #print(signal)

    signal_max = np.max(signal)

    if (not scale_confidence) or signal_max > NOISE_THRESHOLD:

        center = np.argmax(signal) + offset - (WINDOW_WIDTH / 2)

        if scale_confidence:
            confidence = signal_max / WINDOW_MAX_SINGAL
        else:
            confidence = 1.0
    else:
        center = None
        confidence = 0.0
    
    return center, confidence


def _estimate_first_centroid(gray_warped, n_rows, n_cols, signal, centroids_buffer):
 

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
    
    left_sum = np.sum(gray_warped[window_top:, left_min_index:left_max_index], axis=0)
    left_signal = np.convolve(signal, left_sum)
    left_x, left_confidence = _compute_signal_center(left_signal, left_min_index, scale_confidence = False)
    
    right_sum = np.sum(gray_warped[window_top:, right_min_index:right_max_index], axis=0)
    right_signal = np.convolve(signal, right_sum)
    right_x, right_confidence = _compute_signal_center(right_signal, right_min_index, scale_confidence = False)

    # Centroid: (left_x, right_x, y)
    return left_x, right_x,  y, left_confidence, right_confidence

def _compute_signal_centroid(signal, n_cols, prev_center):

    offset = WINDOW_WIDTH / 2

    min_index = int(max(prev_center + offset - WINDOW_MARGIN, 0))
    max_index = int(min(prev_center + offset + WINDOW_MARGIN, n_cols))

    conv_window = signal[min_index:max_index]

    center, confidence = _compute_signal_center(conv_window, min_index, scale_confidence = True)

    return center, confidence

def _estimate_centroids(gray_warped, n_rows, n_cols, signal, prev_left_x, prev_right_x, centroids, layer):

    window_top = int(n_rows - (layer + 1) * WINDOW_HEIGHT)
    window_bottom = int(n_rows - layer * WINDOW_HEIGHT)
    y = int(window_bottom - WINDOW_HEIGHT / 2)

    window_signal = np.sum(gray_warped[window_top:window_bottom, :], axis=0)

    conv_signal = np.convolve(signal, window_signal)

    left_x, left_confidence = _compute_signal_centroid(conv_signal, n_cols, prev_left_x)
    right_x, right_confidence = _compute_signal_centroid(conv_signal, n_cols, prev_right_x)

    # Attempt to handle missing values

    if (left_x is None) or (right_x is None):

        if len(centroids) > MIN_POINTS_FIT:
            
            left_fit, right_fit = _fit_lanes(np.array(centroids), 1, 1)

            if left_x is None:
                #print("Option 1.1")
                left_x = _fit_point(left_fit, y, n_cols)

            if right_x is None:
                #print("Option 1.2")
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




"""
def _decode_lane_error_code(c):

    if c == LANE_IS_OK:
        print("Both lanes are OK!")
    elif c == LANE_OUT_OF_RANGE:
        print("LANE_ERROR: Lane positions are outside the range: ["+str(MIN_LANES_DISTANCE)+','+str(MAX_LANES_DISTANCE)+']!')
    elif c == LANE_DEVIATES_FROM_MEAN:
        print('LANE_ERROR: Lane positions deviates more than ' + str(MAX_DISTANCE_MEAN_DEVIATION) + ' pixels from the mean!')
    elif c == LANE_DEVIATES_FROM_PREV_FRAME:
        print('LANE_ERROR: Lane positions deviates more than ' + str(MAX_DISTANCE_DIFF) + ' pixels from their positions in the previous frame!')
    elif c == LANE_LEFT_NOT_OK:
        print('LANE_ERROR: Left lane not OK!')
    elif c == LANE_RIGHT_NOT_OK:
        print('LANE_ERROR: Right lane not OK!')
    elif c == LANE_BOTH_NOT_OK:
        print('LANE_ERROR: Both lanes not OK!')
"""

