"""
This file contains the implementation of the lane detector.
"""

import numpy as np
import cv2
from collections import deque


# Custom imports

from code.io import save_image

from code._draw import _draw_region, _draw_lanes, _draw_text

from code._math import _fit_lanes, _compute_mean_distance, _compute_curvature, _compute_deviation

from code._error import _infer_lane_error_code

from code._centroids import _estimate_first_centroid, _estimate_centroids
from code._centroids import WINDOW_WIDTH, WINDOW_HEIGHT, IMAGE_MAX

from code._process import _threshold_gradient, _threshold_color, _warp_image, _unwarp_image, _compute_src_and_dst

M_PER_PIXELS_X = 3.7/700
M_PER_PIXELS_Y = 3/110

class LaneDetector:
    """
    Implements a simple lane detector. 
    This is implemented as a class, since it makes managing the two internal states easier.
    """

    def __init__(self, n_rows, n_cols, mtx, dist, buffer_size = 15):

        # Lane detector internal states

        self.centroids_buffer = deque(maxlen = buffer_size)
        self.last_lanes_distance = None

        # Constant values

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.mtx = mtx
        self.dist = dist

        src, dst = _compute_src_and_dst(n_rows, n_cols)
        self.src = src
        self.dst = dst

    def detect(self, image, debug_path = None):

        """
        Detect lane lines in an image

        Inputs
        ----------
        image: numpy.ndarray
            A single RGB image
        debug_path: (None | str)
            Path where debug images will be stored

        Outputs
        -------
        lane_image: numpy.ndarray
            A single RGB image with the lane clearly marked (hopefully)

        """

        #--------------------------
        # Part 1: Pre-process image
        #--------------------------

        image_undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        gradient_binary = _threshold_gradient(image_undistorted)
        color_binary = _threshold_color(image_undistorted)

        combined_threshold = np.zeros_like(gradient_binary)
        combined_threshold[(gradient_binary == 1) | (color_binary) == 1] = IMAGE_MAX

        gray_warped = _warp_image(combined_threshold, self.src, self.dst, self.n_rows, self.n_cols)

        #--------------------------
        # Part 2: Lane detection
        #--------------------------

        centroids = []
        confidences = []

        window_signal = np.ones(WINDOW_WIDTH)

        left_x, right_x, y, left_confidence, right_confidence = _estimate_first_centroid(gray_warped, self.n_rows, self.n_cols, window_signal, self.centroids_buffer)

        centroids.append((left_x, right_x, y))
        confidences.append((left_confidence, right_confidence))

        for layer in range(1, int(self.n_rows / WINDOW_HEIGHT)):

            left_x, right_x, y, left_confidence, right_confidence = _estimate_centroids(gray_warped, self.n_rows, self.n_cols, window_signal, 
                                                                                        left_x, right_x, centroids, layer)

            #print(left_confidence, right_confidence)
            centroids.append((left_x, right_x, y))
            confidences.append((left_confidence, right_confidence))

        centroids = np.array(centroids)
        confidences = np.array(confidences)

        lane_error_code = _infer_lane_error_code(centroids, self.centroids_buffer, confidences, self.last_lanes_distance)
        #_decode_lane_error_code(lane_error_code)

        if (lane_error_code != 0) and (len(self.centroids_buffer) > 0):
            centroids = self.centroids_buffer[-1] 

        self.centroids_buffer.append(centroids)

        if len(self.centroids_buffer) > 0:
            self.last_lanes_distance = _compute_mean_distance(centroids[:,0], centroids[:,1])
            # Average frames for smoothing
            centroids = np.average(self.centroids_buffer, axis = 0)

        left_fit, right_fit = _fit_lanes(centroids, 1, 1)
        left_fit_scaled, right_fit_scaled = _fit_lanes(centroids, M_PER_PIXELS_Y, M_PER_PIXELS_X)


        curvature = _compute_curvature(left_fit_scaled, right_fit_scaled, np.max(centroids[:,:2]) * M_PER_PIXELS_Y)
        deviation = _compute_deviation(left_fit_scaled, right_fit_scaled, self.n_rows * M_PER_PIXELS_Y, self.n_cols * M_PER_PIXELS_X)

        #--------------------------
        # Part 3: Post-processing
        #--------------------------

        image_tmp = _draw_lanes(image_undistorted, self.n_rows, left_fit, right_fit)

        image_tmp = _unwarp_image(image_tmp, self.src, self.dst, self.n_rows, self.n_cols)

        lane_image = cv2.addWeighted(image_undistorted, 1.0, image_tmp, 1.0, 0.0)

        _draw_text(lane_image, curvature, deviation)


        if debug_path is not None:

            # Create a warped RGB image for src and dst
            image_warped = _warp_image(image, self.src, self.dst, self.n_rows, self.n_cols)

            # Create a lane image without the filler
            image_raw_lanes = _draw_lanes(image_undistorted, self.n_rows, left_fit, right_fit, fill_color = (0, 0, 0))

            save_image(debug_path, 'step01_image_undistorted.png', image_undistorted)
            save_image(debug_path, 'step02_gradient_binary.png', gradient_binary * IMAGE_MAX)
            save_image(debug_path, 'step03_color_binary.png', color_binary * IMAGE_MAX)
            save_image(debug_path, 'step04_image_raw_lanes.png', image_raw_lanes)
            save_image(debug_path, 'step05_combined_threshold.png', combined_threshold)
            save_image(debug_path, 'step06_src.png', _draw_region(image, self.src))
            save_image(debug_path, 'step07_dst.png', _draw_region(image_warped, self.dst))
            save_image(debug_path, 'step08_lane_image.png', lane_image)


        return lane_image