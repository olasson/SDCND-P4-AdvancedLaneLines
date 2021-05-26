import numpy as np
import cv2
from collections import deque


from code._centroids import (_fit_lanes, _estimate_first_centroid, _estimate_centroids, _compute_curvature, _compute_deviation, 
                             _compute_lane_error_code, _compute_mean_distance)

from code._config import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_MARGIN


M_PER_PIXELS_X = 3.7/700
M_PER_PIXELS_Y = 3/110


class LaneDetector:


    def __init__(self, n_rows, n_cols, buffer_size = 15):

        self.centroids_buffer = deque(maxlen = buffer_size)
        self.last_lanes_distance = None

        self.n_rows = n_rows
        self.n_cols = n_cols

    def detect(self, gray_warped):

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

        lane_error_code = _compute_lane_error_code(centroids, self.centroids_buffer, confidences, self.last_lanes_distance)
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


        return left_fit, right_fit, curvature, deviation