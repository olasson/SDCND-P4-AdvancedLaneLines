import numpy as np
import cv2
from collections import deque

IMAGE_MAX = 255

WINDOW_WIDTH = 30
WINDOW_HEIGHT = 80
WINDOW_MARGIN = 35
WINDOW_MAX_SINGAL = WINDOW_WIDTH * WINDOW_HEIGHT * IMAGE_MAX

MIN_POINTS_FIT = 4

MIN_CONFIDENCE = 0.16

NOISE_THRESHOLD = 1000

M_PER_PIXELS_X = 3.7/700
M_PER_PIXELS_Y = 3/110

MIN_LANES_DISTANCE = 510
MAX_LANES_DISTANCE = 890

MAX_DISTANCE_DIFF = 60
MAX_DISTANCE_MEAN_DEVIATION = 80


LANE_IS_OK = 0
LANE_OUT_OF_RANGE = 1
LANE_DEVIATES_FROM_MEAN = 2
LANE_DEVIATES_FROM_PREV_FRAME = 3
LANE_LEFT_NOT_OK = 8
LANE_RIGHT_NOT_OK = 9
LANE_BOTH_NOT_OK = 8 + 9





def _fit_lanes(lanes_centroids, ym, xm):

    # Extract all left x values found
    x_left_values = lanes_centroids[:,0] * xm
    
    # Extract all right x values found
    x_right_values = lanes_centroids[:,1] * xm

    # Extract all y values found
    y_values = lanes_centroids[:,2] * ym

    left_fit = np.polyfit(y_values, x_left_values , 2)
    right_fit = np.polyfit(y_values, x_right_values , 2)

    return left_fit, right_fit

def _fit_point(fit, y, n_cols):
    return np.clip(fit[0]*y**2 + fit[1]*y + fit[2], 0, n_cols)

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

def _compute_mean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2) / len(p1))

def _compute_lane_error_code(centroids, centroids_buffer, confidences, last_lanes_distance):


    left_confidence, right_confidence = np.mean(confidences, axis = 0)


    lane_left_rating = 0
    if left_confidence < MIN_CONFIDENCE:
        lane_left_rating = 8

    lane_right_rating = 0
    if right_confidence < MIN_CONFIDENCE:
        lane_right_rating = 9

    if (lane_left_rating + lane_right_rating) == LANE_BOTH_NOT_OK:
        return LANE_BOTH_NOT_OK
    
    if lane_left_rating == LANE_LEFT_NOT_OK and lane_right_rating == LANE_IS_OK:
        return LANE_LEFT_NOT_OK

    if lane_right_rating == LANE_RIGHT_NOT_OK and lane_left_rating == LANE_IS_OK:
        return LANE_RIGHT_NOT_OK

    lanes_current_distance = _compute_mean_distance(centroids[:,0], centroids[:,1])

    if (lanes_current_distance < MIN_LANES_DISTANCE) or lanes_current_distance > MAX_LANES_DISTANCE:
        return LANE_OUT_OF_RANGE

    if last_lanes_distance is not None:
        if abs(lanes_current_distance - last_lanes_distance) > MAX_DISTANCE_DIFF:
            return LANE_DEVIATES_FROM_PREV_FRAME

    if len(centroids_buffer) > 0:

        mean_centroids = np.mean(centroids_buffer, axis = 0)
        mean_lanes_distance = _compute_mean_distance(mean_centroids[:,0], mean_centroids[:,1])

        if abs(lanes_current_distance - mean_lanes_distance) > MAX_DISTANCE_MEAN_DEVIATION:
            return LANE_DEVIATES_FROM_MEAN

    return LANE_IS_OK

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

def _compute_curvature(left_fit, right_fit, y_eval):

   
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

    return (left_curverad, right_curverad)

def _compute_deviation(left_fit, right_fit, y_eval, x_eval):
    
    l_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    r_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    center = (l_x + r_x) / 2.0
    
    return center - x_eval / 2.0





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

        if (lane_error_code != LANE_IS_OK) and (len(self.centroids_buffer) > 0):
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