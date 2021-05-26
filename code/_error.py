import numpy as np

from code._math import _compute_mean_distance

# Lane detection error codes
LANE_IS_OK = 0
LANE_OUT_OF_RANGE = 1
LANE_DEVIATES_FROM_MEAN = 2
LANE_DEVIATES_FROM_PREV_FRAME = 3
LANE_LEFT_NOT_OK = 8
LANE_RIGHT_NOT_OK = 9
LANE_BOTH_NOT_OK = 8 + 9

MIN_LANES_DISTANCE = 510
MAX_LANES_DISTANCE = 890

MIN_CONFIDENCE = 0.16

MAX_DISTANCE_DIFF = 60
MAX_DISTANCE_MEAN_DEVIATION = 80

def _infer_lane_error_code(centroids, centroids_buffer, confidences, last_lanes_distance):


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