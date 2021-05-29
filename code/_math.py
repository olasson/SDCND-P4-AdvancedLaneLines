"""
This file contains internal, supporting functions for detect.py. They are used to perform some basic math operations specific to the lane line detection.
"""

import numpy as np

def _fit_lanes(centroids, ym, xm):
    """
    Perform a polyfit based on detected centroids

    Inputs
    ----------
    centroids: numpy.ndarray
        Numpy array containing detected centroids. See _centroids.py
    ym, xm: float, float
        Conversion factors: meters per pixels in the y-dir and x-dir respectively

    Outputs
    -------
    left_fit, right_fit: numpy.ndarray, numpy.ndarray
        Arrays containing coefficients for the left and right polynomial
        describing the left and right lane
    """

    # Extract all left x values found
    x_left_values = centroids[:,0] * xm
    
    # Extract all right x values found
    x_right_values = centroids[:,1] * xm

    # Extract all y values found
    y_values = centroids[:,2] * ym

    left_fit = np.polyfit(y_values, x_left_values , 2)
    right_fit = np.polyfit(y_values, x_right_values , 2)

    return left_fit, right_fit

def _fit_point(fit, y_value, n_cols):
    """
    Ensure that the x-value of a point lies within an image bounds

    Inputs
    ----------
    fit: numpy.ndarray, numpy.ndarray
        Arrays containing coefficients descbribing a polynomial
    y_eval: float
        y value for evaluation the polynomial
    n_cols: int
        Number of columns in the image

    Outputs
    -------
    x_value: float
        An x value such that (x_value, y_value) is within image bounds
    """

    x_value = np.clip(fit[0]*y_value**2 + fit[1]*y_value + fit[2], 0, n_cols)

    return x_value

def _compute_mean_distance(p1, p2):
    """
    Compute mean distance between two points

    Inputs
    ----------
    p1, p2: numpy.ndarray, numpy.ndarray
        Arrays containing points on the form [x, y]

    Outputs
    -------
    mean_distance: float
        The mean distance between p1 and p2
    """

    mean_distance = np.sqrt(np.sum((p1 - p2)**2) / len(p1))

    return mean_distance

def _compute_curvature(left_fit, right_fit, y_eval):
    """
    Compute the curvature of the lines described by left_fit and right_fit

    Inputs
    ----------
    left_fit, right_fit: numpy.ndarray, numpy.ndarray
        Arrays containing coefficients for the left and right polynomial
        describing the left and right lane
    y_eval: float
        y value for evaluating the curvature

    Outputs
    -------
    (left_curverad, right_curverad): (float, float)
        Tuple containing the left and right curvature
    """
   
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

    return (left_curverad, right_curverad)

def _compute_deviation(left_fit, right_fit, y_eval, x_eval):

    """
    Compute the deviation from a "reference" polynomial

    Inputs
    ----------
    left_fit, right_fit: numpy.ndarray, numpy.ndarray
        Arrays containing coefficients for the left and right polynomial
        describing the left and right lane
    y_eval: float
        y value for evaluating the deviation
    x_eval: float
         value for evaluating the deviation

    Outputs
    -------
        deviation: float
            The average deviation from the polynomial following the center of the lane
    """
    
    l_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    r_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    center = (l_x + r_x) / 2.0

    deviation = center - x_eval / 2.0
    
    return deviation