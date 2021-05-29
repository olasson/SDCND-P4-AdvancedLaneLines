"""
This file contains internal, supporting functions for detect.py. They are used for debugging and actually drawing the detected lane lines.
"""

import cv2
import numpy as np

def _draw_line(image, line, color = [255, 0, 0], thickness = 10):
    """
    Wrapper for cv2.line

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image
    line: numpy.ndarray
        An array containing four values - [x1, y1, x2, y2] - defining a line
    color: list
        A list with three elements defining a color in RGB space
    thickness: int
        How many pixels thick the drawn line will be

    Outputs
    -------
        The original 'image' will be modified with a drawn in line
        NOT a copy
    """

    cv2.line(image, (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])), color = color, thickness = thickness)

def _draw_region(image, points):
    """
    Draw a region defined by four points

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image
    points: numpy.ndarray
        An array containing exactly four points - [p1, p2, p3, p4] where pN = [xN ,yN]

    Outputs
    -------
        image_out: numpy.ndarray
            A copy of 'image' with a region drawn in
    """

    line_1 = np.array([points[0][0], points[0][1], points[2][0], points[2][1]]) # Top left to bottom left
    line_2 = np.array([points[1][0], points[1][1], points[3][0], points[3][1]]) # Top right to bottom right

    line_3 = np.array([points[0][0], points[0][1], points[1][0], points[1][1]]) # Top left to top right
    line_4 = np.array([points[2][0], points[2][1], points[3][0], points[3][1]]) # Bottom left to bottom right

    image_out = np.copy(image)

    _draw_line(image_out, line_1)
    _draw_line(image_out, line_2)
    _draw_line(image_out, line_3)
    _draw_line(image_out, line_4)

    return image_out

def _draw_lanes(image, n_rows, left_fit, right_fit, thickness = 20, fill_color = [0, 255, 0]):
    """
    Draw a region defined by four points

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image
    n_rows: int
        The number of rows in 'image'
    left_fit, right_fit: numpy.ndarray, numpy.ndarray
        Numpy arrays containing polynomial coefficients from np.polyfit
    thickness: int
        How many pixels thick the drawn line will be
    fill_color: list
        List containing three ints describing the RGB color used to fill in between detected lanes


    Outputs
    -------
        image_out: numpy.ndarray
            A copy of 'image' with both lane lines drawn in and the space between them filled with color
    """

    y_vals = range(0, n_rows)

    left_x_vals = left_fit[0] * y_vals * y_vals + left_fit[1] * y_vals + left_fit[2]
    right_x_vals = right_fit[0] * y_vals * y_vals + right_fit[1] * y_vals + right_fit[2]

    image_out = np.zeros_like(image)

    cv2.polylines(image_out, np.int_([list(zip(left_x_vals, y_vals))]), False, (255, 0, 0), thickness)
    cv2.polylines(image_out, np.int_([list(zip(right_x_vals, y_vals))]), False, (0, 0, 255), thickness)

    if fill_color is not None:

        offset = thickness / 2

        inner_x = np.concatenate((left_x_vals + offset, right_x_vals[::-1] - offset), axis = 0)
        inner_y = np.concatenate((y_vals, y_vals[::-1]), axis = 0)

        cv2.fillPoly(image_out, np.int_([list(zip(inner_x, inner_y))]), color = fill_color)

    return image_out

def _draw_text(image, curvature, deviation, font_color = (0, 255, 0)):
    """
    Draw lane line metadata in an image

    Inputs
    ----------
    image : numpy.ndarray
        A single RGB image
    curvature: numpy.ndarray
        Tuple containing the left and right lane line curvature
    deviation: float
        How much the detected lane lines deviates from the center polynomial of the lane
    thickness: int
        How many pixels thick the drawn line will be
    font_color: list
        List containing three ints describing the RGB color used for the text
        


    Outputs
    -------
        image_out: numpy.ndarray
            A copy of 'image' with metadata drawn in
    """
    
    curvature_str1 = 'Left Curvature: ' + str(round(curvature[0], 3)) 
    curvature_str2 = 'Right Curvature: ' + str(round(curvature[1], 3))
    deviation_str = 'Center deviation: ' + str(round(deviation, 3))

    cv2.putText(image, curvature_str1, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2)
    cv2.putText(image, curvature_str2, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2)
    cv2.putText(image, deviation_str, (30, 120), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2)