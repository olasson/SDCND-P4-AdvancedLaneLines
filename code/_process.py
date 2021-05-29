"""
This file contains internal, supporting functions for detect.py. They are used for preparing the image for lane detection.
"""

import cv2
import numpy as np

IMAGE_MAX = 255

SOBEL_KERNEL = 5

# Internals

def _threshold_binary(image, thresholds):
    """
    Apply binary thresholding to an image

    Inputs
    ----------
    image: numpy.ndarray
        A single image, grayscale or RGB
    thresholds: numpy.ndarray
        A list containing a min and max value for thresholding

    Outputs
    -------
        binary_out: numpy.ndarray
            Binary thresholded image with same x,y dimensions as image
    """

    binary_out = np.zeros_like(image)

    binary_out[(image >= thresholds[0]) & (image <= thresholds[1])] = 1

    return binary_out

def _threshold_binary_sobel_abs(sobel, thresholds):
    """
    Apply binary thresholding to a sobel derivative

    Inputs
    ----------
    sobel: numpy.ndarray
        Sobel derivative
    thresholds: numpy.ndarray
        A list containing a min and max value for thresholding

    Outputs
    -------
        binary_out: numpy.ndarray
            Binary thresholded image with same x,y dimensions as image
    """

    sobel_tmp= np.absolute(sobel)

    # Scale and convert to 'uint8'
    sobel_abs = np.uint8((IMAGE_MAX * sobel_tmp) / np.max(sobel_tmp))

    binary_out = _threshold_binary(sobel_abs, thresholds)

    return binary_out

def _threshold_binary_sobel_gradmag(sobel_x, sobel_y, thresholds):
    """
    Apply binary thresholding to the sobel gradient magnitude

    Inputs
    ----------
    sobel_x, sobel_y: numpy.ndarray, numpy.ndarray
        Sobel derivatives in x-dir and y-dir
    thresholds: numpy.ndarray
        A list containing a min and max value for thresholding

    Outputs
    -------
        binary_out: numpy.ndarray
            Binary thresholded image with same x,y dimensions as image
    """

    sobel_tmp = np.sqrt((sobel_x * sobel_x) + (sobel_y * sobel_y))

    # Scale and convert to 'uint8'
    alpha = np.max(sobel_tmp) / IMAGE_MAX
    sobel_gradmag = np.uint8(sobel_tmp / alpha)

    binary_out = _threshold_binary(sobel_gradmag, thresholds)

    return binary_out


def _threshold_binary_sobel_dir(sobel_x, sobel_y, thresholds):
    """
    Apply binary thresholding to the sobel direction

    Inputs
    ----------
    sobel_x, sobel_y: numpy.ndarray, numpy.ndarray
        Sobel derivatives in x-dir and y-dir
    thresholds: numpy.ndarray
        A list containing a min and max value for thresholding

    Outputs
    -------
        binary_out: numpy.ndarray
            Binary thresholded image with same x,y dimensions as image
    """

    sobel_x_abs = np.absolute(sobel_x)
    sobel_y_abs = np.absolute(sobel_y)

    sobel_dir = np.arctan2(sobel_y_abs, sobel_x_abs)

    binary_out = _threshold_binary(sobel_dir, thresholds)

    return binary_out


def _threshold_binary_hsv_channel(image, c, thresholds):
    """
    Apply binary thresholding a HSV color channel

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image
    c: int
        The index of the color channel

    Outputs
    -------
        binary_out: numpy.ndarray
            Binary thresholded image with same x,y dimensions as image
    """

    color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, c]

    binary_out = _threshold_binary(color_channel, thresholds)

    return binary_out

def _threshold_binary_hls_channel(image, c, thresholds):
    """
    Apply binary thresholding a HLS color channel

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image
    c: int
        The index of the color channel

    Outputs
    -------
        binary_out: numpy.ndarray
            Binary thresholded image with same x,y dimensions as image
    """
    color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, c]

    binary_out = _threshold_binary(color_channel, thresholds)

    return binary_out

def _threshold_binary_lab_channel(image, c, thresholds):
    """
    Apply binary thresholding a LAB color channel

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image
    c: int
        The index of the color channel

    Outputs
    -------
        binary_out: numpy.ndarray
            Binary thresholded image with same x,y dimensions as image
    """

    color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, c]

    binary_out = _threshold_binary(color_channel, thresholds)

    return binary_out

# Gradient

def _threshold_gradient(image):
    """
    Apply gradient thresholding to an image

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image

    Outputs
    -------
        gradient_binary: numpy.ndarray
            Composite thresholded image with same x,y dimensions as image
    """

    gray_image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize = SOBEL_KERNEL)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize = SOBEL_KERNEL)

    sobel_x_binary = _threshold_binary_sobel_abs(sobel_x, [15, IMAGE_MAX])
    sobel_y_binary = _threshold_binary_sobel_abs(sobel_y, [25, IMAGE_MAX])

    sobel_gradmag_binary = _threshold_binary_sobel_gradmag(sobel_x, sobel_y, [40, IMAGE_MAX])
    sobel_dir_binary = _threshold_binary_sobel_dir(sobel_x, sobel_y, [0.7, 1.3])

    v_channel_binary = _threshold_binary_hsv_channel(image, 2, [180, IMAGE_MAX])

    gradient_binary = np.zeros_like(sobel_x_binary)

    gradient_binary[(((sobel_x_binary == 1) & (sobel_y_binary == 1)) | (sobel_dir_binary == 1)) & (sobel_gradmag_binary == 1) & (v_channel_binary == 1)] = 1

    return gradient_binary

# Color

def _threshold_color(image):
    """
    Apply color thresholding to an image

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB image

    Outputs
    -------
        color_binary: numpy.ndarray
            Composite thresholded image with same x,y dimensions as image
    """

    r_channel_binary = _threshold_binary(image[:, :, 2], [195, IMAGE_MAX])
    l_channel_binary = _threshold_binary_hls_channel(image, 1, [195, IMAGE_MAX])
    s_channel_binary = _threshold_binary_hls_channel(image, 2, [100, IMAGE_MAX])
    v_channel_binary = _threshold_binary_hsv_channel(image, 2, [140, IMAGE_MAX])
    b_channel_binary = _threshold_binary_lab_channel(image, 2, [150, IMAGE_MAX])

    color_binary = np.zeros_like(r_channel_binary)


    color_binary[((b_channel_binary == 1) & (v_channel_binary == 1)) | ((r_channel_binary == 1) & (l_channel_binary == 1)) | ((s_channel_binary == 1) & (v_channel_binary == 1))] = 1

    return color_binary


def _warp_image(image, src, dst, n_rows, n_cols):
    """
    Apply perspective transform (warping) to an image

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB or grayscale image
    src: numpy.ndarray
        An array containing four source points
    dst: numpy.ndarray
        An array containing four destination points
    n_rows: int
        Number of rows in image
    n_cols: int
        Number of columns in image

    Outputs
    -------
        image_warped: numpy.ndarray
            Image with perspective transform applied
    """

    warp_matrix = cv2.getPerspectiveTransform(src, dst)

    image_warped = cv2.warpPerspective(image, warp_matrix, (n_cols, n_rows))

    return image_warped

def _unwarp_image(image, src, dst, n_rows, n_cols):
    """
    Apply "inverse" perspective transform (warping) to an image

    Inputs
    ----------
    image: numpy.ndarray
        A single RGB or grayscale image
    src: numpy.ndarray
        An array containing four source points
    dst: numpy.ndarray
        An array containing four destination points
    n_rows: int
        Number of rows in image
    n_cols: int
        Number of columns in image

    Outputs
    -------
        image_warped: numpy.ndarray
            Image with "inverse" perspective transform applied
    """

    # Flip the order of src and dst
    unwarped_image = _warp_image(image, dst, src, n_rows, n_cols)

    return unwarped_image

def _compute_src_and_dst(n_rows, n_cols):
    """
    Helper function to compute src and dst quickly

    Inputs
    ----------
    n_rows: int
        Number of rows in image
    n_cols: int
        Number of columns in image

    Outputs
    -------
        src: numpy.ndarray
            An array containing four source points
        dst: numpy.ndarray
            An array containing four destination points

    Notes
    -------
        Use _draw_region() from _draw.py to visualize
    """

    # Compute src

    # Pre-computed values from the line: [570, 470, 220, 720]
    left_slope = -0.7142857143 
    left_intercept = 877.142857146 

    # Pre-computed values from the line: [722, 470, 1110, 720]
    right_slope = 0.6443298969
    right_intercept = 4.793814441

    src_top_offset = 0.645 * n_rows
    src_bottom_offset = 0.020 * n_rows

    src = np.float32([[(src_top_offset - left_intercept) / left_slope, src_top_offset], # Top left
                      [(src_top_offset - right_intercept) / right_slope, src_top_offset], # Top right
                      [(n_rows - src_bottom_offset - left_intercept) / left_slope, n_rows - src_bottom_offset], # Bottom left
                      [(n_rows - src_bottom_offset - right_intercept) / right_slope, n_rows - src_bottom_offset]]) # Bottom right


    # Compute dst

    dst_offset = 0.220 * n_cols

    dst = np.float32([[dst_offset, 0], # Top left
                      [n_cols - dst_offset, 0], # Top right
                      [dst_offset, n_rows], # Bottom left
                      [n_cols - dst_offset, n_rows]]) # Bottom right

    return src, dst