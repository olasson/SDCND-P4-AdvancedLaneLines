"""
This file contains functions for performing camera calibration.
"""

import cv2
import pickle
import numpy as np


from code.io import load_images, save_image

# https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html 

def camera_calibrate(path, n_icorners_x = 9, n_icorners_y = 6, debug_path = None):
    """
    Perform camera calibration based on a set of images

    Inputs
    ----------
    path: str
        Path to a folder containing a set of calibration images
    icorners_x, n_icorners_y: int, int
        The number of "inner" corners in the x and y dir
    debug_path: (None | str)
        Path where debug images will be stored

    Outputs
    -------
    ret: bool
        Boolean indicating the success (success = True) of the calibration
    mtx: numpy.ndarray
        Camera matrix
    dist: numpy.ndarray
        Camera distortion coefficients

    """

    mtx = None
    dist = None
    ret = False

    images, file_names = load_images(path)

    n_images = len(images)

    n_obj_points = n_icorners_x * n_icorners_y

    # Prepare object points on the form (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    base_object_points = np.zeros((n_obj_points, 3), np.float32)
    base_object_points[:,:2] = np.mgrid[0:n_icorners_x, 0:n_icorners_y].T.reshape(-1, 2)

    found_corner_points = []
    found_object_points = []

    for i in range(n_images):

        image = images[i]
        file_name = file_names[i]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray_image, (n_icorners_x, n_icorners_y), None)

        if ret:
            found_corner_points.append(corners)
            found_object_points.append(base_object_points)
        else:
            print('WARNING:camera_calibrate(): No corner points found for image: ' + file_name)

        # Debug

        if debug_path is not None:
            if not ret:
                file_name = 'FAILED_' + file_name
            cv2.drawChessboardCorners(image, (n_icorners_x, n_icorners_y), corners, ret)
            save_image(debug_path, file_name, image)

    gray_image_shape = gray_image.shape[::-1]
    n_corner_points = len(found_corner_points)

    if  n_corner_points > 0 and (gray_image_shape is not None):
        ret, mtx, dist, _, _ = cv2.calibrateCamera(found_object_points, found_corner_points, gray_image_shape, None, None)

    return ret, mtx, dist


def save_calibration_data(path, mtx, dist):
    """
    Save camera matrix and distortion coefficients as a pickled file

    Inputs
    ----------
    path: str
        Path to where the data will be saved
    mtx: numpy.ndarray
        Camera matrix
    dist: numpy.ndarray
        Camera distortion coefficients

    Outputs
    -------
        A pickled file saved at 'path'

    """

    data = {'mtx': mtx, 'dist': dist} 

    with open(path, mode = 'wb') as f:   
        pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)

def load_calibration_data(path):
    """
    Load camera matrix and distortion coefficients from a pickled file

    Inputs
    ----------
    path: str
        Path to where the data will be saved

    Outputs
    -------
    mtx: numpy.ndarray
        Camera matrix
    dist: numpy.ndarray
        Camera distortion coefficients

    """

    with open(path, mode = 'rb') as f:
        data = pickle.load(f)

    mtx = data['mtx']
    dist = data['dist']

    return mtx, dist