"""
This file contains some basic functions for loading and saving images.
"""

import cv2
import pickle

import numpy as np

# Custom imports

from os import listdir
from os.path import join as path_join

def load_images(path):
    """
    Load a set of images from a folder

    Inputs
    ----------
    path: str
        Path to a folder containing a set of images

    Outputs
    -------
    images: numpy.ndarray
        An array containing a set of images
    file_names: numpy.ndarray
        An array containing the file names of 'images'

    """

    file_names = sorted(listdir(path))

    n_images = len(file_names)

    n_rows, n_cols, n_channels = cv2.imread(path_join(path, file_names[0])).shape

    images = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = 'uint8')

    for i in range(n_images):
        images[i] = cv2.imread(path_join(path, file_names[i]))

    return images, file_names

def save_image(path, file_name, image):
    """
    Wrapper for saving image
    """
    cv2.imwrite(path_join(path, file_name), image)