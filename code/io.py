import cv2

import pickle

import numpy as np

from os import listdir
from os.path import join as path_join

def load_images(path):

    file_names = sorted(listdir(path))

    n_images = len(file_names)

    n_rows, n_cols, n_channels = cv2.imread(path_join(path, file_names[0])).shape

    images = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = 'uint8')

    for i in range(n_images):
        images[i] = cv2.imread(path_join(path, file_names[i]))

    return images, file_names

def save_image(path, file_name, image):
    cv2.imwrite(path_join(path, file_name), image)

"""
def save_processed_data(path, images_undistorted, images_gray):

    data = {'images_undistorted': images_undistorted, 'images_gray': images_gray} 

    with open(path, mode = 'wb') as f:   
        pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)

def load_processed_data(path):

    with open(path, mode = 'rb') as f:
        data = pickle.load(f)

    images_undistorted = data['images_undistorted']
    images_gray = data['images_gray']

    return images_undistorted, images_gray
"""