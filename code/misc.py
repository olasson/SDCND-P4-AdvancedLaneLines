
import numpy as np

from os import mkdir, listdir
from os.path import isdir as folder_exists
from os.path import isfile, splitext

def file_exists(path):

    if path is None:
        return False

    if not isfile(path):
        return False

    return True

def folder_guard(path):
    if not folder_exists(path):
        print('INFO:folder_guard(): Creating folder: ' + path + '...')
        mkdir(path)

def folder_is_empty(path):

    if folder_exists(path):
        return (len(listdir(path)) == 0)
    
    return True

def remove_ext(file_name):

    base_name = splitext(file_name)[0]

    return base_name

def compute_src_and_dst(n_rows, n_cols):

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