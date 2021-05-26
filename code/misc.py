
import numpy as np

from os import mkdir, listdir
from os.path import isdir as folder_exists
from os.path import isfile, splitext, basename

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

def get_file_name_base(path):

    file_name = basename(path)

    file_name_base = remove_ext(file_name)

    return file_name_base



