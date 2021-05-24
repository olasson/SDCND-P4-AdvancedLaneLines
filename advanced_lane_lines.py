import argparse

from shutil import rmtree
from os.path import join as path_join

from code.misc import file_exists, folder_guard, folder_is_empty, remove_ext
from code.io import load_images
from code.plots import plot_images
from code.calibration import camera_calibrate, save_calibration_data, load_calibration_data

INFO_PREFIX = 'INFO:main(): '
WARNING_PREFIX = 'WARNING:main(): '
ERROR_PREFIX = 'ERROR:main(): '

FOLDER_DATA = './data'

def main():

    parser = argparse.ArgumentParser(description = 'Advanced Lane Lines')

    # Run on images

    parser.add_argument(
        '--images',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a folder containing images the pipeline will run on.',
    )

    # Show

    parser.add_argument(
        '--show_images',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a folder containing images.',
    )

    parser.add_argument(
        '--n_max_cols',
        type = int,
        nargs = '+',
        default = 4,
        help = 'The maximum number of columns in the image plot.'
    )

    # Video 

    parser.add_argument(
        '--video',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a folder containing images.',
    )

    parser.add_argument(
        '--frame_size',
        type = int,
        nargs = '+',
        default = [1280, 720],
        help = 'The frame size of the output video on the form [n_cols, n_rows]'
    )

    parser.add_argument(
        '--fps',
        type = int,
        default = 25,
        help = 'The fps of the output video'
    )

    parser.add_argument(
        '--video_codec',
        type = str,
        nargs = '?',
        default = 'mp4v',
        help = 'Output video codec.'
    )

    # Misc

    parser.add_argument(
        '--debug',
        action = 'store_true',
        help = 'Enables debugging.'
    )

    parser.add_argument(
        '--clean',
        action = 'store_true',
        help = 'Deletes the folder: ' + FOLDER_DATA + ' and all its contents.' 
    )

    args = parser.parse_args()

    # Init paths

    path_show_images = args.show_images

    path_cam_calibrate_load = './images/calibration'
    path_cam_calibrate_save = path_join(FOLDER_DATA, 'calibrated.p')
    path_cam_calibrate_debug = path_join(FOLDER_DATA, 'debug_calibrate')

    # Init flags

    flag_show_images = (path_show_images != '')

    flag_cam_is_calibrated = file_exists(path_cam_calibrate_save)

    flag_debug = args.debug
    flag_clean = args.clean

    # Init values

    n_rows = args.frame_size[1]
    n_cols = args.frame_size[0]

    n_max_cols = args.n_max_cols

    # Setup

    folder_guard(FOLDER_DATA)
    folder_guard(path_cam_calibrate_debug)

    if flag_clean:
        print(INFO_PREFIX + 'Deleting ' + FOLDER_DATA + ' and all its contents!')
        rmtree(FOLDER_DATA)
        print(INFO_PREFIX + 'Stopping program here! Remove the --clean flag to continue!')
        return

    # Calibrate

    if not flag_cam_is_calibrated:
        print(INFO_PREFIX + 'Calibrating camera!')

        debug_path = None
        if flag_debug:
            debug_path = path_cam_calibrate_debug

        ret, mtx, dist = camera_calibrate(path_cam_calibrate_load, debug_path = debug_path)

        if not ret:
            print(ERROR_PREFIX + 'Camera calibration failed!')
            return

        save_calibration_data(path_cam_calibrate_save, mtx, dist)
        print(INFO_PREFIX + 'Calibration data saved in location: ' + path_cam_calibrate_save)


    # Show

    if flag_show_images:
        print(INFO_PREFIX + 'Showing images from folder: ' + path_show_images)
        images, file_names = load_images(path_show_images)
        plot_images(images, file_names, n_max_cols = n_max_cols)

main()