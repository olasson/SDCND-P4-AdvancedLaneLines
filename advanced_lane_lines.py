import argparse

import cv2
import numpy as np

from shutil import rmtree
from os.path import join as path_join

from code.misc import file_exists, folder_guard, folder_is_empty, remove_ext, get_file_name_base #, compute_src_and_dst
from code.io import load_images, save_processed_data, load_processed_data
from code.plots import plot_images
from code.calibration import camera_calibrate, save_calibration_data, load_calibration_data
#from code.process import pre_process_image, pre_process_frames, post_process_image
from code.detect import LaneDetector

#from code.draw import draw_lanes

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

    path_pipeline_video_input = args.video
    path_pipeline_video_output = path_join(FOLDER_DATA, 'output_video.mp4')
    path_pipeline_processed_frames = path_join(FOLDER_DATA, get_file_name_base(path_pipeline_video_input) + '.p')

    path_pipeline_images = args.images
    path_pipeline_debug = path_join(FOLDER_DATA, 'debug_pipeline')

    path_cam_calibrate_load = './images/calibration'
    path_cam_calibrate_save = path_join(FOLDER_DATA, 'calibrated.p')
    path_cam_calibrate_debug = path_join(FOLDER_DATA, 'debug_calibrate')

    # Init flags

    flag_run_pipeline_on_images = (path_pipeline_images != '')

    flag_run_pipeline_on_videos = (path_pipeline_video_input != '')

    flag_processed_frames_exists = file_exists(path_pipeline_processed_frames)

    flag_show_images = (path_show_images != '')

    flag_cam_is_calibrated = file_exists(path_cam_calibrate_save)

    flag_debug = args.debug
    flag_clean = args.clean

    # Init values

    mtx = None
    dist = None

    fps = args.fps

    n_rows = args.frame_size[1]
    n_cols = args.frame_size[0]

    n_max_cols = args.n_max_cols

    # Setup

    folder_guard(FOLDER_DATA)
    folder_guard(path_cam_calibrate_debug)
    folder_guard(path_pipeline_debug)

    if flag_clean:
        print(INFO_PREFIX + 'Deleting ' + FOLDER_DATA + ' and all its contents!')
        rmtree(FOLDER_DATA)
        print(INFO_PREFIX + 'Stopping program here! Remove the --clean flag to continue!')
        return

    # Checks

    if flag_show_images and folder_is_empty(path_show_images):
        print(ERROR_PREFIX + '--show_images: The folder: ' + path_show_images + ' is empty!')
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
    else:
        print(INFO_PREFIX + 'Loading calibration data!')
        mtx, dist = load_calibration_data(path_cam_calibrate_save)

    # The result at this point should be that mtx and dist is loaded and ready for use
    if (mtx is None) or (dist is None):
        print(ERROR_PREFIX + 'Camera calibration data is still not loaded properly!')
        return


    if flag_run_pipeline_on_images:
        print(INFO_PREFIX + 'Running pipeline on images!')

        #src, dst = compute_src_and_dst(n_rows, n_cols)

        print(INFO_PREFIX + 'Loading images!')
        images, file_names = load_images(path_pipeline_images)

        debug_path = None

        # Set buffer size to 0 since one image is not necessarily related to another
        lane_detector = LaneDetector(n_rows, n_cols, mtx, dist, buffer_size = 0)

        n_images = len(images)

        images_result = np.zeros((n_images, n_rows, n_cols, 3), dtype = 'uint8')

        for i in range(n_images):

            if flag_debug:
                debug_path = path_join(path_pipeline_debug, remove_ext(file_names[i]))
                folder_guard(debug_path)

            #image_undistorted, gray_warped = pre_process_image(images[i], mtx, dist, src, dst, n_rows, n_cols, debug_path = debug_path)
            lane_image = lane_detector.detect(images[i], debug_path = debug_path)
            images_result[i] = lane_image

        plot_images(images_result, file_names)



    if flag_run_pipeline_on_videos and (not flag_debug):
        print(INFO_PREFIX + 'Running pipeline on video!')

        if flag_cam_is_calibrated or ((mtx is None) and (dist is None)):
            print(INFO_PREFIX + 'Loading calibration data from: ' + path_cam_calibrate_save)
            mtx, dist = load_calibration_data(path_cam_calibrate_save)

        #src, dst = compute_src_and_dst(n_rows, n_cols)

        lane_detector = LaneDetector(n_rows, n_cols, mtx, dist)

        cap = cv2.VideoCapture(path_pipeline_video_input)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if flag_debug:
            frames = np.array((n_frames, n_rows, n_cols, 1), dtype = 'uint8')

        fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
        out = cv2.VideoWriter(path_pipeline_video_output, fourcc, fps, tuple(args.frame_size))
        
        i = 0

        while(cap.isOpened()):

            ret, frame = cap.read()

            if ret:
                i = i + 1
                if i % 50 == 0:
                    print(INFO_PREFIX + 'Frame ' + str(i) + '/' + str(n_frames))

                #image_undistorted, gray_warped = pre_process_image(frame, mtx, dist, src, dst, n_rows, n_cols)

                #left_fit, right_fit, curvature, deviation = lane_detector.detect(frame)

                lane_image = lane_detector.detect(frame)

                #lane_image = post_process_image(image_undistorted, left_fit, right_fit, curvature, deviation, src, dst, n_rows, n_cols)
                
                out.write(lane_image)
            else:
                #k = k + 1
                break

        cap.release()
        out.release()

        if flag_debug:
            print(INFO_PREFIX + 'Saving frames!')

        print('Done processing video!')
        print('Number of frames successfully processed: ', i)
        print('Result is found here: ', path_pipeline_video_output)

    # Show

    if flag_show_images:
        print(INFO_PREFIX + 'Showing images from folder: ' + path_show_images)
        images, file_names = load_images(path_show_images)
        plot_images(images, file_names, n_max_cols = n_max_cols)

    """
    if flag_run_pipeline_on_videos and flag_debug:

        # Step 1: Pre process

        if not flag_processed_frames_exists:
            print(INFO_PREFIX + 'Pre processing video frames for: ' + path_pipeline_video_input)

            #src, dst = compute_src_and_dst(n_rows, n_cols)

            images_undistorted, images_gray = pre_process_frames(path_pipeline_video_input, mtx, dist, src, dst, n_rows, n_cols, args.video_codec)

            save_processed_data(path_pipeline_processed_frames, images_undistorted, images_gray)
        else:
            print(INFO_PREFIX + 'Loading pre processed frames from: '+ path_pipeline_video_input)
            images_undistorted, images_gray = load_processed_data(path_pipeline_processed_frames)

        # Step 2: Detect Lanes

        n_frames = len(images_undistorted)

        lane_detector = LaneDetector(n_rows, n_cols)

        tmp = []

        for i in range(n_frames):

            gray_warped = images_gray[i]
            image_undistorted = images_undistorted[i]

            left_fit, right_fit, curvature, deviation = lane_detector.detect(gray_warped)


            if i % 50 == 0:
                print(INFO_PREFIX + 'Processed frames: ' +  str(i) + '/' + str(n_frames))
    """


main()