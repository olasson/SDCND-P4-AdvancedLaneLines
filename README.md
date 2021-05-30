# **Advanced Lane Lines** 

*by olasson*

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*This is a revised version of my Advanced Lane Lines project.*

## Project overview

The majority of the project code is located in the folder `code`:

* [`_centroids.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/_centroids.py)
* [`_draw.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/_draw.py)
* [`_error.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/_error.py)
* [`_math.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/_math.py)
* [`_process.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/code/_process.py)
* [`calibration.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/calibration.py)
* [`detect.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/detect.py)
* [`io.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/io.py)
* [`misc.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/misc.py)
* [`plots.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/code/plots.py)

All files starting with `_` are supporting files for `detect.py` which contains the pipeline implementation. 

The main project script is called [`advanced_lane_lines.py`](https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/advanced_lane_lines.py). It contains the implementation of a very simple command line tool.

The images shown in this readme are found in 

* `images/`

The videos of the pipeline result is found in

* `videos/result/`

## Command line arguments

All arguments are technically optional. The program also checks for "illegal" argument combinations to an extent.

See the main script linked above for a detailed list of supported arguments and their function. I will make an effort to specify which commands were used for different parts of the project.

## Camera Calibration

The code for calibrating the camera is located in `calibration.py`. The project expects to find a set of calibration images located in `./images/calibration/`. When the project is first run, it will automatically check if the calibrating file `calibrated.p` exists or not. If it does not exist, it will calibrate the camera and store the result in `./data/calibrated.p` for future use. 

The camera calibrating starts by preparing a set of "base object points" which corresponds to the (x, y, z) coordinates of the chessboard corners in the real world. To simplify, the calibration assumes that the chessboard is fixed at (x, y, z=0), making the base object points the same for each calibration image. 
    
    ...
    # Prepare object points on the form (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    base_object_points = np.zeros((n_obj_points, 3), np.float32)
    base_object_points[:,:2] = np.mgrid[0:n_icorners_x, 0:n_icorners_y].T.reshape(-1, 2)
    ...

Note that the z-coordinate in each `base_object_points` is zero. Each time a set of corners is detected, a full copy of `base_object_points` will be apended, like so
    
    ...    
    ret, corners = cv2.findChessboardCorners(gray_image, (n_icorners_x, n_icorners_y), None)

    if ret:
        found_corner_points.append(corners)
        found_object_points.append(base_object_points)
    ...

where `corners` is the (x,y) pixel positions of the detected corners in each image. I debugged my camera calibration using the `--debug` flag, which helps to visualize the results of the camera calibration:


<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-T1-P4-AdvancedLaneLines/blob/master/images/calibration_debug.png">
</p>

As can be seen, 3/20 images failed, but the calibration was still successful. 

## Pipeline

### Pre-processing

### Lane Detection

### Post-processing
