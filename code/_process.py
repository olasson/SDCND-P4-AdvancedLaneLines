import cv2
import numpy as np

#from code.draw import draw_region, draw_lanes, draw_text
from code.io import save_image

from code._config import IMAGE_MAX

SOBEL_KERNEL = 5
#IMAGE_MAX = 255

# Internals

def _threshold_binary(image, thresholds):

    binary_out = np.zeros_like(image)

    binary_out[(image >= thresholds[0]) & (image <= thresholds[1])] = 1

    return binary_out

def _threshold_binary_sobel_abs(sobel, thresholds):

    sobel_tmp= np.absolute(sobel)

    # Scale and convert to 'uint8'
    sobel_abs = np.uint8((IMAGE_MAX * sobel_tmp) / np.max(sobel_tmp))

    binary_out = _threshold_binary(sobel_abs, thresholds)

    return binary_out

def _threshold_binary_sobel_gradmag(sobel_x, sobel_y, thresholds):

    sobel_tmp = np.sqrt((sobel_x * sobel_x) + (sobel_y * sobel_y))

    # Scale and convert to 'uint8'
    alpha = np.max(sobel_tmp) / IMAGE_MAX
    sobel_gradmag = np.uint8(sobel_tmp / alpha)

    binary_out = _threshold_binary(sobel_gradmag, thresholds)

    return binary_out


def _threshold_binary_sobel_dir(sobel_x, sobel_y, thresholds):

    sobel_x_abs = np.absolute(sobel_x)
    sobel_y_abs = np.absolute(sobel_y)

    sobel_dir = np.arctan2(sobel_y_abs, sobel_x_abs)

    binary_out = _threshold_binary(sobel_dir, thresholds)

    return binary_out


def _threshold_binary_hsv_channel(image, c, thresholds):

    color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, c]

    binary_out = _threshold_binary(color_channel, thresholds)

    return binary_out

def _threshold_binary_hls_channel(image, c, thresholds):

    color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, c]

    binary_out = _threshold_binary(color_channel, thresholds)

    return binary_out

def _threshold_binary_lab_channel(image, c, thresholds):

    color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, c]

    binary_out = _threshold_binary(color_channel, thresholds)

    return binary_out

# Gradient

def _threshold_gradient(image):

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

    r_channel_binary = _threshold_binary(image[:, :, 2], [195, IMAGE_MAX])
    l_channel_binary = _threshold_binary_hls_channel(image, 1, [195, IMAGE_MAX])
    s_channel_binary = _threshold_binary_hls_channel(image, 2, [100, IMAGE_MAX])
    v_channel_binary = _threshold_binary_hsv_channel(image, 2, [140, IMAGE_MAX])
    b_channel_binary = _threshold_binary_lab_channel(image, 2, [150, IMAGE_MAX])

    color_binary = np.zeros_like(r_channel_binary)


    color_binary[((b_channel_binary == 1) & (v_channel_binary == 1)) | ((r_channel_binary == 1) & (l_channel_binary == 1)) | ((s_channel_binary == 1) & (v_channel_binary == 1))] = 1

    return color_binary
"""
def gamma_correction(image, gamma):

    table = np.zeros(256)
    for i in np.arange(0, 256):
        table[i] = ((i / 255.0) ** gamma) * 255.0

    gamma_image = cv2.LUT(image, table.astype('uint8'))

    return gamma_image
"""

def _warp_image(image, src, dst, n_rows, n_cols):

    warp_matrix = cv2.getPerspectiveTransform(src, dst)

    image_warped = cv2.warpPerspective(image, warp_matrix, (n_cols, n_rows))

    return image_warped

def _unwarp_image(image, src, dst, n_rows, n_cols):

    unwarped_image = _warp_image(image, dst, src, n_rows, n_cols)

    return unwarped_image

def _compute_src_and_dst(n_rows, n_cols):

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

"""
def pre_process_image(image, mtx, dist, src, dst, n_rows, n_cols, debug_path = None):

    image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    #image_gamma = gamma_correction(image_undistorted, 4.0)

    gradient_binary = threshold_gradient(image_undistorted)
    color_binary = threshold_color(image_undistorted)

    combined_threshold = np.zeros_like(gradient_binary)
    combined_threshold[(gradient_binary == 1) | (color_binary) == 1] = IMAGE_MAX

    gray_warped = warp_image(combined_threshold, src, dst, n_rows, n_cols)

    if debug_path is not None:

        # Create a warped RGB image for src and dst
        image_warped = warp_image(image, src, dst, n_rows, n_cols)

        save_image(debug_path, 'step01_image_undistorted.png', image_undistorted)
        #save_image(debug_path, 'step02_image_gamma.png', image_gamma)
        save_image(debug_path, 'step03_gradient_binary.png', gradient_binary * IMAGE_MAX)
        save_image(debug_path, 'step04_color_binary.png', color_binary * IMAGE_MAX)
        save_image(debug_path, 'step05_combined_threshold.png', combined_threshold)
        save_image(debug_path, 'step06_src.png', draw_region(image, src))
        save_image(debug_path, 'step07_dst.png', draw_region(image_warped, dst))

    return image_undistorted, gray_warped


def pre_process_frames(path, mtx, dist, src, dst, n_rows, n_cols, video_codec = '.mp4'):

    cap = cv2.VideoCapture(path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    images_gray = np.zeros((n_frames, n_rows, n_cols), dtype = 'uint8')
    images_undistorted = np.zeros((n_frames, n_rows, n_cols, 3), dtype = 'uint8')

    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    
    i = 0

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret:

            image_undistorted, gray_warped = pre_process_image(frame, mtx, dist, src, dst, n_rows, n_cols)

            images_gray[i] = gray_warped
            images_undistorted[i] = image_undistorted

            i = i + 1
            if i % 50 == 0:
                print('INFO:pre_process_frames(): Processed frame ' + str(i) + '/' + str(n_frames))
        else:
            break

    cap.release()

    return images_undistorted, images_gray


def post_process_image(image_undistorted, left_fit, right_fit, curvature, deviation, src, dst, n_rows, n_cols, debug_path = None):

    image_tmp = draw_lanes(image_undistorted, n_rows, left_fit, right_fit, marker_width = 20, fill_color = (0, 255, 0))

    image_tmp = unwarp_image(image_tmp, src, dst, n_rows, n_cols)

    lane_image = cv2.addWeighted(image_undistorted, 1.0, image_tmp, 1.0, 0.0)

    draw_text(lane_image, curvature, deviation)

    if debug_path is not None:

        save_image(debug_path, 'step08_lane_image.png', lane_image)

    return lane_image
"""