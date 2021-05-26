import cv2
import numpy as np

def draw_line(image, line, color = [255, 0, 0], thickness = 10):

    cv2.line(image, (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])), color = color, thickness = thickness)

def draw_region(image, points):

    line_1 = np.array([points[0][0], points[0][1], points[2][0], points[2][1]]) # Top left to bottom left
    line_2 = np.array([points[1][0], points[1][1], points[3][0], points[3][1]]) # Top right to bottom right

    line_3 = np.array([points[0][0], points[0][1], points[1][0], points[1][1]]) # Top left to top right
    line_4 = np.array([points[2][0], points[2][1], points[3][0], points[3][1]]) # Bottom left to bottom right

    image_out = np.copy(image)

    draw_line(image_out, line_1)
    draw_line(image_out, line_2)
    draw_line(image_out, line_3)
    draw_line(image_out, line_4)

    return image_out

def draw_lanes(image, n_rows, left_fit, right_fit, marker_width = 20, fill_color = (0, 255, 0)):

    y_vals = range(0, n_rows)

    left_x_vals = left_fit[0] * y_vals * y_vals + left_fit[1] * y_vals + left_fit[2]
    right_x_vals = right_fit[0] * y_vals * y_vals + right_fit[1] * y_vals + right_fit[2]

    image_out = np.zeros_like(image)

    cv2.polylines(image_out, np.int_([list(zip(left_x_vals, y_vals))]), False, (255, 0, 0), marker_width)
    cv2.polylines(image_out, np.int_([list(zip(right_x_vals, y_vals))]), False, (0, 0, 255), marker_width)

    if fill_color is not None:

        offset = marker_width / 2

        inner_x = np.concatenate((left_x_vals + offset, right_x_vals[::-1] - offset), axis = 0)
        inner_y = np.concatenate((y_vals, y_vals[::-1]), axis = 0)

        cv2.fillPoly(image_out, np.int_([list(zip(inner_x, inner_y))]), color = fill_color)

    return image_out

def draw_text(image, curvature, deviation, font_color = (0, 255, 0)):

    curvature_str1 = 'Left Curvature: ' + str(round(curvature[0], 3)) 
    curvature_str2 = 'Right Curvature: ' + str(round(curvature[1], 3))
    deviation_str = 'Center deviation: ' + str(round(deviation, 3))

    cv2.putText(image, curvature_str1, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2)
    cv2.putText(image, curvature_str2, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2)
    cv2.putText(image, deviation_str, (30, 120), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2)