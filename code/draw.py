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