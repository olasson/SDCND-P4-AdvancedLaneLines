import numpy as np

def _fit_lanes(centroids, ym, xm):

    # Extract all left x values found
    x_left_values = centroids[:,0] * xm
    
    # Extract all right x values found
    x_right_values = centroids[:,1] * xm

    # Extract all y values found
    y_values = centroids[:,2] * ym

    left_fit = np.polyfit(y_values, x_left_values , 2)
    right_fit = np.polyfit(y_values, x_right_values , 2)

    return left_fit, right_fit

def _fit_point(fit, y, n_cols):
    return np.clip(fit[0]*y**2 + fit[1]*y + fit[2], 0, n_cols)

def _compute_mean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2) / len(p1))

def _compute_curvature(left_fit, right_fit, y_eval):

   
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

    return (left_curverad, right_curverad)

def _compute_deviation(left_fit, right_fit, y_eval, x_eval):
    
    l_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    r_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    center = (l_x + r_x) / 2.0
    
    return center - x_eval / 2.0