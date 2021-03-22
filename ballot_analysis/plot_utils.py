import cv2 as cv
import numpy as np
import os

import matplotlib as mpl
mpl.use('tkagg')  # Hack to make mpl work with Big Sur
import matplotlib.pyplot as plt


def plot_hough_lines(im, im_output, plot_color):
    """
    Finds Hough lines and overlay them on debug plot

    :param np.array im: Binary image
    :param np.array im_output: Image to add overlays to
    :param tuple plot_color: RGB values for plot overlays
    """
    lines = cv.HoughLinesP(im, 1, np.pi/180, 50, minLineLength=500, maxLineGap=30)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(im_output, (x1, y1), (x2, y2), plot_color, 3)


def plot_boxes(im_boxes, keypoints, coords_col, coords_row, template_shape, plot_col):
    """
    For all the detected maxima from convolution with template,
    plot rectangles around ellipse areas.

    :param np.array im_boxes: Image for debug plots
    :param list keypoints: Coordinates for maxima points
    :param list coords_col: Add starting col from cropped image
    :param list coords_row: Add starting row for cropped image
    :param tuple template_shape: Shape of template
    :param tuple plot_col: Color in RGB for plots
    """
    for keypoint in range(len(keypoints)):
        pt = keypoints[keypoint].pt
        pt = (int(coords_col[0] + np.round(pt[0])), int(coords_row[0] + np.round(pt[1])))
        pt2 = (pt[0] + template_shape[1], pt[1] + template_shape[0])
        cv.rectangle(im_boxes, pt, pt2, plot_col, 5)


def plot_ellipses(im_boxes, ellipses, labels):
    """
    For all the detected ellipses, plot rectangles and classification
    result. Filled in magenta, empty in cyan.
    """
    for idx in range(len(ellipses)):
        ellipse = ellipses[idx]
        pt = (ellipse['x'], ellipse['y'])
        pt2 = (ellipse['x'] + ellipse['w'], ellipse['y'] + ellipse['h'])
        plot_col = (255, 255, 0)
        if labels[idx] == 1:
            plot_col = (255, 0, 255)
        cv.rectangle(im_boxes, pt, pt2, plot_col, 5)


def write_debug_profile(im_vert, output_dir, im_name):
    debug_name = os.path.join(output_dir, im_name + "_vert_profile.png")
    profile_vert = np.mean(im_vert, 0)
    plt.plot(np.arange(len(profile_vert)), profile_vert)
    fig_save = plt.gcf()
    fig_save.savefig(debug_name, bbox_inches='tight')
    plt.close(fig_save)
