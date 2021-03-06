import cv2 as cv
import math
import numpy as np
import pyzbar.pyzbar as pyzbar
import scipy.signal

import matplotlib as mpl
mpl.use('tkagg')  # Hack to make mpl work with Big Sur
import matplotlib.pyplot as plt


def find_qr(im, qr_margin=400, debug=False):
    """
    Looks for a QR code in bottom right then top left corner. If found
    in top left corner, the image is upside down.
    Currently doesn't return result of QR read, which could be of interest
    in the future.

    :param np.array im: Grayscale 2D image
    :param int qr_margin: Pixel margin of how large square to search for QR in
    :param bool debug: Display debug plot
    :return bool rotate_180:  Image is upside down
    """
    rotate_180 = False
    im_shape = im.shape
    im_qr = im[im_shape[0] - qr_margin:, im_shape[1] - qr_margin:]
    # Look for qr code
    decoded_objects = pyzbar.decode(im_qr)
    # Image could be upside down
    if len(decoded_objects) == 0:
        rotate_180 = True
        im_qr = np.rot90(im[:qr_margin, :qr_margin], 2)
        decoded_objects = pyzbar.decode(im_qr)

    # Could add search through list to find QR if list len > 1
    assert len(decoded_objects) == 1,\
        "Not one QR code found, but {}".format(len(decoded_objects))

    if debug:
        print("QR data: {}".format(decoded_objects[0].data))
        qr_rect = decoded_objects[0].rect
        im_roi = cv.rectangle(
            np.array(im_qr),
            (qr_rect.left, qr_rect.top),
            (qr_rect.left + qr_rect.width, qr_rect.top + qr_rect.height),
            (255, 0, 0),
            2,
        )
        plt.imshow(im_roi)
        plt.show()
    return rotate_180


def make_log_filter(sigma_gauss=5):
    """
    Creates a uniform 2D Laplacian of Gaussian filter with given sigma.

    :param int sigma_gauss: Standard deviation of Gaussian
    :return np.array log_filter: 2D LoG filter
    """
    n = np.ceil(sigma_gauss * 6)
    rows, cols = np.ogrid[-n // 2:n // 2 + 1, -n // 2:n // 2 + 1]
    sigma_sq = 2 * sigma_gauss ** 2
    row_filter = np.exp(-(rows ** 2 / sigma_sq))
    col_filter = np.exp(-(cols ** 2 / sigma_sq))
    log_filter = (-sigma_sq + cols ** 2 + rows ** 2) * \
                 (col_filter * row_filter) * \
                 (1 / (np.pi * sigma_sq * sigma_gauss ** 2))
    # Total filter should sum to 1 to not alter mean intensity
    log_filter = log_filter / sum(sum(log_filter))
    return log_filter


def filter_and_thresh(im_gray, log_filter):
    """
    Filter an image with a Laplacian of Gaussian then threshold based
    on intensity standard deviation.

    :param np.array im_gray: Grayscale 2D image (uint8)
    :param np.array log_filter: LoG filter
    :return np.array im_thresh: Filtered and thresholded image
    """
    max_intensity = im_gray.max()
    im_norm = (max_intensity - im_gray) / max_intensity
    # Filter with Laplacian of Gaussian
    im_norm = cv.filter2D(im_norm, -1, log_filter)
    # Threshold brighter areas
    im_thresh = (im_norm > 2 * im_norm.std()).astype(np.uint8)
    return im_thresh


def find_lines(im_thresh, strel_len=400, do_vert=True):
    """
    Find horizontal or vertical lines in thresholded image using
    a morphological opening operation with a structuring element that
    is either a long horizontal or vertical line.

    :param np.array im_thresh: Thresholded uint8 image
    :param int strel_len: Length of structuring element line
    :param bool do_vert: Use vertical line as structuring element
    :return np.array im: Image with horizontal or vertical lines
    """
    im_lines = im_thresh.copy()
    strel_shape = (strel_len, 1)
    if do_vert:
        strel_shape = (1, strel_len)
    line_struct = cv.getStructuringElement(cv.MORPH_RECT, strel_shape)
    return cv.morphologyEx(im_lines, cv.MORPH_OPEN, line_struct)


def get_angle_and_rotate(im, log_filter, debug=False):
    """
    Assuming that there are always some horizontal lines at the top of the
    ballot, this function finds Hough lines from a filtered and thresholded
    image and computes their angles.
    The ballot gets rotated by the median of the found line angles.

    :param np.array im: Color ballot image
    :param np.array log_filter: Laplacian of Gaussian filter
    :param bool debug: Determines if debug plot will be made
    :return np.array im_gray: Rotated ballot image
    """
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_thresh = filter_and_thresh(im_gray[25:125, :], log_filter)
    lines = cv.HoughLinesP(
        im_thresh,
        rho=1,
        theta=np.pi / 1440,
        threshold=80,
        minLineLength=500,
        maxLineGap=30,
    )
    im_output = cv.cvtColor(im_gray[25:125, :], cv.COLOR_GRAY2RGB)
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(im_output, (x1, y1), (x2, y2), (255, 0, 0), 3)
            angle = math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
            if abs(angle) < 3:
                angles.append(angle)
    if debug:
        plt.imshow(im_output); plt.show()
    if len(angles) == 0:
        angles = [0]
    angle = np.median(angles)
    print("median of {} lines: {:.3f}".format(len(angles), angle))
    if angle != .0:
        im_gray = rotate_image(im_gray, angle)
    return im_gray


def rotate_image(im, angle):
    """
    Create a 2D translation matrix from angle.
    Rotate around center of image.

    :param np.array im: Image
    :param float angle: Angle in degrees
    :return np.array im_rot: Rotated image
    """
    im_shape = im.shape
    a = np.cos(angle * np.pi / 180)
    b = np.sin(angle * np.pi / 180)
    rot_0 = (1 - a) * im_shape[1] / 2 - b * im_shape[0] / 2
    rot_1 = b * im_shape[1] / 2 + (1 - a) * im_shape[0] / 2
    t_matrix = np.array([[a, b, rot_0],
                        [-b, a, rot_1]])

    im_rot = cv.warpAffine(im, t_matrix,(im_shape[1], im_shape[0]))
    return im_rot


def get_vert_peaks(im_vert, margin=35, peak_min=.15, dist=25, debug=False):
    """
    From a 1D line profile from 2D image averaged over rows, find prominent
    peaks and assume those are the column start and stopping points in the image.

    :param np.array im_vert: Image with vertical lines
    :param int margin: Margin in pixel in which to ignore lines (boundary effects)
    :param float peak_min: Minimum peak height [0, 1]
    :param int dist: Minimum distance in pixels between peaks
    :param bool debug: Make debug plot
    :return np.array peak_idxs: Beginning and end locations in pixels of
        columns [nbr cols, 2]
    """
    profile_vert = np.mean(im_vert, 0)
    peak_idxs, peak_vals = scipy.signal.find_peaks(
        profile_vert,
        height=peak_min,
        distance=dist,
    )
    # Remove outliers
    if peak_idxs[0] < margin:
        peak_idxs = peak_idxs[1:]
    assert len(peak_idxs) > 0, "No vertical lines detected"
    if peak_idxs[-1] > len(profile_vert) - margin:
        peak_idxs = peak_idxs[:-1]

    if debug:
        plt.plot(np.arange(len(profile_vert)), profile_vert)
        for p in peak_idxs:
            plt.plot(p, profile_vert[p], 'g*', ms=8)
        plt.show()

    assert len(peak_idxs) in [2, 4, 6], \
        "wrong number of vertical lines detected: {}".format(len(peak_idxs))
    return np.reshape(peak_idxs, (-1, 2))


def get_hor_peaks(im_column,
                  margin=300,
                  height=.3,
                  dist=25,
                  writein_margin=125,
                  debug=False):
    """
    From a 1D profile of a 2D image averaged over columns, find prominent
    peaks and assume they are the horizontal start and stopping points
    of boxes in columns.

    :param np.array im_column: Image ROI with horizontal lines enhanced
    :param int margin: Margin in pixels to ignore at bottom of image
    :param int height: Minimum height of peaks as percentage of profile intensity max (intensity range [0, 1])
    :param int dist: Minimum distance in pixel between peaks
    :param int writein_margin: Distance in pixels between writein line and box bottom
    :param bool debug: Make debug plot
    :return np.array peak_idxs: Beginning and end locations in pixels of
        columns [nbr cols, 2]
    """
    profile_hor = np.mean(im_column, 1)
    peak_min = height * np.max(profile_hor)
    peak_idxs, peak_vals = scipy.signal.find_peaks(
        profile_hor,
        height=peak_min,
        distance=dist,
    )
    peak_vals = peak_vals['peak_heights']
    # Remove outlier at the top
    if peak_idxs[0] < 25:
        peak_idxs = peak_idxs[1:]
        peak_vals = peak_vals[1:]
    if len(peak_idxs) <= 1:
        print("Only found {} horizontal lines detected, proceeding "
              "with whole length of column".format(len(peak_idxs)))
        return np.array([[0, len(profile_hor)]])
    # Remove line at the bottom of ballot
    bottom_lines = np.where(len(profile_hor) - peak_idxs < margin)[0]
    peak_idxs = np.delete(peak_idxs, bottom_lines)
    peak_vals = np.delete(peak_vals, bottom_lines)
    peak_idxs0 = peak_idxs

    # Check peak amplitudes and remove lesser peaks (remove write-in lines)
    pos = 1
    while pos < len(peak_idxs) - 2:
        if (peak_idxs[pos + 1] - peak_idxs[pos] < writein_margin) and \
           (.9 * peak_vals[pos + 1] > peak_vals[pos]) and \
           (peak_idxs[pos + 2] - peak_idxs[pos + 1] < writein_margin):
            peak_idxs = np.delete(peak_idxs, pos)
            peak_vals = np.delete(peak_vals, pos)
        pos += 1
    # Check last line
    if len(peak_idxs) % 2 == 1:
        if (peak_idxs[-1] - peak_idxs[-2] < writein_margin) and \
           (.9 * peak_vals[-1] > peak_vals[-2]):
            peak_idxs = np.delete(peak_idxs, len(peak_idxs) - 2)
            peak_vals = np.delete(peak_vals, len(peak_idxs) - 2)

    # Remove write-in lines starting from bottom
    pos = len(peak_idxs) - 1
    while pos > 0:
        if len(peak_idxs) % 2 == 1:
            if (peak_idxs[pos] - peak_idxs[pos - 1] < writein_margin) and \
                    (peak_vals[pos] > peak_vals[pos - 1]):
                peak_idxs = np.delete(peak_idxs, pos - 1)
                peak_vals = np.delete(peak_vals, pos - 1)
        pos -= 2

    if debug:
        plt.plot(np.arange(len(profile_hor)), profile_hor)
        for p in peak_idxs0:
            plt.plot(p, profile_hor[p], 'r*', ms=8)
        for p in peak_idxs:
            plt.plot(p, profile_hor[p], 'g*', ms=8)
        plt.show()

    if len(peak_idxs) % 2 == 1:
        print("Wrong number of horizontal lines detected {}, proceeding "
              "with whole length of column".format(len(peak_idxs)))
        if len(peak_idxs) == 1:
            peak_idxs = np.array([[peak_idxs[0], im_shape[0] - margin]])
        else:
            peak_idxs = np.array([[peak_idxs[0], peak_idxs[-1]]])
    else:
        peak_idxs = np.reshape(peak_idxs, (-1, 2))
    return peak_idxs


def make_max_detector(min_thresh=75,
                      max_thresh=255,
                      min_area=25,
                      min_dist_between_blobs=75,
                      min_circularity=.75,
                      min_convexity=.75,
                      min_repeatability=2):
    """
    Initialize OpenCV's simple blob detector in order to find maxima
    from convolution with templates. These parameters could be tuned.

    :param int min_thresh: Minimum intensity threshold
    :param int max_thresh: Maximum intensity threshold
    :param int min_area: Minimum blob area
    :param int min_dist_between_blobs: Minimum distance between detected blobs
    :param float min_circularity: Minimum circularity of blobs
    :param floqt min_convexity: Minimum convexity of blobx
    :param int min_repeatability: Minimal number of times the same spot has
        to be detected at different thresholds
    :return detector: OpenCV simpleblobdetector instance
    """
    # Set spot detection parameters
    blob_params = cv.SimpleBlobDetector_Params()
    # Change thresholds
    blob_params.minThreshold = min_thresh
    blob_params.maxThreshold = max_thresh
    # Filter by Area
    blob_params.filterByArea = True
    blob_params.minArea = min_area
    # blob_params.maxArea = self.max_area
    # Filter by Circularity
    blob_params.filterByCircularity = True
    blob_params.minCircularity = min_circularity
    # Filter by Convexity
    blob_params.filterByConvexity = True
    blob_params.minConvexity = min_convexity
    blob_params.minDistBetweenBlobs = min_dist_between_blobs
    blob_params.minRepeatability = min_repeatability
    # This detects bright spots, which they are after top hat
    blob_params.blobColor = 255
    return cv.SimpleBlobDetector_create(blob_params)


def points2line(p1, p2):
    """
    Given two points on a line, compute coefficients A, B, C in
    Ax + By = C
    :param list p1: Point 1 on line
    :param list p2: Point 2 on line
    :return list A, B, C: Coefficients in linear system
    """
    A = (p1[1] - p2[1]).astype(np.float)
    B = (p2[0] - p1[0]).astype(np.float)
    C = (p1[0] * p2[1] - p2[0] * p1[1]).astype(np.float)
    return [A, B, -C]


def intersection(p_hor, p_vert, im_shape):
    """
    Find intersection point between two lines, e.g.
    # intersections = []
    # for line_hor in lines_hor:
    #     for line_vert in lines_vert:
    #         p_int = intersection(line_vert, line_hor, im_shape)
    #         if p_int is not None:
    #             intersections.append(p_int)
    #
    # for coord in intersections:
    #     cv.circle(im_lines, coord, 7, (0, 0, 255), -1)
    #
    # plt.imshow(im_lines); plt.show()
    """
    line_hor = points2line([p_hor[0][0], p_hor[0][1]], [p_hor[0][2], p_hor[0][3]])
    line_vert = points2line([p_vert[0][0], p_vert[0][1]], [p_vert[0][2], p_vert[0][3]])
    coord = None
    D = line_hor[0] * line_vert[1] - line_hor[1] * line_vert[0]
    Dx = line_hor[2] * line_vert[1] - line_hor[1] * line_vert[2]
    Dy = line_hor[0] * line_vert[2] - line_hor[2] * line_vert[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        # Check that intersection is within image
        if 0 < x < im_shape[1] and 0 < y < im_shape[0]:
            # Check that intersection is close to lines
            if abs(x - p_hor[0][0]) < 25 or abs(x - p_hor[0][0]):
                coord = (int(np.round(x)), int(np.round(y)))
    return coord


def conv_and_norm(im_markers, template):
    """
    Convolve image with a template and convert to uint8 format.

    :param np.array im_markers: Image
    :param np.array template: Template (should be much smaller than image)
    :return np.array im_conv: Convolved image
    """
    im_conv = cv.matchTemplate(im_markers, template, cv.TM_CCOEFF_NORMED)
    im_conv[im_conv < 0] = 0
    return (im_conv * 255).astype(np.uint8)
