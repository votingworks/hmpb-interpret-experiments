import argparse
import cv2 as cv
import glob
import natsort
import numpy as np
import os
import pyzbar.pyzbar as pyzbar
import scipy.signal
import time

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


def find_lines(im_thresh, strel_len=400, do_vert=True):
    """
    Find horizontal or vertical lines in thresholded image using
    a morphological opening operation with a structuring element that
    is either a long horizontal or vertical line.

    :param np.array im_thresh: Thresholded uint8 image
    :param int strel_len: Length of structuring element line
    :param bool do_vert: Use vertical line as structuring element
    """
    im_lines = im_thresh.copy()
    strel_shape = (strel_len, 1)
    if do_vert:
        strel_shape = (1, strel_len)
    line_struct = cv.getStructuringElement(cv.MORPH_RECT, strel_shape)
    return cv.morphologyEx(im_lines, cv.MORPH_OPEN, line_struct)


def get_vert_peaks(im_vert, margin=25, peak_min=.15, dist=25, debug=False):
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
    peak_idxs, peak_vals = scipy.signal.find_peaks(profile_vert, height=peak_min, distance=dist)
    assert len(peak_idxs) > 0, "No vertical lines detected"

    # Remove outliers
    if peak_idxs[0] < margin:
        peak_idxs = peak_idxs[1:]
    assert len(peak_idxs) > 0, "No vertical lines detected"
    if peak_idxs[-1] > len(profile_vert) - margin:
        peak_idxs = peak_idxs[:-1]

    if debug:
        plt.plot(np.arange(len(profile_vert)), profile_vert)
        for p in peak_idxs:
            plt.plot(p, profile_vert[p], 'r*', ms=8)
        plt.show()

    assert len(peak_idxs) in set([2, 4, 6]), \
        "wrong number of vertical lines detected: {}".format(len(peak_idxs))
    return np.reshape(peak_idxs, (-1, 2))


def get_hor_peaks(im_column, margin=300, height=.3, dist=25, debug=False):
    """
    From a 1D profile of a 2D image averaged over columns, find prominent
    peaks and assume they are the horizontal start and stopping points
    of boxes in columns.

    :param np.array im_column: Image ROI with horizontal lines enhanced
    :param int margin: Margin in pixels to ignore at bottom of image
    :param int height: Minimum height of peaks as percentage of profile intensity max (intensity range [0, 1])
    :param int dist: Minimum distance in pixel between peaks
    :param bool debug: Make debug plot
    :return np.array peak_idxs: Beginning and end locations in pixels of
        columns [nbr cols, 2]
    """
    profile_hor = np.mean(im_column, 1)
    peak_min = height * np.max(profile_hor)
    peak_idxs, peak_vals = scipy.signal.find_peaks(profile_hor, height=peak_min, distance=dist)
    peak_vals = peak_vals['peak_heights']
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
    pos = 0
    while pos < len(peak_idxs) - 2:
        if (peak_idxs[pos + 2] - peak_idxs[pos + 1] < 125) and \
                (.85 * peak_vals[pos + 2] > peak_vals[pos + 1]):
            peak_idxs = np.delete(peak_idxs, pos + 1)
            peak_vals = np.delete(peak_vals, pos + 1)
        pos += 1

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


def plot_hough_lines(im, im_output, plot_color):
    """
    Finds Hough lines and overlay them on debug plot

    :param np.array im: Binary image
    :param np.array im_output: Image to add overlays to
    :param tuple plot_color: RGB values for plot overlays
    """
    lines_hor = cv.HoughLinesP(im, 1, np.pi/180, 10, minLineLength=500, maxLineGap=30)
    for line in lines_hor:
        for x1, y1, x2, y2 in line:
            cv.line(im_output, (x1, y1), (x2, y2), plot_color, 3)


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


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="Input directory path",
    )
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        help="Create debug plots. Default: False",
    )
    parser.set_defaults(debug=False)
    parser.add_argument(
        '--file_idx',
        type=int,
        default=None,
        help="Run only file with this index in a sorted list of files in dir"
    )
    return parser.parse_args()


def ballot_analyzer(args):
    """
    Detect columns and checked boxes in ballots by finding vertical and
    horizontal lines in an image that has been filtered with a Laplacian
    of Gaussian then thresholded.
    Only prototyping at this point, no results are saved except debug plots.
    Assumes that ballots can be upside down but that smaller rotations
    are negligible.
    It finds empty and filled ellipses using an empty and filled template,
    so it will fail to detect boxes that have been filled with 'x'es or
    checkmarks.

    :param args: Command line arguments
        input: Full path to directory containing scanned ballots
        debug: Create debug plots (only recommended if running one/few images)
        file_idx: If None, run all files, otherwise run only file with given idx
    """
    input_dir = args.input
    debug = args.debug
    file_idx = args.file_idx
    # Create subdirectory where output is written
    output_dir = os.path.join(input_dir, 'analysis_debug')
    os.makedirs(output_dir, exist_ok=True)

    # Read template
    template = cv.imread('marker_template.png', cv.IMREAD_GRAYSCALE)
    template_filled = cv.imread('filled_template.png', cv.IMREAD_GRAYSCALE)
    template_shape = template.shape

    max_detector = make_max_detector()
    max_detector_filled = make_max_detector(min_area=250)

    file_names = natsort.natsorted(
        glob.glob(os.path.join(input_dir, '*original.png'),)
    )
    if file_idx is not None:
        assert 0 <= file_idx < len(file_names),\
            "File idx {} doesn't work".format(file_idx)
        file_names = [file_names[file_idx]]

    for file_name in file_names:
        start_time = time.time()
        im_name = os.path.basename(file_name)[:-4]
        print("Analyzing: {}".format(im_name))
        im = cv.imread(file_name)
        # Check if images are upside down by finding QR code
        try:
            rotate_180 = find_qr(im=im, debug=debug)
        except AssertionError:
            print("Can't find QR for {}".format(file_name))
            continue
        if rotate_180:
            im = np.rot90(im, 2)
        # Convert to grayscale
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        max_intensity = im_gray.max()
        im_norm = (max_intensity - im_gray) / max_intensity
        # Filter with Laplacian of Gaussian
        log_filter = make_log_filter()
        im_norm = cv.filter2D(im_norm, -1, log_filter)
        # Threshold brighter areas
        im_thresh = (im_norm > 2 * im_norm.std()).astype(np.uint8)
        # Find horizontal and vertical lines
        im_hor = find_lines(im_thresh, strel_len=300, do_vert=False)
        im_vert = find_lines(im_thresh, strel_len=300, do_vert=True)
        if debug:
            plt.imshow(im_hor, cmap='gray'); plt.show()
            plt.imshow(im_vert, cmap='gray'); plt.show()

        # For plotting only
        im_boxes = np.array(im.copy())
        # Plot detected horizontal and vertical lines
        plot_hough_lines(im_hor, im_boxes, (255, 0, 0))
        plot_hough_lines(im_vert, im_boxes, (0, 255, 0))

        # Compress horizontal and vertical images to 1D profiles and
        # detect boxes as maxima
        try:
            peak_idxs = get_vert_peaks(im_vert, debug=debug)
        except AssertionError as e:
            print("Failed to detect correct vertical lines", e)
            # Write profile where error occurred
            profile_vert = np.mean(im_vert, 0)
            plt.plot(np.arange(len(profile_vert)), profile_vert)
            fig_save = plt.gcf()
            debug_name = os.path.join(output_dir, im_name + "_vert_profile.png")
            fig_save.savefig(debug_name, bbox_inches='tight')
            plt.close(fig_save)
            continue

        # Pair boxes using vertical lines
        for col in range(peak_idxs.shape[0]):
            # Get horizontal lines profile for column
            coords_col = peak_idxs[col, :]
            im_column = im_hor[:, coords_col[0]: coords_col[1]]
            hor_peak_idxs = get_hor_peaks(im_column, debug=debug)

            for row in range(hor_peak_idxs.shape[0]):
                coords_row = hor_peak_idxs[row, :]
                # Get left 20% of box (that's where the fill-in boxes are)
                col_width = coords_col[0] + int((coords_col[1] - coords_col[0]) / 5)
                im_markers = im_gray[coords_row[0]:coords_row[1], coords_col[0]:col_width]
                marker_shape = im_markers.shape
                if marker_shape[0] <= template_shape[0] or marker_shape[1] <= template_shape[1]:
                    print("Image ROI too small for marker detection")
                    continue
                # Search for empty boxes with empty ellipse template
                im_conv = conv_and_norm(im_markers, template)
                keypoints = max_detector.detect(im_conv)
                plot_boxes(im_boxes, keypoints, coords_col, coords_row, template_shape, (255, 255, 0))
                # Search for filled in boxes with filled elliptical template
                im_conv_filled = conv_and_norm(im_markers, template_filled)
                keypoints = max_detector_filled.detect(im_conv_filled)
                plot_boxes(im_boxes, keypoints, coords_col, coords_row, template_shape, (255, 0, 255))
                # TODO: store box coordinates and filled/empty values

        print("Processing time: {:.3f} s".format(time.time() - start_time))
        # Write debug image
        cv.imwrite(os.path.join(output_dir, im_name + "_debug.png"), im_boxes)
        if debug:
            plt.imshow(im_boxes); plt.show()


if __name__ == '__main__':
    args = parse_args()
    ballot_analyzer(args)


