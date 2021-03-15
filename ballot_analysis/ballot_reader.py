import argparse
import cv2 as cv
import glob
import natsort
import numpy as np
import os
import time

import matplotlib as mpl
mpl.use('tkagg')  # Hack to make mpl work with Big Sur
import matplotlib.pyplot as plt

import im_processing as im_proc
import plot_utils as plt_utils


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

    # Instantiate spot detectors and create filter
    max_detector = im_proc.make_max_detector()
    max_detector_filled = im_proc.make_max_detector(min_area=250)
    log_filter = im_proc.make_log_filter(sigma_gauss=5)

    file_names = natsort.natsorted(
        glob.glob(os.path.join(input_dir, '*original.png'),)
    )
    if file_idx is not None:
        assert 0 <= file_idx < len(file_names),\
            "File idx {} doesn't work".format(file_idx)
        file_names = [file_names[file_idx]]

    for idx, file_name in enumerate(file_names):
        start_time = time.time()
        im_name = os.path.basename(file_name)[:-4]
        print("Analyzing idx: {}, name: {}".format(idx, im_name))
        im = cv.imread(file_name)
        # Check if images are upside down by finding QR code
        try:
            rotate_180 = im_proc.find_qr(im=im, debug=debug)
        except AssertionError:
            print("Can't find QR for {}".format(file_name))
            continue
        if rotate_180:
            im = np.rot90(im, 2)
        # Rotate image based on angle of bottom line in ballot
        im_gray = im_proc.get_angle_and_rotate(
            im,
            log_filter,
            debug,
        )
        if debug:
            plt.imshow(im_gray, cmap='gray'); plt.show()
        # Filter and threshold rotated image
        im_thresh = im_proc.filter_and_thresh(im_gray, log_filter)

        # Find horizontal and vertical lines
        im_hor = im_proc.find_lines(im_thresh, strel_len=200, do_vert=False)
        im_vert = im_proc.find_lines(im_thresh, strel_len=200, do_vert=True)

        # For plotting only
        im_boxes = np.array(cv.cvtColor(im_gray, cv.COLOR_GRAY2RGB))
        # Plot detected horizontal and vertical lines
        plt_utils.plot_hough_lines(im_hor, im_boxes, (255, 0, 0))
        plt_utils.plot_hough_lines(im_vert, im_boxes, (0, 255, 0))

        # Compress horizontal and vertical images to 1D profiles and
        # detect boxes as maxima
        try:
            peak_idxs = im_proc.get_vert_peaks(im_vert, debug=debug)
        except AssertionError as e:
            print("Failed to detect correct vertical lines", e)
            # Write profile where error occurred
            plt_utils.write_debug_profile(im_vert, output_dir, im_name)
            continue

        # Pair boxes using vertical lines
        for col in range(peak_idxs.shape[0]):
            # Get horizontal lines profile for column
            coords_col = peak_idxs[col, :]
            im_column = im_hor[:, coords_col[0]: coords_col[1]]
            hor_peak_idxs = im_proc.get_hor_peaks(im_column, debug=debug)

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
                im_conv = im_proc.conv_and_norm(im_markers, template)
                keypoints = max_detector.detect(im_conv)
                plt_utils.plot_boxes(
                    im_boxes,
                    keypoints,
                    coords_col,
                    coords_row,
                    template_shape,
                    (255, 255, 0),
                )
                # Search for filled in boxes with filled elliptical template
                im_conv_filled = im_proc.conv_and_norm(im_markers, template_filled)
                keypoints = max_detector_filled.detect(im_conv_filled)
                plt_utils.plot_boxes(
                    im_boxes,
                    keypoints,
                    coords_col,
                    coords_row,
                    template_shape,
                    (255, 0, 255),
                )
                # TODO: store box coordinates and filled/empty values

        print("Processing time: {:.3f} s".format(time.time() - start_time))
        # Write debug image
        cv.imwrite(os.path.join(output_dir, im_name + "_debug.png"), im_boxes)
        if debug:
            plt.imshow(im_boxes); plt.show()


if __name__ == '__main__':
    args = parse_args()
    ballot_analyzer(args)


