import argparse
import cv2 as cv
import glob
import natsort
import numpy as np
import os
import pandas as pd
import pickle
import time

import matplotlib as mpl
mpl.use('tkagg')  # Hack to make mpl work with Big Sur
import matplotlib.pyplot as plt

import ballot_analysis.im_processing as im_proc
import ballot_analysis.plot_utils as plt_utils

DF_NAMES = ['col', 'row', 'ellipse_nbr', 'prob', 'label']


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

    # Load logistic regression model
    lr_model = pickle.load(open('ballot_analysis/lr_model.sav', 'rb'))
    # Create LoG filter
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
        results_df = pd.DataFrame(columns=DF_NAMES)
        print("Analyzing idx: {}, name: {}".format(idx, im_name))
        im = cv.imread(file_name)
        # Check if images are upside down by finding QR code
        try:
            im = im_proc.find_qr(im=im, debug=debug)
        except AssertionError:
            print("Can't find QR for {}".format(file_name))
            continue
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

        # Create 1D profiles of lines and detect boxes as maxima
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
                # Extract ellipses and classify them
                feat_mat, ellipses, row_coord, col_coord, width, height = \
                    im_proc.get_ellipses(
                    im_markers,
                    (coords_col[0], coords_row[0]),
                )
                # Do binary classification for now
                if feat_mat.shape[0] == 0:
                    print("No ellipses detected")
                    continue
                labels = lr_model.predict(feat_mat)
                probs = lr_model.predict_proba(feat_mat)
                probs = probs[:, 1]
                nbr_ellipses = labels.shape[0]
                results = pd.DataFrame({
                    'ellipse_nbr': np.arange(nbr_ellipses, dtype=np.int).tolist(),
                    'row': (row * np.ones(nbr_ellipses, dtype=np.int)).tolist(),
                    'col': (col * np.ones(nbr_ellipses, dtype=np.int)).tolist(),
                    'prob': probs.tolist(),
                    'label': labels.astype(np.int).tolist(),
                    'row_coord': row_coord,
                    'col_coord': col_coord,
                    'width': width,
                    'height': height,
                })
                results_df = results_df.append(
                    results,
                    ignore_index=True,
                )
                # Plot results
                plt_utils.plot_ellipses(im_boxes, ellipses, labels)

        print("Processing time: {:.3f} s".format(time.time() - start_time))
        # Write results
        results_df.to_csv(os.path.join(output_dir, "results_" + im_name + ".csv"))
        # Write debug image
        cv.imwrite(os.path.join(output_dir, im_name + "_debug.png"), im_boxes)
        if debug:
            plt.imshow(im_boxes); plt.show()


if __name__ == '__main__':
    args = parse_args()
    ballot_analyzer(args)


