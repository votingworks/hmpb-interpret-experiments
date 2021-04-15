import argparse
import cv2 as cv
import glob
import natsort
import numpy as np
import os
import pandas as pd
import pickle
import time

import ballot_analysis.database_ops as db_ops

THRESH = .12


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--database',
        type=str,
        required=True,
        help="Directory where ballot.db database file lives",
    )
    parser.add_argument(
        '-r', '--results',
        type=str,
        required=True,
        help="Path to directory containing result csvs",
    )
    return parser.parse_args()


def ballot_stats(args):

    db_dir = args.database
    result_dir = args.results

    file_names = natsort.natsorted(
        glob.glob(os.path.join(db_dir, '*original.png'),)
    )
    ellipse_mismatch = 0
    no_qr = 0
    no_markinfo = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for idx, file_name in enumerate(file_names):
        # Convert from results csv to png
        im_name = os.path.basename(file_name)
        csv_name = os.path.join(
            result_dir,
            "results_" + im_name[:-3] + 'csv',
        )
        if not os.path.isfile(csv_name):
            no_qr += 1
            print("No QR code found in {}, idx {}".format(im_name, idx))
            continue
        # Read Jenny's results
        res = pd.read_csv(csv_name)
        # Read Brian's results
        mark_info = db_ops.get_image_records(db_dir, im_name)
        if len(mark_info) == 0:
            no_markinfo += 1
            print("No markInfo in {}, idx {}".format(im_name, idx))
        elif len(mark_info) == res.shape[0]:
            for idx in range(len(mark_info)):
                mark_label = int(mark_info[idx]['score'] > THRESH)
                label = res.iloc[idx]['label']
                if mark_label == 0 and label == 0:
                    tn += 1
                if mark_label == 1 and label == 1:
                    tp += 1
                if mark_label == 0 and label == 1:
                    fp += 1
                    print("FP found in {}".format(im_name))
                if mark_label == 1 and label == 0:
                    fn += 1
                    print("FN found in {}".format(im_name))
        else:
            ellipse_mismatch += 1
            print("Wrong nbr of ellipses in {}, idx {}".format(im_name, idx))

    print("Nbr mismatching ellipses: {}, no markInfo: {}, "
          "no QR: {} of total: {}".format(
        ellipse_mismatch,
        no_markinfo,
        no_qr,
        len(file_names),
    ))
    print("Stats of successful analysis:")
    print("True positives: {}".format(tp))
    print("True negatives: {}".format(tn))
    print("False positives: {}".format(fp))
    print("False negatives: {}".format(fn))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("Accuracy: {:.2f}".format(accuracy))
    precision = tp / (tp + fp)
    print("Precision: {:.2f}".format(precision))
    recall = tp / (tp + fn)
    print("Recall: {:.2f}".format(recall))
    f1 = 2 * precision * recall / (precision + recall)
    print("F1 score: {:.2f}".format(f1))


if __name__ == '__main__':
    args = parse_args()
    ballot_stats(args)