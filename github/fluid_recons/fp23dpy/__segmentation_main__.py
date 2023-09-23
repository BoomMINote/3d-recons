""" Module to help with automatic segmentation and improvement of start segmentation """
import argparse
import numpy as np
import os.path as osp
from skimage import io
from tqdm import tqdm

from . import helpers
from .calibration import Calibration
from .segmentation import simple_segmentation, simple_segmentation_improvement

segmented_prefix = "segmented_"
reconstructed_prefix = "reconstructed_"


def segmentation_main():
    parser = argparse.ArgumentParser(
        description="3D reconstruct images with Fringe Patterns, segmenting the input images."
    )
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help='Input image files to segment, files with prefix "reconstructed" and "segmented" will not be considered',
    )
    parser.add_argument(
        "--T-lims",
        type=float,
        nargs=2,
        help="Limits in fringe period resolution to use for estimating which pixels are bad.",
    )
    parser.add_argument(
        "--min-island-size", type=int, default=50, help="Smallest island accepted"
    )
    parser.add_argument(
        "--min-hole-size", type=int, default=5, help="Smallest hole accepted"
    )
    parser.add_argument(
        "--min-ridge-width", type=int, default=5, help="Smallest hole accepted"
    )
    args = parser.parse_args()

    to_segment = [
        f
        for f in args.files
        if helpers.is_image_file(f)
        and not f.startswith(segmented_prefix)
        and not f.startswith(reconstructed_prefix)
    ]

    for f in tqdm(to_segment):
        signal = io.imread(f, as_gray=True)
        calibration = helpers.get_calibration(f)
        if calibration is None:
            print(
                "Warning: No calibration file found, using automatic calibration algorithm"
            )
            calibration = Calibration.calibrate(signal)
            print(calibration)

        initial_segmentation = helpers.get_segmentation(f)

        if initial_segmentation is None:
            # If no found segmentation do a segmentation on the estimated amplitude of the signal
            initial_segmentation = simple_segmentation(signal, calibration)
        else:
            initial_segmentation == initial_segmentation == 255
        segmentation = simple_segmentation_improvement(
            signal,
            calibration,
            initial_segmentation,
            args.T_lims,
            args.min_island_size,
            args.min_hole_size,
            args.min_ridge_width,
        )

        d, f = osp.split(f)
        segmentation_file = osp.join(d, "segmented_{}".format(f))
        io.imsave(
            segmentation_file, segmentation.astype(np.uint8) * 255, check_contrast=False
        )
