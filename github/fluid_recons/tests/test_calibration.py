import numpy as np
import os
import os.path as osp
import json

import fp23dpy
import fp23dpy.helpers

def dict_equality(expected, actual):
    for key, value in expected.items():
        if not key in actual:
            return False
        elif actual[key] != value:
            return False
    return True

def test_calibration_read_write():
    test_file = "test.json"
    raw_calibration = {"T": 20, "gamma": 43 }
    # converting from radians to degrees when reading and writing
    calibration = fp23dpy.Calibration(raw_calibration)
    calibration["gamma"] *= np.pi / 180
    try:
        calibration.write(test_file)
        with open(test_file, "r") as f:
            written_calibration = json.load(f)
        assert dict_equality(raw_calibration, written_calibration)

        read_calibration = fp23dpy.Calibration.read(test_file)
        assert dict_equality(calibration, read_calibration)

    finally:
        if osp.isfile(test_file):
            os.remove(test_file)
