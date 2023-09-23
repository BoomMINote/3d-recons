import numpy as np

# import matplotlib.pyplot as plt

from fp23dpy import fp23d
from fp23dpy.examples import (
    example_plane,
    example_cylinder,
    example_cone,
    example_drop,
    example_cone_bump,
)


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def check_example_structure(module, with_mask=True, with_square_carrier=False):
    calibration = module.get_calibration()
    grid = module.get_projected_coordinate_grid()
    signal = 255 / 2 * module.render()
    signal.data[signal.mask] = 0

    if not with_mask:
        signal = signal.data
    if with_square_carrier:
        calibration["T"] = [0, 2 * np.pi / calibration["T"], 0, 0, 0]

    reconstructed_grid = fp23d(signal, calibration)
    # for i in range(3):
    #     plt.figure()
    #     plt.imshow(reconstructed_grid[i] - grid[i])
    #     plt.colorbar()
    # plt.show()
    rmse = rms(reconstructed_grid - grid) / (np.max(grid) - np.min(grid))
    print("rmse {}, {}".format(module.name, rmse))
    assert rmse < module.max_rmse, "Expected rmse below {}, got {}".format(
        module.max_rmse, rmse
    )


def test_fp23d_plane():
    check_example_structure(example_plane, with_mask=False)


def test_fp23d_square_carrier():
    check_example_structure(example_plane, with_mask=False, with_square_carrier=True)


def test_fp23d_cylinder():
    check_example_structure(example_cylinder)


def test_fp23d_cone():
    check_example_structure(example_cone)


def test_fp23d_cone_bump():
    check_example_structure(example_cone_bump)


def test_fp23d_drop():
    check_example_structure(example_drop)
