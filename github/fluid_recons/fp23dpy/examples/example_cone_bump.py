#!/usr/bin/env python3
"""
This module produces a simulated cone with a bump
It is a rotational symmetric structure where the radius along the y-axis is
a linear function where a gaussian has been added to the top of it
"""
import numpy as np
from scipy import interpolate

import fp23dpy
from .. import simulation

name = "cone_bump"
max_rmse = 0.0021  # expected rmse below this value for reconstruction

# Parameters for cone
radius0 = 20  # the minimum radius
# _cone_ra = -(512 / 2 - radius0) / 512  # the coefficient for how the radius increase
default_cone_angle = 50 * np.pi / 180

default_bump = {
    "height": 15,
    "width": 50,
    "center_y": 150,
}


def get_rotation_matrix(cone_angle):
    rotation_matrix = np.array(
        [
            [np.cos(cone_angle), -np.sin(cone_angle)],
            [np.sin(cone_angle), np.cos(cone_angle)],
        ]
    )
    return rotation_matrix


def radius(
    Y,
    height=default_bump["height"],
    width=default_bump["width"],
    gaussian_center_y=default_bump["center_y"],
    cone_angle=default_cone_angle,
):
    proj_fraction = 1 / np.cos(cone_angle)
    gaussian_center_y_prime = -gaussian_center_y * proj_fraction
    sigma = width / 2 / np.sqrt(2 * np.log(2))  # from fwhm to sigma

    if isinstance(Y, np.ndarray):
        n_y = len(Y)
    else:
        n_y = 10
    y = np.linspace(np.max(Y) * proj_fraction + 2, np.min(Y) * proj_fraction - 2, n_y)
    if sigma > 0:
        vertical_gaussian_bump_radius = height * np.exp(
            -((y - gaussian_center_y_prime) ** 2) / 2 / sigma**2
        )
    else:
        vertical_gaussian_bump_radius = np.zeros(len(y))
    rotation_matrix = get_rotation_matrix(cone_angle)
    rot_r = (
        rotation_matrix[0, 0] * vertical_gaussian_bump_radius
        + rotation_matrix[0, 1] * y
    )
    rot_y = (
        rotation_matrix[1, 0] * vertical_gaussian_bump_radius
        + rotation_matrix[1, 1] * y
    )
    if not np.all(np.diff(rot_y) < 0):
        raise ValueError("Unresolvable gaussian bump")
    interpolator = interpolate.interp1d(rot_y, rot_r, fill_value="extrapolate")
    cone_gaussian_radius = interpolator(Y) + radius0

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(vertical_gaussian_bump_radius, y)
    # plt.plot(rot_r, rot_y)
    # plt.plot(cone_gaussian_radius[:, 0], y)
    # plt.axis("equal")
    # plt.xlim(left=0)
    # plt.show()
    # exit()
    return cone_gaussian_radius


shape = (512, 512)


def get_scale(shape):
    return np.min(shape) / 512


def get_mid_y():
    return 0


def get_calibration(shape=shape, with_absolute=True):
    calibration = fp23dpy.Calibration()
    calibration["T"] = 12.0
    calibration["gamma"] = 0.0
    calibration["theta"] = 15 * np.pi / 180
    calibration["scale"] = get_scale(shape)

    mid_x = int(shape[1] / 2)
    calibration["principal_point"] = [mid_x, get_mid_y()]

    if with_absolute:
        threed_y = 20
        absolute_threeD = radius(-threed_y)
        new_x, new_y, absolute_threeD = simulation.get_rotsym_projection_map(
            np.array([0, -threed_y, absolute_threeD]), calibration
        ).data
        calibration["absolute_threeD"] = [int(mid_x), threed_y, absolute_threeD]
    return calibration


def get_projected_coordinate_grid(shape=shape, calibration=None):
    if calibration is None:
        calibration = get_calibration(shape, with_absolute=False)
    return simulation.get_rotsym_projection_map(
        radius,
        calibration,
        shape=shape,
        scale=calibration["scale"],
        mid_y=get_mid_y(),  # not the same as mid_y above
    )


def render(shape=shape, calibration=None):
    """Helper function to render the cone structure"""
    if calibration is None:
        calibration = get_calibration(shape)
    coordinate_grid = simulation.get_rotsym_coordinate_grid(
        radius, shape, scale=calibration["scale"], mid_y=get_mid_y()
    )
    return simulation.render(coordinate_grid, shape, calibration, use_rotsym=True)
