#!/usr/bin/env python3
"""
This module produces a simulated cylinder
It is a rotational symmetric structure where the radius along the y-axis is simply a linear function
"""
import numpy as np

import fp23dpy
from .. import simulation

name = "cylinder"
max_rmse = 0.2  # expected rmse below this value for reconstruction
# errors at the edges contribute to this

# Parameters for cylinder
_cylinder_r = 250  # the cylinder radius


def radius(Y):
    if isinstance(Y, np.ndarray):
        return _cylinder_r * np.ones(Y.shape)
    else:
        return _cylinder_r


shape = (512, 512)


def get_scale(shape):
    return np.min(shape) / 512


def get_mid_y():
    return 20


def get_calibration(shape=shape):
    calibration = fp23dpy.Calibration()
    calibration["T"] = 12.0
    calibration["gamma"] = 0.0
    calibration["theta"] = 15 * np.pi / 180
    calibration["scale"] = get_scale(shape)

    mid_y, mid_x = (np.array(shape) / 2).astype(int)
    calibration["principal_point"] = [int(mid_x), int(mid_y)]

    absolute_threeD = simulation.get_rotsym_depth(0, -mid_y, radius).data[0]
    new_x, new_y, absolute_threeD = simulation.get_rotsym_projection_map(
        np.array([0, -mid_y, absolute_threeD]), calibration
    ).data

    calibration["absolute_threeD"] = [int(mid_x), int(mid_y), absolute_threeD]
    return calibration


def get_projected_coordinate_grid(shape=shape, calibration=None):
    scale = get_scale(shape)
    if calibration is None:
        calibration = get_calibration(shape)
    return simulation.get_rotsym_projection_map(
        radius, calibration, shape=shape, scale=scale
    )


def render(shape=shape, calibration=None):
    """Helper function to render the cylinder structure"""
    if calibration is None:
        calibration = get_calibration(shape)
    coordinate_grid = simulation.get_rotsym_coordinate_grid(
        radius, shape, scale=calibration["scale"], mid_y=get_mid_y()
    )
    return simulation.render(coordinate_grid, shape, calibration, True)
