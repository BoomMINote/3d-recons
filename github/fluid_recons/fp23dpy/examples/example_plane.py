#!/usr/bin/env python3
"""
This module produces a simulated plane
It is a rotational symmetric structure where the radius along the y-axis is simply a linear function
"""
import numpy as np

import fp23dpy
from .. import simulation

name = "plane"
max_rmse = 0.001  # expected rmse below this value for reconstruction

shape = (512, 512)

plane_z = 1


def get_scale(shape):
    return np.min(shape) / 512


def get_calibration(shape=shape):
    calibration = fp23dpy.Calibration()
    calibration["T"] = 12.0
    calibration["gamma"] = 0.0
    calibration["theta"] = 15 * np.pi / 180
    calibration["scale"] = get_scale(shape)

    mid_y, mid_x = np.array(shape) / 2
    calibration["principal_point"] = [int(mid_x), int(mid_y)]

    calibration["absolute_threeD"] = [int(mid_x), int(mid_y), plane_z]
    return calibration


def get_projected_coordinate_grid(shape=shape, calibration=None):
    scale = get_scale(shape)
    if calibration is None:
        calibration = get_calibration(shape)
    mid_x, mid_y = calibration["principal_point"]

    Y, X = np.mgrid[: shape[0], : shape[1]]
    X = (X - mid_x) / scale
    Y = -(Y - mid_y) / scale
    depth = np.ones(shape) * plane_z

    X /= np.cos(calibration["theta"])
    grid = np.stack((X, Y, depth))
    return grid


def render(shape=shape, calibration=None):
    """Helper function to render the plane structure"""
    if calibration is None:
        calibration = get_calibration(shape)
    dmap = np.ones(shape) * plane_z
    return simulation.render_from_map(dmap, calibration)
