#!/usr/bin/env python3
"""
This module produces a simulated drop used in the article referenced in the README file
It is a rotational symmetric structure where the radius along the y-axis is a combination of a third degree polynomial and a sphere as seen in function radius.
"""
import numpy as np

import fp23dpy
from .. import simulation

name = "drop"
max_rmse = 0.004  # expected rmse below this value for reconstruction

shape = (512, 512)

# Parameters for drop
_drop_r = 1  # radius of drop
_s = 0.8  # scale of minima width
_x1 = -1  # derivative of the cube poly in zero
_xmin = -1.6  # the minimal x value

_a3 = 2 * _drop_r * (_s - 1) / _x1 ** 3
_b3 = 3 / 2 * _x1 * _a3
_c3 = 0
_d3 = _drop_r
# print(_a3, _b3, _c3, _d3)


def _radius_poly(x):
    return _a3 * x ** 3 + _b3 * x ** 2 + _c3 * x + _d3


def _radius_circle(x):
    return _drop_r * np.sin(np.arccos(x / _drop_r))


def radius(y):
    """Radius of drop at different Y coordinates"""
    isscalar = np.isscalar(y)
    if isscalar:
        y = np.array([y])
    res = np.zeros(y.shape)
    start_ind = y >= 0
    res[start_ind] = _radius_poly(y[start_ind])
    mid_ind = (y < 0) & (np.abs(y) < _drop_r)
    res[mid_ind] = _radius_circle(y[mid_ind])
    if isscalar:
        res = res[0]
    return res


def get_scale(shape):
    # return min((shape[0] - 10) / (_drop_r - _xmin), (shape[1] - 10) / 2 / max(radius(_xmin), radius(_x1), radius(0)))
    return (shape[0] - 10) / (_drop_r - _xmin)


def get_mid_y(scale):
    return -_xmin * scale


def get_calibration(shape=shape):
    calibration = fp23dpy.Calibration()
    calibration["T"] = 12.0
    calibration["gamma"] = 0.0
    calibration["theta"] = 15 * np.pi / 180
    scale = get_scale(shape)
    calibration["scale"] = scale

    mid_y = get_mid_y(scale)
    _, mid_x = [int(shape[0] / 2), int(shape[1] / 2)]
    calibration["principal_point"] = [mid_x, mid_y]

    absolute_threeD = simulation.get_rotsym_depth(0, 0, radius).data[0]
    new_x, new_y, absolute_threeD = simulation.get_rotsym_projection_map(
        np.array([0, 0, absolute_threeD]), calibration
    ).data

    calibration["absolute_threeD"] = [int(mid_x), int(mid_y), absolute_threeD]
    return calibration


def get_projected_coordinate_grid(shape=shape, calibration=None):
    scale = get_scale(shape)
    if calibration is None:
        calibration = get_calibration(shape)
    return simulation.get_rotsym_projection_map(
        radius, calibration, shape=shape, scale=scale, mid_y=get_mid_y(scale)
    )


def render(shape=shape, calibration=None):
    """Helper function to render the drop structure"""
    if calibration is None:
        calibration = get_calibration(shape)
    scale = get_scale(shape)
    coordinate_grid = simulation.get_rotsym_coordinate_grid(radius, shape, scale=scale, mid_y=get_mid_y(scale))
    return simulation.render(coordinate_grid, shape, calibration, True)
