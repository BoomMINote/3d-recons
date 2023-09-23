"""
Module for 3D reconstructio of a fringe pattern image (the one you get from an FP-LIF setup)

The main method is fp23d which takes an image and an optional calibration as input and reconstructs 3D from it
"""
# import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from . import helpers as h
from . import Roi
from . import demodulation as dd


def phase_to_threeD_const(T, theta, s=1):
    """
    Calculation of the proportionality constant between the threeD
    coordinates and the phase of a fringe pattern, assumes ortographic camera.

    :param T: Fringe pattern carrier period on a flat surface as seen with a camera with zero angle
    :param theta: Angle between camera direction to the fringe pattern projection direction in radians.
    :param s: The focal of the telecentric lens which corresponds to the length pixels/unit in the real world.
    :returns: Proportionality constant
    """
    return T / 2 / np.pi / np.sin(theta) / s


def threeD_to_phase_const(T, theta, s=1):
    """Simply the inverse of phase_to_threeD_const"""
    return 1 / phase_to_threeD_const(T, theta, s)


def get_scales(calibration, shape=None):
    if "scale" in calibration:
        scale = calibration["scale"]
    else:
        if shape is None:
            raise ValueError("Shape must be given if scale is not found in calibration")
        # if no scale, make the output have a maximum size of 5
        scale = np.max(shape) / 5

    xscale = yscale = dscale = 1.0 / scale
    if "theta" in calibration:
        xscale *= 1 / np.cos(calibration["theta"])
        dscale *= phase_to_threeD_const(calibration["T"], calibration["theta"])
    if "phi" in calibration:  ## Phi stuff is not really tested
        dscale *= 1 / np.cos(np.abs(calibration["phi"] - calibration["gamma"]))
    return xscale, yscale, dscale


# _min_percentile = 0.005
def fp23d(signal: np.ndarray, calibration: dict):
    """Function for 3D reconstruction of a fringe pattern, if no calibration has been performed you can call the automated calibration fuction `fp23dpy.Calibration.calibrate` and use that as input calibration"""
    isMasked = np.ma.isMaskedArray(signal)
    signal = signal.astype(float)

    calibration = calibration.copy()
    # might have a square carrier in the T
    full_T = calibration["T"]
    calibration["T"] = h.get_T_from_square(calibration["T"], signal.shape)

    principal_point = None
    if isMasked:
        if isinstance(signal.mask, np.bool_):
            signal.mask = np.ones(signal.shape, dtype=bool) * signal.mask
        roi = Roi.find_from_mask(signal.mask)
        signal = roi.apply(signal)
        mask = signal.mask
        if "principal_point" in calibration:
            principal_point = roi.apply_to_points([calibration["principal_point"]])[0]

    else:
        mask = np.zeros(signal.shape, dtype=bool)
        roi = None
        if "principal_point" in calibration:
            principal_point = calibration["principal_point"]
    if principal_point is None:
        # setting as center of the image
        principal_point = np.array(signal.shape[::-1]) / 2

    shape = signal.shape

    # main estimation of phase here, important!
    phase_with_carrier = dd.demodulate(signal, calibration)
    carrier = h.make_carrier(
        signal.shape[-2:], full_T, calibration["gamma"], principal_point, roi
    )
    phase = phase_with_carrier - carrier

    labels, n_labels = measure.label(
        (~mask).astype(int), return_num=True, connectivity=1
    )
    absolute_labels = []
    if "absolute_phase" in calibration:
        # Set phases for areas to relative where they are known
        absolute_phases = np.array(calibration["absolute_phase"])
        if len(absolute_phases.shape) == 1:
            absolute_phases = np.expand_dims(absolute_phases, 0)
        if isMasked:
            absolute_phases[:, :2] = roi.apply_to_points(absolute_phases[:, :2])

        for x_a, y_a, absolute_phase in absolute_phases:
            x_a = int(x_a)
            y_a = int(y_a)
            if mask[y_a, x_a]:
                # value is masked in this array
                continue
            absolute_labels.append(labels[y_a, x_a])
            absolute_area = labels == absolute_labels[-1]
            phase[absolute_area] += absolute_phase - carrier[y_a, x_a] - phase[y_a, x_a]

    xscale, yscale, dscale = get_scales(calibration, signal.shape)
    threeD = phase * dscale

    if "absolute_threeD" in calibration:
        # Same as absolute_phase but for absolute_threeD points
        absolute_threeDs = np.array(calibration["absolute_threeD"])
        if len(absolute_threeDs.shape) == 1:
            absolute_threeDs = np.expand_dims(absolute_threeDs, 0)
        if isMasked:
            absolute_threeDs[:, :2] = roi.apply_to_points(absolute_threeDs[:, :2])

        for x_a, y_a, absolute_threeD in absolute_threeDs:
            x_a = int(x_a)
            y_a = int(y_a)
            if mask[y_a, x_a]:
                # value is masked in this array
                continue
            absolute_labels.append(labels[y_a, x_a])
            absolute_area = labels == absolute_labels[-1]
            threeD[absolute_area] += absolute_threeD - threeD[y_a, x_a]

    # setting all labeled areas without absolute_labels to have minimum close to threeD = 0
    # sign = int(np.sign(calibration['theta'])) if 'theta' in calibration else 1
    for j in range(1, n_labels + 1):
        if j in absolute_labels:
            continue
        area = labels == j
        # points = np.sort(threeD[area].flatten())
        # threeD[area] -= points[int(sign * round(_min_percentile * points.size))]
        threeD[area] -= np.min(threeD[area])

    Y, X = np.mgrid[: shape[0], : shape[1]]
    x_0, y_0 = principal_point
    X = (X - x_0) * xscale
    Y = -(Y - y_0) * yscale
    if "theta" in calibration and "scale" in calibration:
        X = X + threeD * calibration["T"] / 2 / np.pi / dscale * xscale

    if "phi" in calibration:
        phi = calibration["phi"]
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        X_copy = X.copy()
        X = cos_phi * X_copy + sin_phi * Y
        Y = -sin_phi * X_copy + cos_phi * Y

    # Print extent of bounding box for debug
    # print(np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(threeD) - np.min(threeD))

    if isMasked:
        mask = threeD.mask
        grid3d = np.ma.stack(
            (np.ma.array(X, mask=mask), np.ma.array(Y, mask=mask), threeD)
        )
        grid3d = roi.unapply(grid3d)
    else:
        grid3d = np.stack((X, Y, threeD))
    return grid3d
