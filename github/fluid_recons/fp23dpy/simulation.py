import numpy as np
import scipy
from scipy import interpolate

# import trimesh
import skimage
import warnings

from . import threeD_to_phase_const

# from . import export
from . import helpers as h


def _create_camera_matrix(calibration):  # not fully implemented
    if "camera_type" in calibration:
        raise ValueError("camera_type not supported yet")
    else:
        return np.eye((3, 4))


def estimate_rotsym_projection_map(
    coordinate_grid,
    image_shape,
    camera_radius,
    recorded_image_shape,
    calibration,
    camera_y=150,  # -40,
    cone_angle=45 * np.pi / 180,
    min_Y=20,
):  # currently not fully implemented
    """
    Use projective imaging, the results should be used with render_from_xmap function
    """
    camera_theta_rad = calibration["theta"]
    camera_position = camera_radius * np.array(
        [np.sin(camera_theta_rad), camera_y / camera_radius, np.cos(camera_theta_rad)]
    )
    camera_rotation = np.array(
        [
            [-np.cos(camera_theta_rad), 0, np.sin(camera_theta_rad)],
            [0, 1, 0],
            [-np.sin(camera_theta_rad), 0, -np.cos(camera_theta_rad)],
        ]
    )
    camera_extrinsic_matrix = np.column_stack(
        (camera_rotation, -camera_rotation @ camera_position)
    )
    scale_points = np.array([[0, -1 / 2, 0, 1], [0, 1 / 2, 0, 1]])
    projected_scale_points = camera_extrinsic_matrix @ scale_points.T
    projected_scale_points_normed = (
        projected_scale_points[:-1] / projected_scale_points[-1]
    )
    focal_length = 1 / (
        projected_scale_points_normed[1, 1] - projected_scale_points_normed[1, 0]
    )
    camera_intrinsic_matrix = np.array(
        [
            [-focal_length, 0, recorded_image_shape[0] / 2],
            [0, -focal_length, recorded_image_shape[1] / 2],
            [0, 0, 1],
        ]
    )
    camera_matrix = camera_intrinsic_matrix @ camera_extrinsic_matrix

    # coordinate_grid[1] -= coordinate_grid[
    #     1, recorded_image_shape[0] // 2, recorded_image_shape[1] // 2
    # ]
    # coordinate_grid = coordinate_grid.filled(np.nan)

    coordinate_grid_stacked = np.vstack(
        (coordinate_grid, np.ones((1,) + coordinate_grid.shape[1:]))
    )

    projected_camera_coordinates = np.dot(
        camera_matrix, coordinate_grid_stacked.transpose((1, 0, 2))
    )
    projected_camera_coordinates_normed = (
        projected_camera_coordinates[:-1] / projected_camera_coordinates[-1]
    )
    xdiffs = np.diff(projected_camera_coordinates_normed[0], axis=1)
    # these parts of the original structure is behind other structures
    xinvalid = xdiffs < 0
    xinvalid = np.hstack((xinvalid, np.zeros((len(xinvalid), 1), dtype=bool)))
    projected_camera_coordinates_normed[:, xinvalid] = np.nan

    valid_points = np.isfinite(projected_camera_coordinates_normed[0])
    points = projected_camera_coordinates_normed[:, valid_points].reshape((2, -1)).T
    values = coordinate_grid[:, valid_points].reshape((3, -1)).T
    interpolator = interpolate.LinearNDInterpolator(points, values)
    Y, X = np.mgrid[: recorded_image_shape[0], : recorded_image_shape[1]]
    image_map = interpolator(np.dstack((X, Y))).transpose((2, 0, 1))

    too_loow = image_map[1] < min_Y
    image_map[:, too_loow] = np.nan

    # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # axs[0].imshow(image_map[0])
    # axs[1].imshow(image_map[1])
    # axs[2].imshow(image_map[2])
    # plt.show()
    # exit()

    current_calibration = calibration.copy()
    current_calibration["scale"] = 1
    current_calibration["absolute_threeD"] = [
        image_shape[1] // 2,
        image_shape[0] // 2,
        image_map[2, recorded_image_shape[0] // 2, recorded_image_shape[1] // 2],
    ]
    y0 = np.nanargmin(np.abs(image_map[1, :, recorded_image_shape[0] // 2]))
    x0 = recorded_image_shape[1] // 2
    current_calibration["principal_point"] = [x0, y0]

    return image_map, current_calibration


#### Simplified functions for rotational symmetry in the XZ plane ###
def get_rotsym_depth(X, Y, radius):
    """Get the third coordinate of from X and Y (X and Y must have same shape) coordinates.
    Radius is either a function or value that describes the radius of the rotational symmetric structure in the XZ plane
    """
    if isinstance(X, (int, float)):
        X = np.array([X])
        Y = np.array([Y])
    assert X.shape == Y.shape
    depth = np.zeros(X.shape)
    if hasattr(radius, "__call__"):
        # radius is a function that takes the Y parameter
        rs = radius(Y)
    else:
        # assuming radius is a number
        rs = np.ones(X.shape) * radius
    inside = np.abs(X) < rs
    depth[inside] = rs[inside] * np.sin(np.arccos(X[inside] / rs[inside]))
    depth = np.ma.array(depth, mask=~inside)
    return depth


def get_rotsym_coordinate_grid(radius, shape, scale=1, mid_y=0):
    """Create 3D coordinates for a half of the rotational symmetric structure described by radius function or value."""
    mid_x = shape[0] / 2

    Y, X = np.mgrid[: shape[0], : shape[1]]
    X = (X - mid_x) / scale
    Y = -(Y - mid_y) / scale
    depth = get_rotsym_depth(X, Y, radius)
    mask = depth.mask
    return np.ma.stack((np.ma.array(X, mask=mask), np.ma.array(Y, mask=mask), depth))


def get_rotsym_projection_map(coordinate_grid, calibration, **kwargs):
    """Camera mapping of image x to world X coordinate on the cone structure taking advantage of rotational symmetry"""
    if hasattr(coordinate_grid, "__call__"):
        # Assuming here that the coordinate_grid variable is radius to estimate coordinate_grid
        coordinate_grid = get_rotsym_coordinate_grid(radius=coordinate_grid, **kwargs)
    if not np.ma.isMaskedArray(coordinate_grid):
        coordinate_grid = np.ma.array(coordinate_grid)
    shape = coordinate_grid.shape
    coordinate_grid = coordinate_grid.reshape((3, -1))
    theta = calibration["theta"]
    Xmap = coordinate_grid[0] * np.cos(theta) + coordinate_grid[2] * np.sin(theta)
    proj = -coordinate_grid[0] * np.sin(theta) + coordinate_grid[2] * np.cos(theta)
    masked = proj < 0
    Xmap[masked] = np.ma.masked
    Ymap = np.ma.array(coordinate_grid[1], mask=Xmap.mask)
    proj[masked] = np.ma.masked
    return np.ma.stack((Xmap, Ymap, proj)).reshape(shape)


#################

footprint_par = np.array([[1, 1], [1, 0]], dtype=bool)
footprint_diag = np.array([[1, 0], [0, 1]], dtype=bool)


def estimate_phase_gradient(phase):
    min_phase = np.min(phase)
    scaling = (2**12 - 1) / (np.max(phase) - min_phase)
    phase_normalized = np.round((phase - min_phase) * scaling).astype(np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gradient_par = skimage.filters.rank.gradient(phase_normalized, footprint_par)
        gradient_diag = skimage.filters.rank.gradient(
            phase_normalized, footprint_diag
        ) / np.sqrt(2)
    gradient = np.maximum(gradient_par, gradient_diag) / scaling
    if np.ma.isMaskedArray(phase):
        gradient[phase.mask] = np.nan
    return gradient


def is_phase_resolved(phase, resolution_limit=np.pi):
    gradient = estimate_phase_gradient(phase)
    return np.all(gradient < resolution_limit)


def get_phase_from_map(dmap, calibration):
    dmap = np.ma.asarray(dmap)
    tpc = threeD_to_phase_const(
        calibration["T"], calibration["theta"], calibration["scale"]
    )
    if "phi" in calibration:
        raise NotImplementedError()
        pass  # multiply constant slightly
    carrier = h.make_carrier(dmap.shape, calibration["T"], calibration["gamma"])
    phase = dmap * tpc + carrier
    return phase


def get_signal_from_phase(phase, amplitude, background):
    signal = amplitude * np.cos(phase) + background

    mask = phase.mask
    # convolving to simulate the integral over each pixel for a real camera
    kernel = 1 / 5 * np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    signal = np.ma.array(scipy.signal.convolve2d(signal, kernel, "same"), mask=mask)
    background = background * np.ones(phase.shape)
    signal.data[signal.mask] = background[signal.mask]
    return signal


def render_from_map(dmap, calibration, amplitude=1, background=1):
    """
    Method used to simulate an FP image of a 3D structure by mainly using a map of the world third coordinate of the object surface in the camera.
    Assumes that an orthographic camera rotated theta radians round the y-axis.
    If not the full image is used as for the drop case the dmap parameter should be a masked numpy array and the function will always return a masked array.
    """
    phase = get_phase_from_map(dmap, calibration)
    # if not is_phase_resolved(phase):
    #     raise ValueError("Input parameters gives an unresolvable phase.")
    signal = get_signal_from_phase(phase, amplitude, background)
    return signal


def render_from_xmap(xmap, calibration, amplitude=1, background=1):
    """
    similar to previous but use xmap instead of dmap, should be used with the projective simulations
    """
    Tg = calibration["T"] / calibration["scale"] / np.cos(calibration["theta"])
    phase = 2 * np.pi / Tg * xmap
    signal = get_signal_from_phase(phase, amplitude, background)
    return signal


def render(coordinate_grid, shape, calibration, use_rotsym=True):
    """Helper function to render a general 3D structure, if use_rotsym=True rotational symmetry in the XZ plane is assumed."""
    if use_rotsym:
        _, _, dmap = get_rotsym_projection_map(coordinate_grid, calibration)
    else:
        _, _, dmap = estimate_projection_map(coordinate_grid, calibration, shape)
    return render_from_map(dmap, calibration)
