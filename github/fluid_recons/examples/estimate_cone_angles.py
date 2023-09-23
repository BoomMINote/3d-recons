from skimage import measure
import numpy as np

from fp23dpy.examples import example_cone


def get_azimuth_angles(grid3d):
    mask = grid3d.mask[0]
    shape = grid3d.shape[1:]
    azimuth_angles = np.empty(shape)
    azimuth_angles[mask] = np.nan
    azimuth_angles[~mask] = np.arctan2(grid3d[2][~mask], grid3d[0][~mask]) * 180 / np.pi
    return mask, azimuth_angles


def get_azimuth_angle_coordinates(azimuth_angles, mask, angle):
    rs, cs = np.where(azimuth_angles >= angle)

    ys = np.unique(rs).astype(int)
    xs = np.zeros(mask.shape[0], dtype=int)
    valid = np.zeros(xs.size, dtype=bool)
    for y in ys:
        current_inds = rs == y
        if np.sum(current_inds) < 0:
            continue
        x = np.max(cs[current_inds])
        # if the pixel to the left or right of the chosen one is
        # masked it is an invalid pixel
        if x <= 0 or mask.shape[1] <= x + 1 or mask[y, x + 1] or mask[y, x - 1]:
            continue
        xs[y] = np.max(cs[current_inds])
        valid[y] = True
    ys = np.arange(mask.shape[0], dtype=int)
    return (ys, xs), valid


def get_radiuses(grid3d, azimuth_angles_to_extract, radius0):
    mask, azimuth_angles = get_azimuth_angles(grid3d)
    radiuses = np.zeros((len(azimuth_angles_to_extract), grid3d.shape[1]))
    radiuses[:] = np.nan
    for i, azimuth_angle_to_extract in enumerate(azimuth_angles_to_extract):
        coordinates, valid = get_azimuth_angle_coordinates(
            azimuth_angles, mask, azimuth_angle_to_extract
        )
        valid_coordinates = (coordinates[0][valid], coordinates[1][valid])
        radius = np.empty(radiuses.shape[1])
        radius[valid] = (
            np.sqrt(
                grid3d[0][valid_coordinates] ** 2 + grid3d[2][valid_coordinates] ** 2
            )
            - radius0
        )
        radius[~valid] = np.nan
        radiuses[i] = radius
    return radiuses


def estimate_single_spray_angle(data):
    linemodel = measure.LineModelND()
    try:
        linemodel.estimate(data)
    except Exception as e:
        print(e)
        print("Exception warning")
        return np.nan
    spray_angle = np.arctan(linemodel.params[1][0] / linemodel.params[1][1])
    spray_angle *= 180 / np.pi * -2  # the -2 is to correspond to the notation
    return spray_angle


def get_spray_angles(grid3d, azimuth_angles_to_extract, ylims, radius0):
    radiuses = get_radiuses(grid3d, azimuth_angles_to_extract, radius0)
    y = np.nanmax(grid3d[1], axis=1)  # assumes stuff about the yaxis
    ymin, ymax = ylims
    valid_inds = (ymin < y) & (y < ymax)

    estimated_spray_angles = np.zeros(len(azimuth_angles_to_extract))
    estimated_spray_angles[:] = np.nan
    for i in range(len(azimuth_angles_to_extract)):
        current_valid_inds = valid_inds & np.isfinite(radiuses[i])
        radius_valid = radiuses[i][current_valid_inds]
        y_valid = y[current_valid_inds]
        n_inliers = np.sum(np.isfinite(radius_valid))
        if n_inliers < 20:
            print("warning, few inliers in the estimation", n_inliers)
            continue
        data = np.column_stack((radius_valid, y_valid))
        estimated_spray_angle = estimate_single_spray_angle(data)
        estimated_spray_angles[i] = estimated_spray_angle

    return estimated_spray_angles


if __name__ == "__main__":
    ylims = (-50, -10)
    azimuth_angles = [40, 80, 120]
    cone_grid = example_cone.get_projected_coordinate_grid()
    spray_angles = get_spray_angles(
        cone_grid, azimuth_angles, ylims, example_cone.radius0
    )

    # should all be equal to 50
    print(spray_angles)
