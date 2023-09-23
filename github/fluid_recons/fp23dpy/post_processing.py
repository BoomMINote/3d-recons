""" Functions to post process the reconstructed 3D data """

import numpy as np
from skimage import measure

from . import helpers
from . import fp23dpy
# import matplotlib.pyplot as plt


def flood_find_closest(start_point, segmentation):
    shape = segmentation.shape
    queue = [start_point]
    closest_point = None
    visited = np.zeros(shape, dtype=bool)
    while len(queue) > 0:
        r, c = queue.pop(0)
        if r < 0 or r >= shape[0] or c < 0 or c >= shape[1]:
            continue

        if not visited[r, c]:
            visited[r, c] = True
            queue.extend([[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]])
        if segmentation[r, c]:
            closest_point = np.array([r, c])
            break

    return closest_point


def get_label_centers(labels, n_labels):
    label_centers = []
    for l in range(1, n_labels + 1):
        area = labels == l
        label_center = np.round(np.mean(np.where(area), axis=1)).astype(int)
        if not area[tuple(label_center)]:
            # find closest pixel to the center to use as center
            label_center = flood_find_closest(label_center, area)
        label_centers.append(label_center)
    return label_centers

def temporal_alignment(reconstructions, mean_velocity=None):
    """Attempt to temporally align the reconstructions by tracking blobs.
    This will only be applied to blobs that get disconnected from the main area with pixels that are connected to the absolute_phase pixel in calibration.
    All reconstructions should have the same original shape of images and absolute_coordinates set for their calibration. If not these are met, an assertion error is raised.

    This function is not that robust but works ok if the objects are moving less than 10 pixels per frame

    Maybe works maybe not, if you do not try you will never find out

    :reconstructions: list of dicts
        all reconstructions to align, each reconstruction is a dict with at least the keys grid and calibration
    :returns: None
    """
    # Check that all reconstructions have absolute_coordinates
    # and check that the reconstructions all have the same shape
    absolute_coordinates = []
    shape = reconstructions[0]["grid"].shape
    for reconstruction in reconstructions:
        current_absolute_coordinates = []
        if "absolute_phase" in reconstruction["calibration"]:
            current_absolute_coordinates.append(
                np.array(reconstruction["calibration"]["absolute_phase"])
            )
        if "absolute_threeD" in reconstruction["calibration"]:
            current_absolute_coordinates.append(
                np.array(reconstruction["calibration"]["absolute_threeD"])
            )
        if len(current_absolute_coordinates) > 0:
            for i in range(len(current_absolute_coordinates)):
                if len(current_absolute_coordinates[i].shape) == 1:
                    current_absolute_coordinates[i] = np.expand_dims(current_absolute_coordinates[i], 0)
            current_absolute_coordinates = np.concatenate(current_absolute_coordinates).astype(int)
            current_absolute_coordinates = current_absolute_coordinates[:, :2]
            absolute_coordinates.append(current_absolute_coordinates)
        assert (
            shape == reconstruction["grid"].shape
        ), "All shapes must be the same for temporal alignment"
    assert len(absolute_coordinates) == len(
        reconstructions
    ), "Need absolute_coordinates for all reconstructions to use --temporal-alignment"

    last_grid = reconstructions[0]["grid"]
    last_segmentation = ~helpers.get_mask(last_grid[0])
    last_labels, last_n_labels = measure.label(
        last_segmentation.astype(int), return_num=True, connectivity=1
    )
    last_label_centers = np.expand_dims(get_label_centers(last_labels, last_n_labels), 0)

    # to estimate velocity in 3D at least two previous reconstructions are required
    for i in range(1, len(reconstructions)):
        reconstruction = reconstructions[i]
        grid = reconstruction["grid"]
        segmentation = ~helpers.get_mask(grid[0])

        # Using labeled connected components to find the different areas in the image
        labels, n_labels = measure.label((segmentation).astype(int), return_num=True, connectivity=1)
        current_absolute_coordinates = absolute_coordinates[i]
        absolute_labels = set([labels[tuple(coordinates[::-1])] for coordinates in current_absolute_coordinates])
        label_centers = np.expand_dims(get_label_centers(labels, n_labels), 0)

        area_tracks = {}
        if "area_tracks" in reconstruction["calibration"]:
            for tracked_area in reconstruction["calibration"]["area_tracks"]:
                area_label = labels[tuple(tracked_area[0][::-1])]
                if area_label == 0:
                    raise ValueError("Area track is on a masked area for {}".format(reconstruction["filename"]))
                if not last_segmentation[tuple(tracked_area[1][::-1])]:
                    raise ValueError("Area track on last image is on a masked area for {}".format(reconstruction["filename"]))
                area_tracks[area_label] = tracked_area
        else:
            continue

        for j in range(1, n_labels + 1):
            if j in absolute_labels:
                continue  # no tracking required for areas with known absolute phase
            area = labels == j
            if j in area_tracks:
                this_area_coordinates = area_tracks[j][0][::-1]
                other_area_coordinates = area_tracks[j][1][::-1]
            else:
                print("Warning: No area_track found for {} label {}, using closest area.".format(reconstruction["filename"], j))
                distances = np.linalg.norm(label_centers - np.transpose(last_label_centers, (1, 0, 2)), axis=-1)
                closest_ind = np.argmin(distances)
                closest_absolute_pixel_ind, closest_area_ind = np.unravel_index(closest_ind, distances.shape)
                other_area_coordinates = last_label_centers[0, closest_absolute_pixel_ind]
                this_area_coordinates = label_centers[0, closest_area_ind]
            
            this_area_val = grid[(2, this_area_coordinates[0], this_area_coordinates[1])]
            other_area_val = last_grid[(2, other_area_coordinates[0], other_area_coordinates[1])]

            calibration = reconstruction["calibration"]
            xscale, yscale, dscale = fp23dpy.get_scales(calibration)
            xdiff = ydiff = 0
            ddiff = other_area_val - this_area_val
            # similar as in fp23dpy.fp23dpy
            if 'theta' in calibration and 'scale' in calibration:
                xdiff = ddiff * calibration['T'] / 2 / np.pi / dscale * xscale 
            if 'phi' in calibration:
                phi = calibration['phi']
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                xdiff_copy = xdiff
                xdiff =  cos_phi * xdiff_copy + sin_phi * ydiff
                ydiff = -sin_phi * xdiff_copy + cos_phi * ydiff

            diff = np.array([[xdiff], [ydiff], [ddiff]])

            extraction_tuple = (slice(None),) + np.where(area)
            grid[extraction_tuple] = grid[extraction_tuple] + diff

        last_grid = grid
        last_segmentation = segmentation
        last_label_centers = label_centers
