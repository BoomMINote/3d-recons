"""
Module for manipulating rectangle of interest stuff
"""
import numpy as np


class Roi:
    """
    Class for manipulating rectangle of interest (roi) stuff using cropping of images
    """

    def __init__(self, l=None):
        """
        l is the input of roi information that can either be list of length 4 or 6
        if length 4
            l = [min_row, max_row, min_col, max_col]
        if length 6
            add from original image [height, width]
            with this you can also unapply (go from small image to large)
        """
        if l is None:
            self.roi = None
        elif isinstance(l, (np.ndarray, list)):
            assert (
                len(l) == 6 or len(l) == 4
            ), "Roi list should have length 4 or 6, {} found".format(len(l))
            l = np.asarray(l)
            if len(l) == 6:
                # indicies can use negative values to take inds from the end of array.
                # Here they are converted to positive indicies using the width and height of the original image
                l[1] = l[1] if l[1] > 0 else l[-2] + l[1]
                l[3] = l[3] if l[3] > 0 else l[-1] + l[3]
            self.roi = l

    def full(shape):
        """ Roi with same shape as shape """
        h, w = shape
        return Roi([0, h, 0, w, h, w])

    def find_from_mask(mask):
        """
        With a mask where false values are of interest. Find the smallest rectangle than enclose all unmasked values
        """
        [h, w] = mask.shape
        r, c = np.where(~mask)
        edge = 10
        minr = max(0, np.min(r) - edge)
        minc = max(0, np.min(c) - edge)
        maxr = min(mask.shape[0], np.max(r) + edge)
        maxc = min(mask.shape[1], np.max(c) + edge)
        return Roi([minr, maxr, minc, maxc, h, w])

    def mask_from_roi(self):
        """
        With the roi return a mask where false values are inside Roi
        """
        assert (
            not self.full_shape() is None
        ), "Full shape of the original image has not been given when constructing this roi"
        mask = np.ones(self.full_shape(), dtype=bool)
        roi = self.roi
        mask[roi[0] : roi[1], roi[2] : roi[3]] = False
        return mask

    def apply(self, image):
        """
        Crop an image using the roi. Extract only the rows and columns of interest.
        For a three dimensional array a matching is attempted to see which dimension should be broadcasted.
        """
        if self.empty():
            return image
        roi = self.roi
        if len(image.shape) == 2:
            image = image[roi[0] : roi[1], roi[2] : roi[3]]
        elif len(image.shape) == 3:
            assert (
                not self.full_shape() is None
            ), "Full shape of the original image has not been given when constructing this roi"
            full_shape = tuple(self.full_shape())
            if image.shape[1:] == full_shape:
                image = image[:, roi[0] : roi[1], roi[2] : roi[3]]
            elif image.shape[:2] == full_shape:
                image = image[roi[0] : roi[1], roi[2] : roi[3], :]
            else:
                raise ValueError(
                    "Shape added as two last values in constructor {} does not match any of the image dimensions {}".format(
                        full_shape, image.shape
                    )
                )
        else:
            raise ValueError(
                "Wrong number of image dimensions or not able to detemine the channel axis of images"
            )
        return image

    def unapply(self, image):
        """
        Pad a cropped image (from apply) with zeros to get back to the full shape of the image
        """
        if self.empty():
            return image

        roi = self.roi
        full_shape = self.full_shape()
        if full_shape is None:
            assert (
                roi[1] < 0 and roi[3] < 0
            ), "Max rows and cols for roi must be negative if not full shape is given"
            # Full shape was not given when creating the roi, estimating it
            small_shape = image.shape
            extra_rows = roi[0] - roi[1]
            extra_cols = roi[2] - roi[3]
            full_shape = [small_shape[0] + extra_rows, small_shape[1] + extra_cols]

        if len(image.shape) == 2:
            full_image = np.ma.zeros(full_shape)
            full_image.mask = True
            full_image[roi[0] : roi[1], roi[2] : roi[3]] = image
        elif len(image.shape) == 3 and image.shape[0] == 3:
            full_shape = [
                image.shape[0],
            ] + list(full_shape)
            full_image = np.ma.zeros(full_shape)
            full_image.mask = True
            full_image[:, roi[0] : roi[1], roi[2] : roi[3]] = image
        elif len(image.shape) == 3 and image.shape[0] == 3:
            full_shape = list(full_shape) + [
                image.shape[2],
            ]
            full_image = np.ma.zeros(full_shape)
            full_image.mask = True
            full_image[roi[0] : roi[1], roi[2] : roi[3], :] = image
        else:
            raise ValueError(
                "Wrong number of image dimensions or not able to detemine the channel axis of images"
            )
        return full_image

    def apply_to_points(self, points):
        """
        Each row in the points array is point and the columns are x and y
        Update the points so that zero is on top left of roi
        """
        if self.empty():
            return points
        points = np.array(points)
        points[:, 0] -= self.roi[2]
        points[:, 1] -= self.roi[0]
        return points

    def unapply_to_points(self, points):
        """
        Each row in the points array is point and the columns are x and y
        Update the points that are in the roi coordinate system so that zero is on top left of the original image
        """
        if self.empty():
            return points
        points = np.array(points)
        points[:, 0] += self.roi[2]
        points[:, 1] += self.roi[0]
        return points

    def rebase(self, full_roi):
        """ Use a new roi as full image shape """
        if self.empty() or full_roi.empty():
            return
        full_roi = full_roi.roi
        new_roi = self.roi.copy()
        new_roi[0] -= full_roi[0]
        new_roi[1] -= full_roi[0]
        new_roi[2] -= full_roi[2]
        new_roi[3] -= full_roi[2]
        new_roi[4] = full_roi[1] - full_roi[0]
        new_roi[5] = full_roi[3] - full_roi[2]
        self.roi = new_roi

    def maximize(self, roi2):
        """ update this roi as one roi is large enough to enclose both self and roi2 """
        roi1 = self.roi
        roi2 = roi2.roi
        if roi1 is None and roi2 is None:
            roi = None
        elif roi1 is None:
            roi = roi2.copy()
        elif roi2 is None:
            roi = roi1
        else:
            roi = [
                min(roi1[0], roi2[0]),
                max(roi1[1], roi2[1]),
                min(roi1[2], roi2[2]),
                max(roi1[3], roi2[3]),
                max(roi1[4], roi2[4]),
                max(roi1[5], roi2[5]),
            ]
        self.roi = roi

    def enlarge(self, diff):
        """ make roi diff larger in all directions """
        self.roi[0] -= diff
        self.roi[1] += diff
        self.roi[2] -= diff
        self.roi[3] += diff

        # Avoid negative values
        self.roi[0] = max(0, self.roi[0])
        self.roi[2] = max(0, self.roi[2])

        if len(self.roi) == 6:
            # with original image shape too large numbers are also avoided
            self.roi[1] = min(self.roi[4], self.roi[1])
            self.roi[3] = min(self.roi[5], self.roi[3])

    def empty(self):
        return self.roi is None

    def full_shape(self):
        if self.empty() or len(self.roi) == 4:
            return None
        else:
            return tuple(self.roi[4:])

    def shape(self):
        h = self.roi[1] - self.roi[0]
        w = self.roi[3] - self.roi[2]
        return (h, w)

    def apply_to_square_carrier(self, square_T):
        x0 = self.roi[2]
        y0 = self.roi[0]
        new_square_T = square_T.copy()
        new_square_T[0] = (
            square_T[0]
            + square_T[1] * x0
            + square_T[2] * y0
            + square_T[3] * x0 ** 2
            + square_T[4] * y0 ** 2
        )
        new_square_T[1] = square_T[1] + 2 * square_T[3] * x0
        new_square_T[2] = square_T[2] + 2 * square_T[4] * y0
        return new_square_T

    def as_pyplot_imshow_extent(self):
        roi = self.roi
        return [roi[2], roi[3], roi[1], roi[0]]

    def copy(self):
        if self.empty():
            return Roi()
        else:
            return Roi(self.roi.copy())

    def __str__(self):
        return "Roi: " + self.roi.__str__()

    def write(self, path):
        np.savetxt(path, self.roi, fmt="%d")
