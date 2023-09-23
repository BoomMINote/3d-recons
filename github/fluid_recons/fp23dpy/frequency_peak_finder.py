"""
Module for finding the peak of an image in the fourier plane.
It is used as calibration tool for finding the period length T and angle gamma of a Fringe Pattern in an image
"""
import numpy as np
from skimage import filters
# import time


_DEBUG = False
if _DEBUG:
    import matplotlib.pyplot as plt

def _find_zero_peak(F):
    """Special case for finding and removing the middle zero frequency peak of the fourier transform"""
    shape = np.array(F.shape)
    if shape[0] % 2 == 0 and shape[1] % 2 == 0:
        p = (shape / 2).astype(int)
    else:
        p_low = (shape / 2).astype(int)
        points = np.array([(p_low[0] + 0) * shape[1] + (p_low[1] + 0),
                           (p_low[0] + 1) * shape[1] + (p_low[1] + 0),
                           (p_low[0] + 0) * shape[1] + (p_low[1] + 1),
                           (p_low[0] + 1) * shape[1] + (p_low[1] + 1)])
        max_point = points[np.argmax(F.flatten()[points])]
        p = np.unravel_index(max_point, shape)
    return p

def _bfs_peak_rm(F, r, c):
    """Breadth first search of finding and removing a frequency peak, the full "mountain" instead of just the single value"""
    shape = np.array(F.shape)
    queue = [(r, c, float('inf'))]
    removed = {}
    while queue:
        ri, ci, last = queue.pop(0)
        if (ri >= 0 and ri < shape[0] and
            ci >= 0 and ci < shape[1] and
            0 < F[ri, ci] < last):
            removed[(ri, ci)] = F[ri, ci]
            last = F[ri, ci]
            F[ri, ci] = 0
            queue.extend([(ri, ci + 1, last), (ri, ci - 1, last),
                          (ri + 1, ci, last), (ri - 1, ci, last)])
    return np.array([k + (v,) for k, v in removed.items()])

def _subpixel_peak(peak):
    """Possibility to optimize the peak after removing, this has not been found to be robust"""
    r = np.average(peak[:, 0], weights=peak[:, 2])
    c = np.average(peak[:, 1], weights=peak[:, 2])
    return [r, c]

def _frequency_limits(shape, peak):
    """Calculates the frequency length of the mountain peak removed. The result is not used but it might be interesting to see"""
    f2s = np.sum(np.square((peak[:, :2] - shape / 2)  / shape), axis=1)
    min_f2 = np.min(f2s)
    max_f2 = np.max(f2s)
    return np.sqrt([min_f2, max_f2])

def _to_periods(peaks):
    """Going from frequencies to periods, T = 1 / f"""
    new_peaks = peaks.copy()
    new_peaks[:, 0] = 1 / peaks[:, 0]
    new_peaks[:, 2] = 1 / peaks[:, 3]
    new_peaks[:, 3] = 1 / peaks[:, 2]
    return new_peaks

_tol_freq = 5e-3
def find(F, n, return_frequency=False, subpixel_accuracy=False, equal_frequency=False, with_zero_peak=False):
    """
    Function to find the n largest frequency peaks in the Fourier transform of an image F.
    returns the peak frequency and the angle together with a rough size from min and max frequency of the peak.

    return return_frequency decides wheter frequency of periods should be returned.
    There is possibility to use subpixel accuracy (not that robust).
    If there are multiple frequency peaks in the image and all have the same frequency and different angle equal frequency can be set to True to filter out some of the bad ones.
    One might want to include the zero frequency peak even, not used.
    """
    shape = np.array(F.shape)
    l = 2 * n
    if equal_frequency:
        l += 5

    F = filters.gaussian(np.abs(F), 5)

    # zero peak
    zp = np.zeros(4)
    p = _find_zero_peak(F)
    if _DEBUG:
        print("Zero peak")
        print(p)
        plt.subplot(121)
        plt.imshow(F, cmap='gray')
        plt.plot(p[1], p[0], '.')
    zp[:2] = p
    zp_peak = _bfs_peak_rm(F, p[0], p[1])
    zp[2:] = _frequency_limits(shape, zp_peak)
    peak = _bfs_peak_rm(F, p[0], p[1])
    if _DEBUG:
        plt.subplot(122)
        plt.imshow(F, cmap='gray')
        plt.show()
        print("Rest peaks")

    m_diagonal_line = -shape[0] / shape[1]

    peaks = np.zeros((n + 1, 4))
    j = 0
    for i in range(l):
        p = np.unravel_index(np.argmax(F), shape)
        if _DEBUG:
            print(p)
            plt.subplot(121)
            plt.imshow(F, cmap='gray')
            plt.plot(p[1], p[0], '.')
        peak = _bfs_peak_rm(F, p[0], p[1])
        if p[0] >= m_diagonal_line * p[1] + shape[0]:
            if subpixel_accuracy:
                p = _subpixel_peak(peak)
                # print(p)
            peaks[j, :2] = p
            peaks[j, 2:] = _frequency_limits(shape, peak)
            j += 1
        if _DEBUG:
            plt.subplot(122)
            plt.imshow(F, cmap='gray')
            plt.show()

    peaks[:, :2] -= shape / 2
    rs = np.sqrt(np.sum(np.square(peaks[:, :2] / shape), axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        angles = np.arctan2(peaks[:, 0], peaks[:, 1])
    peaks[:, 0] = rs
    peaks[:, 1] = angles
    peaks[np.isnan(peaks)] = 0
    if _DEBUG:
        print(peaks)

    # Removing possible double 3 pi / 4 peaks, not yet implemented
    # ind = np.where(np.abs(np.abs(peaks[:, 1]) - np.pi / 2) < _tol_freq)[0][:2]
    # if ind.size >= 2:
    #     peaks[ind[0], 0] = np.mean(peaks[ind, 0])
    #     peaks[ind[0], 1] = np.mean(np.abs(peaks[ind, 1]))
    #     peaks[ind[0], 2] = np.min(peaks[ind, 2])
    #     peaks[ind[0], 3] = np.max(peaks[ind, 3])
    #     peaks = np.delete(peaks, ind[1], axis=0)

    if equal_frequency:
        freq = peaks[0, 0]
        ind = np.abs(peaks[:, 0] - freq) < _tol_freq
        peaks = peaks[ind]
    if with_zero_peak:
        peaks = np.vstack((zp, peaks))
        n += 1

    if not return_frequency:
        peaks = _to_periods(peaks[:n])
    return peaks

def find_from_signal(s, n, **kwargs):
    """Same as the find method with the difference that the input image is not fourier transformed"""
    if len(s.shape) > 2:
        raise ValueError("Signal image (s) must only have one channel")
    F = np.fft.fftshift(np.fft.fft2(s))
    return find(F, n, **kwargs)
