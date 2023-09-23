"""
Module for phase demodulation of an FP-LIF image (fringe pattern image)

The main method is demodulate which takes an image and an optional calibration as input and extract the phase from it
"""
# import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration

from . import helpers as h
from . import wavelets as wl


def _possibly_masked(inArray, outArray):
    """Mask output array if inArray array is masked"""
    if np.ma.isMaskedArray(inArray):
        return np.ma.array(outArray, mask=inArray.mask, fill_value=inArray.fill_value)
    else:
        return outArray


def wavelet(s, T):
    """1D phase demodulation using wavelet"""
    c = wl.cwt(s, T)
    ridge = np.argmax(np.abs(c), axis=0)
    wrapped_phase = _possibly_masked(s, np.angle(c[ridge, np.arange(s.size)]))
    return wrapped_phase


def wave2_demodulation(s, T, gamma, wavelet):
    """Helper function for phase demodulation of image using wavelets"""
    # assert isinstance(gamma, float), "Wavelet demodulation only support gamma as float"
    shape = s.shape
    y, x = np.mgrid[: shape[0], : shape[1]]
    c = wl.cwt2(s, T, gamma, wavelet)
    if len(c.shape) == 4:
        c = c.reshape((gamma.size * T.size,) + shape)
    amplitude = np.abs(c)
    ridge = np.argmax(amplitude, axis=0)
    amplitude = (
        amplitude[ridge, y, x] / 1.6
    )  # scaling to get to real amplitude for some reason
    wrapped_phase = _possibly_masked(s, np.angle(c[ridge, y, x]))

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(s, cmap='gray'), plt.title("signal")
    # plt.subplot(122)
    # plt.imshow(T[ridge], cmap='gray'), plt.title("ridge Ts")
    # plt.show()
    # exit()

    return wrapped_phase, amplitude, ridge


def wavelet2(s, T, gamma):
    """Helper function for phase demodulation of image using the morlet wavelet"""
    return wave2_demodulation(s, T, gamma, "morl")


def isowavelet2(s, T, gamma=np.pi / 2):
    """Helper function for phase demodulation of image using the semi mexican hat"""
    return wave2_demodulation(s, T, gamma, "mexh")


def ft2(s, T, gamma, sigma=None):
    """Helper function for phase demodulatoin of image using the Fourier Transform method, see
    M. Takeda and K. Mutoh, “Fourier transform profilometry for the automatic measurement of 3-d object shapes,” Appl. Opt. 22, 3977–3982 (1983).
    """
    shape = s.shape
    T = T
    if sigma is None:
        sigma = 0.32 / T * 1.96 * np.mean(shape)
    lf_sync_freq = h.ft2_helper(s, T, gamma, sigma)
    wrapped_phase = _possibly_masked(s, np.angle(lf_sync_freq))
    return wrapped_phase


def get_Trange_from_calibration(calibration):
    Tstep = 1
    if "Tstep" in calibration:
        Tstep = calibration["Tstep"]
    Trange = None
    if "Tlim" in calibration:
        # Attempting to set using Tlim
        Tlim = calibration["Tlim"]
        if 3 < (Tlim[-1] - Tlim[0]) / Tstep < 10:
            Trange = np.arange(Tlim[0], Tlim[1], Tstep)
    if Trange is None:
        ### Use linear range in frequency instead
        T0 = calibration["T"]
        Trange = T0 + np.arange(-5 * Tstep, 5 * Tstep, Tstep)
        Trange = Trange[Trange >= 2]
    return Trange


def demodulate(
    signal: np.ndarray,
    calibration: dict,
    return_amplitude: bool = False,
    return_ridge: bool = False,
):
    """
    Function for demodulation using the Continuous Wavelet Transform method
    note that the returned amplitude is max - min of a cos wave. In other words A = 2 * a => I = a * cos(phase)
    """
    assert len(signal.shape) == 2, "Signal must be a single channel"

    Trange = get_Trange_from_calibration(calibration)

    # threeD = isowavelet2(signal, Trange, calibration['gamma'])
    wrapped_phase, amplitude, ridge = wavelet2(signal, Trange, calibration["gamma"])
    unwrapped_phase = restoration.unwrap_phase(wrapped_phase)

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(wrapped_phase), plt.title("wrapped phase")
    # plt.subplot(122)
    # plt.imshow(unwrapped_phase), plt.title("unwrapped phase")
    # plt.show()
    # exit()

    if return_amplitude and return_ridge:
        return unwrapped_phase, amplitude, ridge
    elif return_amplitude:
        return unwrapped_phase, amplitude
    elif return_ridge:
        return unwrapped_phase, ridge
    else:
        return unwrapped_phase
