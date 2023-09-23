"""
Module for the Continuous Wavelet Transform of functions
"""
import numpy as np
from scipy import fft

# ------------- Constants and functions ------------------- #
_sigma = 0.6
_eta_morl = 6.
_epsilon = 3.
_A = np.array([[1, 0], [0, 1 / np.sqrt(_epsilon)]])
_k0 = np.array([_eta_morl, 0])

_sigma_mexh_angle = 0.6
_eta_mexh = 4 / 3 / np.sqrt(np.pi) * 2  # should the two be there?

# f_inst = eta / 2 / pi / a

# Morlet/Gabor wavelets
def _morl1d(x):
    return np.exp(1j * _eta_morl * x) * np.exp(-np.square(x) / 2 / _sigma**2)


def _fmorl1d(w):
    return np.exp(-_sigma**2 / 2 * np.square(w - _eta_morl))


def _morl2d(x):
    return np.exp(1j * (_k0[0] * x[:, :, 0] + _k0[1] * x[:, :, 1]) -
                  np.linalg.norm(np.dot(x, _A), 
                                 axis=-1)**2 / 2 / _sigma**2)

def _fmorl2d(w):
    s = w.shape
    w2 = w.view()
    w2.shape = (int(w.size / 2), 2)
    xind = w2[:, 0] > 0
    xind.shape = w.shape[:-1]
    return xind * np.exp(-_sigma**2 / 2 *
                  np.linalg.norm(np.dot(w, _A) -
                                 _k0.reshape((1,) * (len(s) - 1) + (2,)),
                                 axis=-1)**2)

# Mexican hat wavelets
def _mexh2d(x):
    sqnorm = np.square(np.linalg.norm(x, axis=-1))
    return (1 - sqnorm) * np.exp(-sqnorm / 2)

# Does not work for phase demodulation
def _fmexh2d(w):
    sqnorm = np.square(np.linalg.norm(w, axis=-1))
    return sqnorm * np.exp(-sqnorm / 2)

# This one works for phase demodulation
def _fsemimexh2d(w):
    w2 = w.view()
    w2.shape = (int(w.size / 2), 2)
    # gamma = np.arctan2(w2[:, 1], w2[:, 0])
    # gamma_gauss = np.exp(-gamma**2 / 2 / _sigma_mexh_angle**2)
    # gamma_gauss[w2[:, 0] < 0] = 0
    # gamma_gauss.shape = w.shape[:-1]
    # multiply all coefficients with negative x-freq with zero (remove them) do NOT ask me why!
    negative_freq_mul = w2[:, 0] > 0
    negative_freq_mul.shape = w.shape[:-1]


    sqnorm = np.square(np.linalg.norm(w, axis=-1))
    return sqnorm * np.exp(-sqnorm / 2) * negative_freq_mul


def cwt(s, T):
    """1D continuous wavelet transform"""
    a = np.reshape(_eta_morl / 2 / np.pi * T, (T.size, 1))
    F = np.tile(fft.fft(s), (a.size, 1))
    w = np.tile(2 * np.pi * fft.fftfreq(s.size), (a.size, 1))
    return np.sqrt(a) * fft.ifft(F * np.conj(_fmorl1d(w * a)), workers=-1)


def cwt2(s, T, gamma=0., wavelet="morl", combinations=True):
    """
    2D Continuous Wavelet Transform
    
    T is the period lenghts for the wavelets ang gamma is the rotation of the wavelets
    the currently possible wavelets to use are "morl" (morlet) and mexh" (mexican hat)
    combinations is an indication if all possible combinations of T and gamma should be used
    """
    wb = WaveBank(s.shape, T, gamma, wavelet, combinations)
    return wb.transform(s)

def _make_2d_w(shape):
    y, x = np.mgrid[:shape[0], :shape[1]]
    w = np.dstack((x, y))
    w = w / np.array([[[shape[1] - 1, shape[0] - 1]]]) * 2 * np.pi - np.pi
    return fft.ifftshift(w, axes=(0, 1))

def _create_wavebank(shape, wavefun, a, gamma, combinations):
    is_a_array = isinstance(a, np.ndarray)
    is_gamma_array = isinstance(gamma, np.ndarray)
    a = a if is_a_array else np.array([a])
    gamma = gamma if is_gamma_array else np.array([gamma])
    if combinations:
        a_shape = (1, a.size, 1, 1, 1)
        gamma_shape = (1, 1, 1, 1, gamma.size)
        wf_shape = (gamma.size, a.size, 1, 1, 1)
    else:
        assert a.size == gamma.size, "With combinations==False, a and gamma must be of equal length"
        a_shape = (a.size, 1, 1, 1)
        gamma_shape = (1, 1, 1, gamma.size)
        wf_shape = (a.size, 1, 1, 1)

    a.shape = a_shape
    rotations = gamma.reshape(gamma_shape)
    rotations = np.tile(rotations, (2, 2) + (len(gamma_shape) - 2) *(1,))
    rotations[0, 0] = np.cos(rotations[0, 0])
    rotations[1, 0] = np.sin(rotations[1, 0])
    rotations[0, 1] = -rotations[1, 0]  # -np.sin(rotations[1, 0])
    rotations[1, 1] = rotations[0, 0]  # np.cos(rotations[1, 1])
    rotations = np.transpose(rotations)
    rotations = np.swapaxes(rotations, -2, -1)

    wf = np.tile(_make_2d_w(shape), wf_shape)
    wf2 = np.matmul(wf, rotations)

    # Deleting negative x values for the isotropic wavelet to work
    # wf2_shape = wf2.shape
    # wf2 = wf2.reshape((int(wf2.size / 2), 2))
    # wf2[wf2[:, 0] < 0, :] = 0
    # wf2 = wf2.reshape(wf2_shape)

    wavebank = wavefun(wf2 * a)
    if not is_a_array and combinations:
        wavebank = wavebank[:, 0]
    if not is_gamma_array or (not is_a_array and not combinations):
        wavebank = wavebank[0]
    return wavebank

class WaveBank:
    """Class for handling multiple wavelets and making all transforms at once"""
    def __init__(self, shape, T, gamma, wavelet, combinations):
        assert (isinstance(shape, tuple)
           and isinstance(wavelet, str) or isinstance(wavelet, dict)
           and (isinstance(T, np.ndarray) or isinstance(T, float))
           and (isinstance(gamma, np.ndarray) or isinstance(gamma, float))
           and isinstance(combinations, bool)
            ), "Input types not correct"
        inv_omega = T / 2 / np.pi
        if wavelet == "morl":
            wavefun = _fmorl2d
            a = _eta_morl * inv_omega
        elif wavelet == "mexh":
            wavefun = _fsemimexh2d
            a = _eta_mexh * inv_omega
        elif isinstance(wavelet, dict):
            wavefun = wavelet['fun']
            a = wavelet['eta'] * inv_omega
        else:
            raise ValueError("Wavelet " + wavelet + " not implemented")
        self.__shape = shape
        self.__a = a
        self.__fwave = np.conj(_create_wavebank(shape, wavefun, a, gamma, combinations))

    def transform(self, signal):
        assert signal.shape == self.__shape, "signal must have same shape as wave bank"
        wb_shape = self.__fwave.shape
        F = fft.fft2(signal)
        if len(wb_shape) == 2:
            return self.__a * fft.ifft2(F * self.__fwave)
        elif len(wb_shape) == 3:
            a = self.__a.reshape((wb_shape[0], 1, 1))
            F = np.tile(F, (a.size, 1, 1))
        else:
            a = self.__a.reshape((1, wb_shape[1], 1, 1))
            F = np.tile(F, (wb_shape[0], a.size, 1, 1))

        return a * fft.ifft2(F * self.__fwave, workers=-1)
