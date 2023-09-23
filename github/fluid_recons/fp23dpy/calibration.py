"""
Module for either manual of automatic calibration of an FP-LIF image (Fringe Pattern image)
The automatic algorithm is mostly found in the frequency_peak_finder module
"""
import numpy as np
import json

import os
import os.path as osp
from . import frequency_peak_finder as fpf

class Calibration(dict):
    """Calibration class for handling write and read of calibration files and manipulation of the values
       All angles are converted to radians when read from file 
    """
    def __init__(self, start=None, empty=False):
        if not empty:
            self['T'] = 0
            self['gamma'] = 0
        if not start is None:
            for key, value in start.items():
                self.set(key, value)

    allowed_keys = ['T', 'gamma', 'theta', 'Tstep', 'Tlim', 'phi', 'scale', 'scale_unit', 'absolute_phase', 'absolute_threeD', 'principal_point', 'area_tracks', 'image_shape']
    def _allowed_key(key):
        if key in Calibration.allowed_keys:
            return 1
        else:
            raise ValueError("Key {} not allowed in calibration. Possible keys are {}".format(key, Calibration.allowed_keys))

    angle_keys = ['gamma', 'theta', 'phi']
    def _scale_angles(self, factor):
        for key in Calibration.angle_keys:
            if key in self:
                self[key] *= factor

    def copy(self):
        return Calibration(super(Calibration, self).copy())

    def _to_radians(self):
        self._scale_angles(np.pi / 180)
        return self

    def _to_degrees(self):
        self._scale_angles(180 / np.pi)
        return self

    def get(self, key, default=None):
        """Specialized get function where default value is returned if key if allowed but not found""" 
        if key in self:
            return super(Calibration, self).__getitem__(key)
        elif Calibration._allowed_key(key) and not default is None and not default == False:
            return default
        else:
            raise AttributeError("Key {} not set in calibration".format(key))

    def set(self, key, value):
        """Specialized set function which only allow setting of certain keys"""
        Calibration._allowed_key(key) # checking if valid key
        super(Calibration, self).__setitem__(key, value)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, key):
        return self.get(key)

    def update(self, calibration):
        for key, value in calibration.items():
            self.__setitem__(key, value)

    def set_theta_direction(self, direction):
        if 'theta' in self:
            self.set('theta', direction * abs(self.get('theta')))

    def read(filepath, **kwargs):
        """Read a calibration file in JSON format from file, kwargs is sent to constructor of Calibration object"""
        with open(filepath) as f:
            content = f.readlines()
        calibration = Calibration(json.loads("".join(content)), **kwargs)
        calibration._to_radians()
        return calibration

    def write(self, filepath):
        """Write a calibration file in JSON format from file"""
        calibration = self.copy()._to_degrees()
        content = json.dumps(calibration, sort_keys=True,
                             indent=4, separators=(',', ': ')) + "\n"
        with open(filepath, 'w') as f:
            f.write(content)

    def calibrate(signal):
        """Automatic algorithm for calibration which use the frequency_peak_finder modules result"""
        signal = signal.astype(float)
        peak = fpf.find_from_signal(signal, 1)
        calibration = Calibration()
        calibration['T'] = float(peak[0, 0])
        calibration['gamma'] = float(peak[0, 1])
        # calibration['Tlim'] = [float(peak[0, 2]), float(peak[0, 3])]
        return calibration


def automatic(im, calibration_path):
    return Calibration.calibrate(im)

## check if visual plots can be produced, if not do not try to load matplotlib stuff
if os.environ.get('DISPLAY'):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    def manual_symmetric(im, calibration):
        if not 'theta' in calibration:
            print('Error: symmetric not possible if theta is not specified in the current calibration file')
            return {}

        print("Click on points most to the left and right of the structure")
        plt.figure()
        plt.imshow(im)
        points = np.array(plt.ginput(2, 0, True))
        plt.close()
        middle_x, middle_y = np.mean(points, axis=0)
        radius = middle_x - points[0, 0]
        principal_point = [middle_x, middle_y]
        absolute_phase = [int(round(middle_x - radius * np.sin(calibration['theta']))), int(round(middle_y)), 0]
        return { 'principal_point': principal_point, 'absolute_phase': absolute_phase }

    def print_help():
        print("Click on one of the bright lines in the figure.")
        print("Follow the fringe pattern for !5! periods and click again on the brightest part.")
        print("If you click in error or want to zoom use right click to undo last click.")
        print()


    def manual(im, calibration_path, symmetric=False):
        """
        To use for manual calibration of an image
        
        The image will be plotted in a matplotlib figure and you should click on one place in the figure and then follow the fringe pattern for 5 periods and click again. Right click is used to undo last click. A calibration dictionary is returned with the results
        """
        file_name, _ = osp.splitext(osp.basename(calibration_path))
        go = True
        if osp.isfile(calibration_path):
            string = input('Do you want to overwrite existing ' + calibration_path + ' [Yes/no]: ')
            go = True if string == '' or string.lower() == 'yes' else False
            calibration = Calibration.read(calibration_path)
        else:
            calibration = Calibration()
        if go:
            print_help()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            mpl = _ManualCalibration(ax)
            ax.imshow(im)
            ax.set_title(file_name.replace('_', '-'))
            plt.show()
            mpl.disconnect()
            T = mpl.T
            if T != 0:
                gamma = mpl.gamma
                calibration['T'] = float(T)
                calibration['gamma'] = float(gamma)
                calibration.set_theta_direction(mpl.d)
                return calibration

            if symmetric:
                calibration.update(manual_symmetric(im, calibration))

            return calibration


    class _ManualCalibration:
        """Class for handling the input in the manual calibration case"""
        def __init__(self, ax):
            self.ax = ax
            self.line = None
            self.orth_line = None
            self.T = 0
            self.gamma = 0
            self.d = 0
            self.cidpress = self.ax.figure.canvas.mpl_connect(
                'button_press_event', self.on_press)
            self.cidmotion = self.ax.figure.canvas.mpl_connect(
                'motion_notify_event', self.on_motion)

        def on_press(self, event):
            x, y = [int(event.xdata), int(event.ydata)]
            if event.button == 3:
                self.line.remove()
                self.orth_line.remove()
                del self.line
                del self.orth_line
                self.line = None
                self.orth_line = None
                self.ax.figure.canvas.draw()
            elif self.line is None:
                self.line = mlines.Line2D([x, x], [y, y], color='r')
                self.orth_line = mlines.Line2D([x, x], [y, y], color='r')
                self.ax.add_line(self.line)
                self.ax.add_line(self.orth_line)
                self.ax.figure.canvas.draw()
            else:
                nperiods = 5
                l = self.line.get_xydata()
                x0 = l[0, 0]; y0 = l[0, 1]
                self.T = np.sqrt((x - x0)**2 + (y - y0)**2) / nperiods
                self.gamma = np.arctan2(y - y0, x - x0)
                self.d = (self.gamma < 0) * 2 - 1
                self.gamma = self.gamma if self.gamma > 0 else self.gamma + np.pi
                plt.close(self.ax.figure)

        def on_motion(self, event):
            if self.line is None or event.inaxes is None: return
            l = self.line.get_xydata()
            x0 = l[0, 0]; y0 = l[0, 1]
            x1, y1 = [int(event.xdata), int(event.ydata)]
            self.line.set_xdata([x0, x1])
            self.line.set_ydata([y0, y1])

            orth_x1 = x0 + (y1 - y0)
            orth_y1 = y0 - (x1 - x0)
            self.orth_line.set_xdata([x0, orth_x1])
            self.orth_line.set_ydata([y0, orth_y1])
            self.ax.figure.canvas.draw()

        def disconnect(self):
            self.ax.figure.canvas.mpl_disconnect(self.cidpress)
            self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
else:
    manual = automatic
