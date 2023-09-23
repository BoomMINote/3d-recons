"""
The Fringe Pattern to 3D python package
See README.md file or https://spray-imaging.com/fp-lif.html for more information
"""

from .version import __version__

from .calibration import Calibration
from .roi import Roi

from .fp23dpy import fp23d, phase_to_threeD_const, threeD_to_phase_const
from .demodulation import demodulate
from .wavelets import cwt2
from .export import export3D
from . import frequency_peak_finder
from .helpers import make_carrier, estimate_square_carrier
from .simulation import render, render_from_map
from .segmentation import simple_segmentation as segment

