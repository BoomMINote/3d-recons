# Fringe Pattern to 3D python
This is a python package for applying phase demodulation and 3D reconstruction as post processing of Fringe Pattern (FP) images recorded using the Fringe Projection - Laser Induced Fluorescence (FP-LIF) technique.
The package has been developed at the division of Combustion Physics at Lund University and a more detailed explanation of the technique is found in the article cited below together with on our webpage [spray-imaging.com/fplif.html](https://spray-imaging.com/fp-lif.html).
For any questions and code errors please contact submit an issue at [gitlab](https://gitlab.com/roth.adrian/fp23dpy).

## Installation through pip
```
pip install fp23dpy
```

## Usage
Either 
```
python3 -m fp23dpy <FP-image>
```
or 
```
fp23d <FP-image>
```
can be used that will by take the input image of a fringe pattern (and optional segmentation and calibration files) and attempt to 3D reconstruct together with the result as a GL Transmission Format file `.glb` which can be imported into most 3D viewing software, for example [threejs-editor](https://threejs.org/editor/).
Other 3D file formats are supported with `--ext <extension>` flag such as `.stl` or `obj`, the package use [trimesh](https://github.com/mikedh/trimesh) so all export formats for that package are supported.
For more information on possible flags to the command file use `python -m pf23dpy -h`.

Examples of different 3D structures can be found in the `fp23dpy/examples` folder of the source code.
To print these examples' segmentation, calibration and simulated FP-image run the following from the examples folder of the repository,
```
python print_example_structures.py
```
Then to reconstruct for example a pending drop run,
```
python -m fp23dpy example_drop.png
```
open the written `reconstructed_example_drop.glb` in a 3D viewer such as blender.

### Calibration
A calibration file can be used for each FP-LIF image, if this is not provided the program will try to calibrate using the given image which is not a robust procedure.
The calibration file name is either `calibration_<FP-image-filename>.txt` (without image type extension) or `calibration.txt` where the second will be default for the whole directory.
The file should include a JSON format object with the following attributes:
```
{
    "T":     float,		 # describing the fringe pattern period length in pixels with an image recorded of a plane 3D object orthogonal to the fringe projection direction
    "gamma": float,		 # float describing the angle in degrees of the fringes in the image, 0 if vertical fringes and 90 if horizontal
    "theta": float,	 	 # the angle in degrees from the camera to the fringe projection direction (optional, will create wrong of output)
    "scale": float,  	 # scale of the image, number of pixels per meter (optional, will scale output to pixels otherwise)
    "phi":   float,	 	 # the rotation in degrees of the camera in spherical coordinates to the angle of the fringe pattern with a certain radius (optional)
    "Tlim":  list of floats  # suggestion of T limits to search within, will not always be respected (optional)
    "principal_point": list with length 2, [x_p, y_p]  # Connected to the principal point in camera calibration. This is usually where the center of the camera chip is but when assuming orthographic camera it can be anywhere on the image and descripes which point the theta and phi angles is rotating around (optional, center of image is used by default)
    "absolute_phase": list with length 3, [x_a, y_a, absolute_phase]  # To get real 3D values the absolute phase is required. By defining the absolute_phase at a single pixel x_a, y_a the pixels connected to this pixel will also have known absolute phase. This parameter can also be a list of multiple absolute_phase points for different areas in the mask. If you have multiple absolute_phase points in the same area, the last one will be used. If you do not know the absolute_phase and have a time series where the area once was connected to an absolute_phase area the flag --temporal-alignment tries to track the area and estimate the real 3D coordinates (currently requires some manual points to be given for each image. (optional, if no absolute phase for unmasked area then its min value is set to zero for the third dimension)
    "absolute_threeD": list with length 3, [x_a, y_a, absolute_threeD]  # Same as absolute_phase but here you can set the threeD values of pixels instead. If both absolute keys are found in the calibration this value will be used.
}
```
The script `fp23d calibrate <calibration-image>` can be used for easier estimation of `T` and `theta` from a calibration image.

### Segmentation
If only parts of the image has the required Fringe Pattern lines, which is the case example drop, a segmentation of the image should be produced.
The segmented file should have filename `segmentation_<FP-image-filename>` (with image type extension).
If a single segmentation file should be used for all FP images in the same directory the segmentation file can be called `segmentation.png`.
The file should have zero values for background pixels and non-zero for foreground pixels.


## Citation
[Adrian Roth, Elias Kristensson, and Edouard Berrocal, "Snapshot 3D reconstruction of liquid surfaces," Opt. Express 28, 17906-17922 (2020)](https://doi.org/10.1364/OE.392325)

### Bibtex
```
@article{Roth:2020,
author = {Adrian Roth and Elias Kristensson and Edouard Berrocal},
journal = {Opt. Express},
number = {12},
pages = {17906--17922},
publisher = {OSA},
title = {Snapshot 3D reconstruction of liquid surfaces},
volume = {28},
month = {Jun},
year = {2020},
url = {http://www.opticsexpress.org/abstract.cfm?URI=oe-28-12-17906},
doi = {10.1364/OE.392325},
}
```

### Open science framework
Cite the software project at [https://doi.org/10.17605/OSF.IO/68SDK](https://doi.org/10.17605/OSF.IO/68SDK)
