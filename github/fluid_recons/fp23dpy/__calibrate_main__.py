"""
Script for helping calibration of an image
the calibration can either be manual or automatic
"""
import argparse
import os.path as osp
from skimage import io

from . import calibration as cb

def calibration_main():
    parser = argparse.ArgumentParser(description='Calibrate an image or data requisition for 3D reconstruction of Fringe Patterns using phase demodulation, for manual calibration remember to measure over 5 periods to get it correct')
    parser.add_argument('files', type=str, nargs='+',
                        help='Input image files to calibrate, image name with calibrat in it will be a global calibration for a directory')
    parser.add_argument('--auto', '-a', action='store_true',
                        help='Automatic calibration')
    parser.add_argument('--print', '-p', action='store_true',
                        help='Print the result to stdout')
    parser.add_argument('--symmetric', action='store_true',
                        help='Assume radial symmetric 3D object (only used for manual calibration)')
    args = parser.parse_args()

    to_calibrate = args.files
    if osp.isdir(to_calibrate[0]):
        output_dir = to_calibrate[0]
        cal_path_png = osp.join(output_dir, 'calibration_image.png')
        cal_path_tif = osp.join(output_dir, 'calibration_image.tif')
        if osp.isfile(cal_path_png):
            cal_path = cal_path_png
        elif osp.isfile(cal_path_tif):
            cal_path = cal_path_tif
        else:
            exit('Image "calibration_image.png/.tif" not found in directory')
        to_calibrate = [cal_path]
    elif 'calibrat' in to_calibrate[0]:
        output_dir = osp.dirname(to_calibrate[0])
        to_calibrate = [to_calibrate[0]]

    to_calibrate = [f for f in to_calibrate if not 'segmented' in f]
    to_calibrate = [f for f in to_calibrate if not 'reconstructed' in f]

    for image_path in to_calibrate:
        output_dir, file_name = osp.split(image_path)
        file_name, _ = osp.splitext(file_name)
        if 'calibrat' in file_name:
            calibration_path = osp.join(output_dir, 'calibration.txt')
        else:
            calibration_path = osp.join(output_dir, 'calibration_' + file_name + '.txt')

        im = io.imread(image_path, as_gray=True)

        if args.auto:
            calibration = cb.automatic(im, calibration_path)
        else:
            calibration = cb.manual(im, calibration_path, args.symmetric)
        
        if calibration is None:
            continue
        if args.print:
            print(calibration)
        else:
            calibration.write(calibration_path)

if __name__ == '__main__':
    calibration_main()
