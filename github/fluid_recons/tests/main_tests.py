""" Tests that write files to disk to see if the __main__ function is working properly """
import numpy as np
from skimage import io
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path

from fp23dpy.examples import example_cone

import show3d



def run_main(signals, segmentations, calibrations, command_flags="", with_show3d=False):
    signal = signals
    segmentation = segmentations
    calibration = calibrations
    base = "test_signal"
    name = f"{base}.png"
    io.imsave(name, signal.astype(np.uint8))
    io.imsave(f"segmented_{name}", segmentation.astype(np.uint8), check_contrast=False)
    calibration.write("calibration.txt")
    glb_file = f"reconstructed_{base}.glb"
    numpy_file = f"reconstructed_{base}.npy"
    try:
        command = f"python -m fp23dpy {name} --print-npy {command_flags}"
        print(command)
        results = subprocess.run(command, shell=True)
        if results.returncode != 0:
            raise ValueError(f"Something went wrong in the reconstruction, returncode {results.returncode}")
        grid3d = np.load(numpy_file)
        grid3d = np.ma.array(grid3d, mask=np.isnan(grid3d))

        if with_show3d:
            show3d.show(glb_file)
    finally:
        rm_files = [name, f"segmented_{name}", "calibration.txt", glb_file, numpy_file]
        for f in rm_files:
            f = Path(f)
            if f.is_file():
                f.unlink()

    return grid3d

def test_simple():
    calibration = example_cone.get_calibration()
    grid = example_cone.get_projected_coordinate_grid()
    signal = 255 / 2 * example_cone.render()
    signal.data[signal.mask] = 0

    segmentation = (signal.mask == False).astype(np.uint8) * 255
    signal = signal.data

    reconstructed_grid = run_main(signal, segmentation, calibration, with_show3d=True)
    # for i in range(3):
    #     plt.figure()
    #     plt.imshow(reconstructed_grid[i] - grid[i])
    #     plt.colorbar()
    # plt.show()
    rmse = np.sqrt(np.mean(np.square((reconstructed_grid - grid) / (np.max(grid) - np.min(grid)))))
    print("rmse {}, {}".format(example_cone.name, rmse))
    assert rmse < example_cone.max_rmse, "Expected rmse below {}, got {}".format(
        example_cone.max_rmse, rmse
        )

# def test_temporal_alignment():
#     calibration = example_cone.get_calibration()
#     grid = module.get_projected_coordinate_grid()
#     signal = 255 / 2 * module.render()
#     signal.data[signal.mask] = 0
# 
#     if not with_mask:
#         signal = signal.data
#     if with_square_carrier:
#         calibration["T"] = [0, 2 * np.pi / calibration["T"], 0, 0, 0]
# 
#     reconstructed_grid = fp23d(signal, calibration)
#     # for i in range(3):
#     #     plt.figure()
#     #     plt.imshow(reconstructed_grid[i] - grid[i])
#     #     plt.colorbar()
#     # plt.show()
#     rmse = rms(reconstructed_grid - grid) / (np.max(grid) - np.min(grid))
#     print("rmse {}, {}".format(module.name, rmse))
#     assert rmse < module.max_rmse, "Expected rmse below {}, got {}".format(
#         module.max_rmse, rmse

# def test_interpolation():
#     calibration = example_cone.get_calibration()
#     grid = example_cone.get_projected_coordinate_grid()
#     signal = 255 / 2 * example_cone.render()
#     signal.data[signal.mask] = 0
# 
#     segmentation = (signal.mask == False).astype(np.uint8) * 255
#     half_shape = np.array(segmentation.shape) // 2
#     segmentation[half_shape[0] - 30: half_shape[0] + 30, half_shape[1] - 29: half_shape[1] + 17] = 255 // 2
# 
#     signal = signal.data
# 
#     reconstructed_grid = run_main(signal, segmentation, calibration, with_show3d=False)
#     for i in range(3):
#         plt.figure()
#         plt.imshow(reconstructed_grid[i] - grid[i])
#         plt.colorbar()
#     plt.show()
#     # rmse = rms(reconstructed_grid - grid) / (np.max(grid) - np.min(grid))
#     # print("rmse {}, {}".format(module.name, rmse))
#     # assert rmse < example_cone.max_rmse, "Expected rmse below {}, got {}".format(
#     #     module.max_rmse, rmse
