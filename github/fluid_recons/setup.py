import setuptools
import os

# load __version__ without importing anything, inspired by https://github.com/mikedh/trimesh/
version_file = os.path.join(os.path.dirname(__file__), "fp23dpy/version.py")
with open(version_file, "r") as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split("=")[-1])

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fp23dpy",
    version=__version__,
    author="Adrian Roth",
    author_email="adrian.roth@forbrf.lth.se",
    description="Package for 3D reconstruction of Fringe Patterns captured using the Fringe Projection - Laser Induced Fluorescence technique.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/roth.adrian/fp23dpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "fp23d = fp23dpy.__main__:main",
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
        "trimesh",
        "tqdm",
    ],
    python_requires=">=3.6",
)
