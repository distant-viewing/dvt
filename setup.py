from setuptools import setup
from setuptools import find_packages

long_description= """
The Distant TV Toolkit is a python module designed to automatically extract
metadata features from a corpus of images. It was initially designed for moving
images but also includes useful features for working with collections of still
images. This file includes a quick tutorial for getting up and running. Further
examples are given as IPython notebooks in the examples directory.
"""

required = [
    "numpy",
    "pandas",
    "keras",
    "tensorflow",
    "scipy",
    "h5py",
    "opencv-python",
    "keras_retinanet",
    "mtcnn"
]

extras = {
    "tests": ["pytest", "pytest-pep8", "pytest-xdist", "pytest-cov", "codecov"],
    "optional": ["pyrtools", "cmake", "dlib", "scikit-image"]
}

setup(
    name="dvt",
    version="0.0.7",
    description="Cultural Analysis of Moving Images",
    long_description=long_description,
    author="Taylor Anold, Lauren Tilton",
    author_email="taylor.arnold@acm.org",
    url="https://github.com/distant-viewing/dvt",
    license="GPL-2",
    install_requires=required,
    extras_require=extras,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2"
        "or later (GPLv2+)",
        "Programming Language :: Python :: 3.7",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
