from setuptools import setup
from setuptools import find_packages

long_description= """
The Distant TV Toolkit is a Python package designed to facilitate the
computational analysis of visual culture. It contains low-level architecture
for applying state-of-the-art computer vision algorithms to still and moving
images. The higher-level functionality of the toolkit allows users to quickly
extract semantic metadata from digitized collections. Extracted information
can be visualized for search and discovery or aggregated and analyzed to find
patterns across a corpus.

More information about the toolkit and project is available on the following
pages:

- Search and discovery interface example: https://www.distantviewing.org/labs/
- Example analysis using aggregated metadata: https://doi.org/10.22148/16.043
- Theory of the project: https://www.distantviewing.org/pdf/distant-viewing.pdf
- Project homepage: https://distantviewing.org
- Software Whitepaper: https://github.com/distant-viewing/dvt/blob/master/paper/paper.pdf
- Documentation: https://distant-viewing.github.io/dvt/
- Source code: https://github.com/distant-viewing/dvt

The Distant Viewing Toolkit is supported by the National Endowment for the
Humanities through a Digital Humanities Advancement Grant. It is released under
the open-source GNU General Public License (GPLv2+).
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
    "mtcnn",
    "matplotlib",
    "scipy",
    "progress"
]

extras = {
    "tests": [
        "pytest",
        "pytest-pep8",
        "pytest-xdist",
        "pytest-cov",
        "codecov"
    ],
    "optional": ["scikit-image"]
}

setup(
    name="dvt",
    version="0.3.0",
    description="Cultural Analysis of Moving Images",
    long_description=long_description,
    author="Taylor Anold, Lauren Tilton",
    author_email="taylor.arnold@acm.org",
    url="https://github.com/distant-viewing/dvt",
    license="GPL-2",
    install_requires=required,
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or "
        "later (LGPLv2+)",
        "Programming Language :: Python :: 3.7",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_data={
          'dvt': ['data/build.zip'],
     },
    packages=find_packages(),
)
