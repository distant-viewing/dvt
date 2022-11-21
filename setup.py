from setuptools import setup
from setuptools import find_packages

long_description= """
The Distant Viewing Toolkit is a Python package designed to facilitate the
computational analysis of visual culture. It contains low-level architecture
for applying state-of-the-art computer vision algorithms to still and moving
images. Extracted information can be visualized for search and discovery or
aggregated and analyzed to find patterns across a corpus.
"""

required = [
    "numpy",
    "pandas",
    "opencv-python",
    "torchvision"
]

setup(
    name="dvt",
    version="1.0.0",
    description="Computational Analysis of Visual Culture",
    long_description=long_description,
    author="Taylor Anold, Lauren Tilton",
    author_email="taylor@dvlab.io",
    url="https://github.com/distant-viewing/dvt",
    install_requires=required,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
