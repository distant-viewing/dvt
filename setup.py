from setuptools import setup
from setuptools import find_packages

setup(
    name="dvt",
    version="0.0.7",
    description="Cultural Analysis of Moving Images",
    long_description="Distant Viewing uses and develops computational"
    "techniques to analyse moving image culture on a large"
    "scale. The project is currently in active development.",
    author="Taylor Anold",
    author_email="taylor.arnold@acm.org",
    url="https://github.com/distant-viewing/dvt",
    license="GPL-2",
    install_requires=["numpy>=1.14.0", "keras>=2.1.4", "scipy>=1.0.0", "h5py>=2.7.1"],
    extras_require={"tests": ["pytest", "pytest-pep8", "pytest-xdist", "pytest-cov"]},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Programming Language :: Python :: 3.6",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
