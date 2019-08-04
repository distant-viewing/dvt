
# Distant Viewing Toolkit for the Analysis of Visual Culture

[![Build Status](https://travis-ci.org/distant-viewing/dvt.svg?branch=master)](https://travis-ci.org/distant-viewing/dvt)  [![codecov](https://codecov.io/gh/distant-viewing/dvt/branch/master/graph/badge.svg)](https://codecov.io/gh/distant-viewing/dvt)  [![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI version shields.io](https://img.shields.io/pypi/v/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI status shields.io](https://img.shields.io/pypi/status/dvt)](https://pypi.python.org/pypi/dvt/)

The Distant TV Toolkit is a Python module designed to facilitate the
computational analysis of visual culture. It contains low-level architecture
for applying state-of-the-art computer vision algorithms to still and moving
images. The higher-level functionality of the toolkit allows users to quickly
extract semantic metadata from digitized collections. Extracted information
can be visualized for search and discovery or aggregated and analyzed to find
patterns across a corpus.

More information about the toolkit and project is available on the following
pages:

* Search and discovery interface example: [DVT Video Visualization](https://www.distantviewing.org/labs/)
* Example analysis using aggregated metadata: ["Visual Style in Two Network Era Sitcoms"](https://doi.org/10.22148/16.043)
* Theory of the project: ["Distant Viewing: Analyzing Large Visual Corpora."](https://www.distantviewing.org/pdf/distant-viewing.pdf)
* Project homepage: [The Distant Viewing Lab](https://distantviewing.org)
* [Documentation](https://distant-viewing.github.io/dvt/)
* [PyPI repository](https://pypi.org/project/dvt/)

If you have any trouble using the toolkit, please open a
[GitHub issue](https://github.com/distant-viewing/dvt/issues). If you
have further questions or are interested in collaborating, please contact
us at tarnold2@richmond.edu and ltilton@richmond.edu.

------------------

<br>

<a href="https://www.neh.gov/" title="National Endowment for the Humanities"><img align="left" src="images/neh_logo_horizlarge.jpg" alt="NEH" class="rpad" width="300px"></a> The Distant Viewing Toolkit is supported by the National Endowment for the Humanities through a Digital Humanities Advancement Grant.

<br>

------------------

## Installation

The distant viewing toolkit has been built and tested using Python 3.7. We suggest
installing the [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section).
The module can then be installed through PyPI:

```
pip install dvt
```

Additional Python requirements should be automatically installed automatically
through PyPI.

## Minimal Demo

The following demo assumes that you have installed the dvt toolkit and have
the video file
[video-clip.mp4](https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4/)
in your working directory.

Run the following command to run the default pipeline of annotators from the
distant viewing toolkit:

```
python3 -m dvt video-clip.mp4
```

This may take several minutes to complete. Some minimal logging information
should display the annotators progress in your terminal. Once finished,
you should have a new directory `dvt-output-data` that contains extracted
metadata and frames from the source material. You can view the extracted
information by starting a local http server:

```
python3 -m http.server --directory dvt-output-data
```

And opening the following: [http://0.0.0.0:8000/](http://0.0.0.0:8000/).

You can repeat the same process with your own video inputs, though keep in
mind that it may take some time (often several times the length of the input
video file) to finish. Further tutorials in the
[toolkit documentation](https://distant-viewing.github.io/dvt/) further explain
other command line options and additional approaches using the lower-level
architecture provided in the distant viewing toolkit.

## Citation

If you make use of the toolkit in your work, please cite the relevant papers
describing the tool and its application to the study of visual culture:

```
@article{,
  title   = "Distant Viewing: Analyzing Large Visual Corpora",
  author  = "Arnold, Taylor B and Tilton, Lauren",
  journal = "Digital Scholarship in the Humanities",
  year    = "2019",
  doi     = "10.1093/digitalsh/fqz013",
  url     = "http://dx.doi.org/10.1093/digitalsh/fqz013"
}
```

```
@article{,
  title   = "Visual Style in Two Network Era Sitcoms",
  author  = "Arnold, Taylor B and Tilton, Lauren and Berke, Annie",
  journal = "Cultural Analytics",
  year    = "2019",
  doi     = "10.22148/16.043",
  url     = "http://dx.doi.org/10.22148/16.043"
}
```

# Contributing

Contributions, including bug fixes and new features, to the toolkit are
welcome. When contributing to this repository, please first discuss the change
you wish to make via issue, email, or any other method with the owners of this
repository before making a change. Small bug fixes can be given directly
as pull requests.

Please note that the project has a
[code of conduct](https://github.com/distant-viewing/dvt/blob/master/.github/CODE_OF_CONDUCT.md).
Contributors are expected to follow the guidelines for all interactions with
the project.
