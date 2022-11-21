## Overview

The Distant Viewing Toolkit is a Python package released under an open-source
license. It is free to download and install, either from source through this
GitHub repository or as pre-built binaries through PyPI. We test and build the
toolkit using Python 3.10 on macOS and a Ubuntu linux distribution. It is likely
that the core functions will also work on any recent Python version.

Note that this software is intended to be run in a terminal and requires
some knowledge of how to run and structure code. If these are new to you, we
suggest first checking out a resource such as those provided by the
[Software Carpentry](https://software-carpentry.org/lessons/) project.
We are in the process of building a pre-built GUI to allow for a greater
range of users. Please be in touch if you are interested in learning more about
this proecss.

## Distant Viewing Toolkit

The toolkit requires that you have a recent version of Python (we tested it 
on 3.10.8); you'll also need to install a few dependecies. The easiest way
to do this is by using a package manager. For example, with pip you would
run the following in a terminal:

```
pip install opencv-python
pip install torch torchvision torchaudio
pip install git+https://github.com/distant-viewing/dvt.git@main
```

This process should install all of the required dependencies and allow you
to use all of the functions provided in the toolkit.