# Distant Viewing Toolkit (DVT): Cultural Analysis of Still and Moving Images

[![Build Status](https://travis-ci.org/distant-viewing/dvt.svg?branch=master)](https://travis-ci.org/distant-viewing/dvt)  [![ReadTheDocs](https://readthedocs.org/projects/dvt/badge/?version=latest)](https://readthedocs.org/projects/dvt/badge/?version=latest) [![codecov](https://codecov.io/gh/distant-viewing/dvt/branch/master/graph/badge.svg)](https://codecov.io/gh/distant-viewing/dvt)  [![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI version shields.io](https://img.shields.io/pypi/v/dvt.svg)](https://pypi.python.org/pypi/dvt/)

The Distant TV Toolkit is a python module designed to automatically extract
metadata features from a corpus of images. It was initially designed for moving
images but also includes useful features for working with collections of still
images. This file includes a quick tutorial for getting up and running. Further
examples are given as IPython notebooks in the examples directory.

------------------

## Design principles

- **open-source**: All required components are released under
an open source license, fully documented, and developed in a public space.
- **ready to use**: Useful as-is without further training or
parameter tuning.
- **extensible**: Easy to extend with new methods or tune to new datasets.

------------------


## Installation

We suggest installing dvt through PyPI by running the following:

```sh
pip install dvt
```

There are additional Python requirements that should be automatically installed
automatically through PyPI. Some of these take several minutes to compile from
source.

------------------

## Demo

The following commands assuming that you have installed the dvt toolkit, are
in a running Python environment, and have a video file [video-clip.mp4](https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4)
in your working directory. To start, load the following from the toolkit:

```python
from dvt.annotate.core import FrameProcessor, FrameInput
from dvt.annotate.diff import DiffAnnotator
from dvt.annotate.face import FaceAnnotator, FaceDetectDlib, FaceEmbedVgg2
from dvt.annotate.meta import MetaAnnotator
from dvt.aggregate import CutAggregator
```

Start by constructing a frame input object attached to the video file. The
bsize argument indicates that we will work with the video by looking through
batches of 128 frames.

```python
finput = FrameInput("video-clip.mp4", bsize=128)
```

Now, create a frame processor and add three annotators: one for metadata, one
for faces, and one for differences between successive frames. The face detector
take a long time to run, so we restrict it to running only every 64 frames. The
`quantiles` input to the `DiffAnnotator` indicates that we want to compute the
40th percentile in differences between frames.

```python
fpobj = FrameProcessor()
fpobj.load_annotator(DiffAnnotator(quantiles=[40]))
fpobj.load_annotator(MetaAnnotator())
fpobj.load_annotator(FaceAnnotator(
  detector=FaceDetectDlib(), freq=64
))
```

Now, we can run the pipeline of annotators over the input object. We will turn
on logging here to see the output as Python processes each annotator over a
batch of frames. The `max_batch` argument restricts the number of batches for
testing purposes; set to None (default) to process the entire video file.

```python
import logging

logging.basicConfig(level='INFO')

fpobj.process(finput, max_batch=2)
```
```
INFO:root:processed 00:00:00,000 to 00:00:08,541 with annotator: 'diff'
INFO:root:processed 00:00:00,000 to 00:00:08,541 with annotator: 'meta'
INFO:root:processed 00:00:00,000 to 00:00:08,541 with annotator: 'face'
INFO:root:processed 00:00:08,541 to 00:00:10,010 with annotator: 'diff'
INFO:root:processed 00:00:08,541 to 00:00:10,010 with annotator: 'meta'
INFO:root:processed 00:00:08,541 to 00:00:10,010 with annotator: 'face'
```

The output is now stored in the `fpobj` object. To access it, we call its
`collect` method. This method returns a custom object (`DictFrame`, an
extension of an ordered dictionary), but can be converted to a Pandas data
frame for ease of viewing the output.

```python
obj_diff = fpobj.collect('diff')
obj_meta = fpobj.collect('meta')
obj_face = fpobj.collect('face')
```

The metadata is not very exciting, but is useful for downstream tasks:

```python
obj_meta.todf()
```
```
    type           vname       fps  frames  height  width
0  video  video-clip.mp4  29.97003     379     480    708
```

The difference annotator indicates the differences between successive frames,
as well as information about the average value (brightness) of each frame.

```python
obj_diff.todf().head()
```
```
        h40           video  frame  avg_value  q40
0  0.004983  video-clip.mp4      0  65.614907  0.0
1  0.007768  video-clip.mp4      1  65.658474  0.0
2  0.011770  video-clip.mp4      2  65.595159  0.0
3  0.006944  video-clip.mp4      3  65.856210  0.0
4  0.011142  video-clip.mp4      4  66.038943  0.0
```



```python
cagg = CutAggregator(cut_vals={'q40': 3})
cagg.aggregate(obj_diff).todf()
```
```
   frame_start  frame_end           video
0            0         74  video-clip.mp4
1           75        154  video-clip.mp4
2          155        255  video-clip.mp4
```

```python
obj_face.todf().head()
```
```
```python
obj_diff.todf().head()
```
```
   right  top           video  left  frame  confidence  bottom
0    549  101  video-clip.mp4   451      0    1.048361     199
1    218  105  video-clip.mp4   136      0    1.014278     187
2    550  105  video-clip.mp4   468     64    1.045316     187
3    226  114  video-clip.mp4   144     64    1.020017     195
4    212   81  video-clip.mp4   173    128    1.050994     121
5    356   89  video-clip.mp4   317    128    0.910692     129
6    111  105  video-clip.mp4    63    128    0.746686     153
7    464  141  video-clip.mp4   425    128    0.259442     181
8    583   89  video-clip.mp4   501    192    1.055945     171
9    218   97  video-clip.mp4   136    192    1.019857     179
```

```


____
<a href="https://www.neh.gov/" title="National Endowment for the Humanities"><img align="left" src="images/neh_logo_horizlarge.jpg" alt="NEH" class="rpad" width="300px"></a> The Distant Viewing Toolkit is supported by the National Endowment for the Humanities through a Digital Humanities Advancement Grant.
