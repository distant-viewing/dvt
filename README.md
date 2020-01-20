
# Distant Viewing Toolkit for the Analysis of Visual Culture

[![Build Status](https://travis-ci.org/distant-viewing/dvt.svg?branch=master)](https://travis-ci.org/distant-viewing/dvt)  [![codecov](https://codecov.io/gh/distant-viewing/dvt/branch/master/graph/badge.svg)](https://codecov.io/gh/distant-viewing/dvt)  [![License: GPL v2](https://img.shields.io/pypi/l/dvt?color=blue)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI version shields.io](https://img.shields.io/pypi/v/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI status shields.io](https://img.shields.io/pypi/status/dvt)](https://pypi.python.org/pypi/dvt/) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01800/status.svg)](https://doi.org/10.21105/joss.01800)


The Distant TV Toolkit is a Python package designed to facilitate the
computational analysis of visual culture. It contains low-level architecture
for applying state-of-the-art computer vision algorithms to still and moving
images. The higher-level functionality of the toolkit allows users to quickly
extract semantic metadata from digitized collections. Extracted information
can be visualized for search and discovery as well as aggregated and analyzed to find
patterns across a corpus.

Both the high-level command-line interface and the low-level object-oriented API
are introduced below. More information about the toolkit and project is available on the
following pages:

* Search and discovery interface example: [DVT Video Visualization](https://www.distantviewing.org/labs/)
* Example analysis using aggregated metadata: ["Visual Style in Two Network Era Sitcoms"](https://culturalanalytics.org/article/11045-visual-style-in-two-network-era-sitcoms)
* Theory of the project: ["Distant Viewing: Analyzing Large Visual Corpora."](https://www.distantviewing.org/pdf/distant-viewing.pdf)
* Project homepage: [The Distant Viewing Lab](https://distantviewing.org)
* Software Whitepaper: [A Python Package for the Analysis of Visual Culture](https://github.com/distant-viewing/dvt/blob/master/paper/paper.pdf)
* [Documentation](https://distant-viewing.github.io/dvt/)
* [PyPI repository](https://pypi.org/project/dvt/)

If you have any trouble using the toolkit, please open a
[GitHub issue](https://github.com/distant-viewing/dvt/issues). If you
have additional questions or are interested in collaborating, please contact
us at tarnold2@richmond.edu and ltilton@richmond.edu.

------------------

<br>

<a href="https://www.neh.gov/" title="National Endowment for the Humanities"><img align="left" src="docs-source/images/neh_logo_horizlarge.jpg" alt="NEH" class="rpad" width="300px"></a> The Distant Viewing Toolkit is supported by the National Endowment for the Humanities through a Digital Humanities Advancement Grant.

<br>

------------------

## Installation

The distant viewing toolkit requires Python 3.7. It is primarily built and
tested using Anaconda Python, which we suggest installing from here:
[Anaconda Distribution](https://www.anaconda.com/distribution/#download-section).
The package can then be installed through PyPI:

```
pip install dvt
```

Additional Python requirements should be installed automatically through PyPI.
It should also be possible to build from a clean virtual environment under
Python 3.7.

## High-level command-line usage: Minimal Demo

The following code assumes that you have installed the dvt toolkit and have
the video file
[video-clip.mp4](https://raw.githubusercontent.com/distant-viewing/dvt/master/tests/test-data/video-clip.mp4)
in your working directory. Run the following command to run the default
pipeline of annotators from dvt:

```
python3 -m dvt video-viz video-clip.mp4
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
video file) to finish. You can see an example of the toolkit's output on
several video files [here](https://www.distantviewing.org/labs/).

## Low-level API: Getting started with the Python interface

The command line tools provide a fast way to get started with the toolkit,
and there is much more functionality available when using the full Python
API provided by the module. The following code assumes that you have installed
the dvt toolkit and have the video file
[video-clip.mp4](https://raw.githubusercontent.com/distant-viewing/dvt/master/tests/test-data/video-clip.mp4),
audio file [video-clip.wav](https://raw.githubusercontent.com/distant-viewing/dvt/master/tests/test-data/video-clip.wav),
and subtitle file [video-clip.srt](https://raw.githubusercontent.com/distant-viewing/dvt/master/tests/test-data/video-clip.srt)
in your working directory.

Using the distant viewing toolkit starts by constructing a `DataExtraction`
object that is associated with input data (either a video file or a
collection of still images). Algorithms are then applied to the
`DataExtraction`; the results are stored as Pandas DataFrames and can be
exported as CSV or JSON files. There are two distinct types of algorithms:

- **annotators**: algorithms that work directly with the visual data source
but are able to only work with a small subset of frames or still images
- **aggregators**: algorithms that have access to information extracted
from previously run annotators across across the entire input, but cannot
direclty access the visual data

The separation of algorithms into these two parts makes it easier to write
straightforward, error-free code. It closely mirrors our
[theory of distant viewing](https://www.distantviewing.org/pdf/distant-viewing.pdf):

> Distant viewing is distinguished from other approaches by making explicit
> the interpretive nature of extracting semantic metadata from images.
> In other words, one must 'view' visual materials before studying them.
> Viewing,  which  we  define  as an interpretive action taken by either a
> person or a model, is necessitated by  the  way  in  which  information  is
> transferred  in visual materials. Therefore, in order to view images
> computationally,  a  representation  of  elements  contained within the
> visual material—a code system in semiotics  or,  similarly,  a  metadata
> schema  in  informatics—must  be  constructed.  Algorithms  capable  of
> automatically  converting  raw  images  into the  established  representation
> are  then  needed  to apply  the  approach  at  scale.

The annotator algorithms conduct the process of 'viewing' the material whereas
the aggregator algorithms perform a 'distant' (e.g., separated from the raw
materials) analysis of the visual inputs.

Here is an example showing the usage of these elements to detect shot breaks
in a video input. We start by running an annotator that detects the differences
between subsequent shots and then apply the cut aggregator to determine where
the changes indicate a pattern consistent with a shot break. As in the Minimal
Demo, the code assumes that the video file
[video-clip.mp4](https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4/)
is in your working directory:

```
from dvt.core import DataExtraction, FrameInput
from dvt.annotate.diff import DiffAnnotator
from dvt.aggregate.cut import CutAggregator

dextra = DataExtraction(FrameInput(input_path="video-clip.mp4"))
dextra.run_annotators([DiffAnnotator(quantiles=[40])])
dextra.run_aggregator(CutAggregator(cut_vals={'q40': 3}))
```

Looking at the output data, we see that there are four detected shots in the
video file:

```
dextra.get_data()['cut']
```

And its output:

```
frame_start  frame_end
0            0         74
1           75        154
2          155        299
3          300        511
```

There are many annotators and aggregators currently available in the toolkit.
Pipelines as well as pre-bundled sequences of annotators and aggregators are also
included in the package. Currently available implementations in the toolkit
are:

| Annotators                    | Aggregators           | Pipelines         |
| ----------------------------- |---------------------- | ----------------- |
| CIElabAnnotator               | CutAggregator         | ImageVizPipeline  |
| DiffAnnotator                 | DisplayAggregator     | VideoCsvPipeline  |
| EmbedAnnotator                | ShotLengthAggregator  | VideoVizPipeline  |
| FaceAnnotator                 | PeopleAggregator      |                   |
| HOFMAnnotator                 |                       |                   |
| ObjectAnnotator               |                       |                   |
| OpticalFlowAnnotator          |                       |                   |
| PngAnnotator                  |                       |                   |
| AudioAnnotator<sup>*</sup>    |                       |                   |
| SubtitleAnnotator<sup>*</sup> |                       |                   |

Details of these implementations can be found in the full
[API documentation](https://distant-viewing.github.io/dvt/). Additionally, it
is possible to construct your own Annotator and Aggregator objects. Details
are available in [this tutorial](https://distant-viewing.github.io/dvt/tutorial/custom.html).
If you develop an object that you think
may be useful to others, consider [contributing](#contributing) your code to
the toolkit.

(*) **Special Annotators**: The audio and subtitle annotators have a special format
because they require additional inputs (metadata about the video file and
access to the raw audio and subtitle data, respectively). To use these
annotators, specify the location of the audio and/or subtitle inputs when
creating the DataExtraction object. Next, run the corresponding DataExtraction
methods, as follows:

```
from dvt.core import DataExtraction, FrameInput

dextra = DataExtraction(
  vinput=FrameInput(input_path="video-clip.mp4"),
  ainput="video-clip.wav",
  sinput="video-clip.srt"
)
dextra.run_audio_annotator()
dextra.run_subtitle_annotator()
```

The audio data will be stored in an annotation named "audio":

```
dextra.get_data()['audio'].head()
```
```
   data  data_left  data_right
0 -1108      -1113       -1103
1 -1546      -1552       -1539
2 -1123      -1121       -1125
3 -1007      -1008       -1006
4 -1258      -1262       -1253
```

And subtitle information will be stored in an annotation named "subtitle":

```
dextra.get_data()['subtitle']
```
```
time_start  time_stop                                            caption  \
0       0.585      2.588              You guy's are messing with me, right?
1       5.145      7.100                                               Yeah
2       8.224      9.789  That was a good one. For a second there, I was...

frame_start  frame_stop
0           17          78
1          154         213
2          246         294
```

Aggregators can be written that further analyze the audio and subtitle data.
These aggregators are written no differently than aggregators that work with
just the video data; in fact, we can build aggregators that put together audio,
textual, and visual information. The pipeline does include several common
audio-based annotators. For example, the PowerToneAggregator calculates the
RMS of the audio power across pre-defined blocks of the audio source (indexed
by frame):

```
from dvt.aggregate.audio import PowerToneAggregator

dextra.run_aggregator(PowerToneAggregator(breaks=[0, 50, 200, 220]))
dextra.get_data()['power']
```
```
   frame_start  frame_end          rms
0            0         50   917.916747
1           50        200  1208.046528
2          200        220   997.259411
```

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

## Testing

The distant viewing toolkit includes a full test suite, based on the python
modules pytest. To run the tests clone the repository and run the following
from a terminal. These commands install pytest and other optional dependencies,
run the pytest unit tests, and display the coverage statistics:

```
pip install -U pytest
pip install .\[tests,optional\]
cd tests; pytest --disable-warnings --cov=dvt --cov-report term-missing .
```

To properly test the entire toolkit, we need to load and run several
configurations of large neural network] models. Therefore, testing can take
upwards of 20 minutes to complete.

## Contributing

Contributions, including bug fixes and new features, to the toolkit are
welcome. When contributing to this repository, please first discuss the change
you wish to make via a GitHub issue or email with the maintainers of this
repository before making a change. Small bug fixes can be given directly
as pull requests.

Please note that the project has a
[code of conduct](https://github.com/distant-viewing/dvt/blob/master/.github/CODE_OF_CONDUCT.md).
Contributors are expected to follow the guidelines.
