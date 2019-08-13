
.. highlight:: python

Building a data extraction pipeline
###################################

The command line interface shown in the Minimal Demo provides a quick and
relatively straightforward way of using the Distant Viewing Toolkit. Assuming
there are no installation issues, users need only a minimal understanding of
the command line in order to apply the toolkit to produce a useful
visualization of their data. This tutorial shows how to make use of the
lower-level functions in the toolkit. It assumes that readers are generally
familiar with running Python code.

In order to run the following code, download the video file at:
https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4

The first step in using the internal functions of the toolkit is the
construction of a DataExtraction object. This object is tied to some kind of
visual input, usually a single video file or a collection of static
images. This is done with the following::

    from dvt.core import DataExtraction, FrameInput

    dextra = DataExtraction(FrameInput(input_path="video-clip.mp4"))

Then, we can run a collection of annotators over the input objects. Annotators
provide a way of extracting information about a local collection of frames.
To run these over a data extraction object, simply pass a list of annotators
to the run_annotators method of the DataExtraction object::

    from dvt.annotate.diff import DiffAnnotator

    dextra.run_annotators([DiffAnnotator(quantiles=[40])])

Notice that we can pass optional parameters to the annotation object, here the
quantiles to extract, as the class is being created. We can access the data
extracted from the material by calling the get_data method. It returns a
dictionary with a key for each annotator or aggregator that has been run. The
values are all pandas data frames. Here is the difference annotator::

    dextra.get_data()['diff'].head()

    frame  avg_value  q40       h40
    0      0  65.614907  0.0  0.004983
    1      1  65.658474  0.0  0.007768
    2      2  65.595159  0.0  0.011770
    3      3  65.856210  0.0  0.006944
    4      4  66.038943  0.0  0.011142

The difference annotator finds how much of a change there is between subsequent
frames in a video file. Looking at this across an entire video file can show
us where the shot breaks occur. To do this we use an aggregator. An aggregator
is able to look at the entire input at once, but must operate only on the
metadata, not the visual input.

Here is an example of how to call the CutAggregator from the DataExtraction
object. We set the cutoff for the measurement q40 to 3 (see the CutAggregator
documentation for a description of how this value is used in the algorithm)::

    from dvt.aggregate.cut import CutAggregator

    dextra.run_aggregator(CutAggregator(cut_vals={'q40': 3}))

The output is now also included in the data held within the DataExtraction
object. To see it, call the get_data option once again::

    dextra.get_data()['cut']

    frame_start  frame_end
    0            0         74
    1           75        154
    2          155        299
    3          300        511

There are many more aggregators and annotators available in the toolkit. It
is also possible to create your own aggregators and annotators. See the
following tutorials and the complete Distant Viewing API for further
information.
