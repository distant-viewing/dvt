
.. highlight:: python


Minimal Demo
#####################

Here we give a quick demo showing a very straightforward application of the
distant viewing toolkit. For a significantly more detailed and comprehensive
example of the toolkit, please see the tutorial.

The following demo assuming that you have installed the dvt toolkit, are
in a running Python environment, and have a video file
`video-clip.mp4 <https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4/>`_ in your working directory.

To start, load the following classes from the toolkit::

    from dvt.annotate.core import FrameProcessor, FrameInput
    from dvt.annotate.diff import DiffAnnotator
    from dvt.aggregate.cut import CutAggregator

We now construct a frame input object attached to the video file. The bsize
argument indicates that we will work with the video by looking through
batches of 128 frames.::

    finput = FrameInput("video-clip.mp4", bsize=128)

Create a frame processor and load the difference annotator, indicating
that we want to compute the 40th percentile in differences between subsequent
frames.::

    fpobj = FrameProcessor()
    fpobj.load_annotator(DiffAnnotator(quantiles=[40]))

With the pipeline created, we now run the annotators over the input object.
The `max_batch` argument restricts the number of batches for testing purposes;
set to None (default) to process the entire video file.::

    fpobj.process(finput, max_batch=2)

To access the output data, call the :code:`collect_all` method of the the
frame processor object.::

    obj = fpobj.collect_all()

Here is the raw output from the difference annotator, formatted as a Pandas
data frame for pretty printing.::

    obj['diff'].todf().head()

       q40       h40           video  frame  avg_value
    0  0.0  0.004983  video-clip.mp4      0  65.614907
    1  0.0  0.007768  video-clip.mp4      1  65.658474
    2  0.0  0.011770  video-clip.mp4      2  65.595159
    3  0.0  0.006944  video-clip.mp4      3  65.856210
    4  0.0  0.011142  video-clip.mp4      4  66.038943

What if we want to find video cuts using these values? In order to
aggregate the values into cuts, use the CutAggregator module. Here
we have configured it to break a cut whenever the `q40` key is at
least 3.::

    cagg = CutAggregator(cut_vals={'q40': 3})
    cagg.aggregate(obj).todf()

                video  frame_start  frame_end
    0  video-clip.mp4            0         74
    1  video-clip.mp4           75        154
    2  video-clip.mp4          155        255

The input video is stored as 29.97 frames per second. If you watch it, you
will in fact see cuts around 2.46 and 5.17 second into the clip.
