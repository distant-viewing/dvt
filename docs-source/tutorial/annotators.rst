
.. highlight:: python


Annotator Tutorial
#####################

This tutorial shows how to build an analysis pipeline using several of the
pre-built annotators available in the distant viewing toolkit.
The following commands assume that you have installed the dvt toolkit, are
in a running Python environment, and have a video file [video-clip.mp4](https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4)
in your working directory.

To start, load the following from the toolkit and turn on logging::

    from dvt.annotate.core import FrameProcessor, FrameInput
    from dvt.annotate.diff import DiffAnnotator
    from dvt.annotate.face import FaceAnnotator, FaceDetectDlib, FaceEmbedVgg2
    from dvt.annotate.meta import MetaAnnotator
    from dvt.annotate.png import PngAnnotator
    from dvt.aggregate.cut import CutAggregator

    import logging
    logging.basicConfig(level='DEBUG')

Start by constructing a frame input object attached to the video file. The
bsize argument indicates that we will work with the video by looking through
batches of 128 frames.::

    finput = FrameInput("video-clip.mp4", bsize=128)

Now, create a frame processor and add four annotators: (i) metadata, (ii) png
files, (iii) differences between successive frames, and (iv) faces. The
`quantiles` input to the `DiffAnnotator` indicates that we want to compute the
40th percentile in differences between frames. The face detector take a long time
to run when not on a GPU, so we restrict it to running only every 64 frames.::

    fpobj = FrameProcessor()
    fpobj.load_annotator(PngAnnotator(output_dir="video-clip-frames"))
    fpobj.load_annotator(MetaAnnotator())
    fpobj.load_annotator(DiffAnnotator(quantiles=[40]))
    fpobj.load_annotator(FaceAnnotator(detector=FaceDetectDlib(), freq=64))

Now, we can run the pipeline of annotators over the input object. We will turn
on logging here to see the output as Python processes each annotator over a
batch of frames. The `max_batch` argument restricts the number of batches for
testing purposes; set to None (default) to process the entire video file.

.. code-block:: python

    fpobj.process(finput, max_batch=2)

.. code-block:: python

    INFO:root:processed 00:00:00,000 to 00:00:08,541 with annotator: 'png'
    INFO:root:processed 00:00:00,000 to 00:00:08,541 with annotator: 'meta'
    INFO:root:processed 00:00:00,000 to 00:00:08,541 with annotator: 'diff'
    INFO:root:processed 00:00:00,000 to 00:00:08,541 with annotator: 'face'
    INFO:root:processed 00:00:08,541 to 00:00:10,010 with annotator: 'png'
    INFO:root:processed 00:00:08,541 to 00:00:10,010 with annotator: 'meta'
    INFO:root:processed 00:00:08,541 to 00:00:10,010 with annotator: 'diff'
    INFO:root:processed 00:00:08,541 to 00:00:10,010 with annotator: 'face'

The output is now stored in the `fpobj` object. To access it, we call its
`collect_all` method. This method returns a dictionary of custom objects
(`DictFrame`, an extension of an ordered dictionary). Each can be converted
to a Pandas data frame for ease of viewing the output or saving as a csv
file.::

    obj = fpobj.collect_all()

We will not look at each output type.

Metadata
--------------------

The metadata is not very exciting, but is useful for downstream tasks.::

    obj['meta'].todf()

       height       fps  width   type           vname  frames
    0     480  29.97003    708  video  video-clip.mp4     379

Png
--------------------

The png annotator does not return any data.::

    obj['png'].todf()

    Empty DataFrame
    Columns: []
    Index: []

Instead, its used for its side-effects. You will see that there are individual
frames from the video now saved in the directory "video-clip-frames".

Difference
--------------------


The difference annotator indicates the differences between successive frames,
as well as information about the average value (brightness) of each frame.::

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

If you look at the constructed frames in "video-clip-frames", you
should see that there are in fact breaks at frames 75 and 155.

Face
--------------------


The face annotator detects faces in the frames. We configured it to
only run every 64 frames, so there is only output in frames 0, 64,
128, and 192.::

    obj['face'].todf()

                video  top  bottom  left  right  frame  confidence
    0  video-clip.mp4  101     199   451    549      0    1.048361
    1  video-clip.mp4  105     187   136    218      0    1.014278
    2  video-clip.mp4  105     187   468    550     64    1.045316
    3  video-clip.mp4  114     195   144    226     64    1.020017
    4  video-clip.mp4   81     121   173    212    128    1.050994
    5  video-clip.mp4   89     129   317    356    128    0.910692
    6  video-clip.mp4  105     153    63    111    128    0.746686
    7  video-clip.mp4  141     181   425    464    128    0.259442
    8  video-clip.mp4   89     171   501    583    192    1.055945
    9  video-clip.mp4   97     179   136    218    192    1.019857

Notice that there are two faces in frame 0, 64, and 192 but four faces
detected in frame 128. In fact, all six of the main cast members are in
frame 128, but two are two small and obscured to be found by the dlib
algorithm.
