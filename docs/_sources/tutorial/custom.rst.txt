
.. highlight:: python

Custom Annotators and Aggregators
###################################

The Distant Viewing Toolkit provides several annotators and aggregators that
we think will be useful to many users. Many analyses, however, will require
additional functionality or might be interested in implementing novel
bleeding-edge techniques. This short tutorial demonstrates how to build a
custom annotator and aggregator.

In order to run the following code, download the video file at:
https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4

In order to build custom annotators and aggregators, we need to load the
abstract classes that will be extended. We will also load a data extraction
object to test the code with::

    from dvt.core import DataExtraction, FrameInput
    from dvt.abstract import FrameAnnotator, Aggregator

As a straightforward example of a custom annotator, we will extract the
red, green, and blue intensities from one pixel in the image. The specific
pixel will be determined by arguments passed to our annotator::

    class DemoAnnotator(FrameAnnotator):

        name = 'demo_anno'

        def __init__(self, **kwargs):
            self.row = kwargs.get('row')
            self.col = kwargs.get('col')

        def annotate(self, batch):
            img = batch.get_batch()
            fnames = batch.get_frame_names()

            output = {'frame': fnames,
                      'red': img[:, self.row, self.col, 0],
                      'green': img[:, self.row, self.col, 1],
                      'blue': img[:, self.row, self.col, 2]}

            return output

Running this gives::

    dextra = DataExtraction(FrameInput(input_path="video-clip.mp4"))
    dextra.run_annotators([DemoAnnotator(row=0, col=0)])

    dextra.get_data()['demo_anno'].head()

With output of::

       frame  red  green  blue
    0      0    6      1     0
    1      1    6      1     0
    2      2    6      1     0
    3      3    6      1     0
    4      4    4      2     0

Our custom aggregator will then simply take the average of the intensities
across all of the frames::

    class DemoAggregator(Aggregator):

        name = 'demo_agg'

        def aggregate(self, ldframe, **kwargs):
            dframe = ldframe['demo_anno']

            output = {'red': dframe["red"].mean(),
                      'green': dframe["green"].mean(),
                      'blue': dframe["blue"].mean()}

            return output

And running this gives an output with three columns::

    dextra.run_aggregator(DemoAggregator())

    dextra.get_data()['demo_agg']

And the output is::

             red      green       blue
    0  15.451172  11.306641  12.529297

For further examples of constructing annotators and aggregators, see the
source code the objects pre-constructed in the Distant Viewing Toolkit
Python package.
