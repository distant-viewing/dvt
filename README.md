# Distant Viewing Toolkit (DVT) for the Cultural Analysis of Moving Images

**Note:** *This project is currently being developed and
will be iteratively improved and streamlined (we plan for
a stable release in early 2019). Code is being made public along
the way to document the process and to encourage comments and
collaborations. Until the form of the toolkit settles down,
documentation and examples will be kept to minimum. Please
contact the maintainer (Taylor Arnold <taylor.arnold@acm.org>)
with any questions or concerns.*

The Distant TV Toolkit is designed to automatically extract
metadata features from a corpus of moving images. Our initial focus is on
features of interest to the study of television shows, but all of the
techniques should work on movies and other media formats.

To install the toolkit, you will first need to install Python 3.6 and the
OpenCV library. Then, clone the repository and run `setup.py`:

```
git clone https://github.com/statsmaths/dvt
cd dvt
python setup.py install
```

Then, in python, you can create annotated data from any movie file (here,
named `input.mp4`) by running the following code:

```
import dvt

vp = dvt.video.VideoProcessor()
vp.load_annotator(dvt.frame.DiffFrameAnnotator())
vp.load_annotator(dvt.frame.FlowFrameAnnotator())
vp.load_annotator(dvt.frame.TerminateFrameAnnotator())
vp.load_annotator(dvt.frame.ObjectCocoFrameAnnotator())
vp.load_annotator(dvt.frame.FaceFrameAnnotator())

vp.setup_input(video_path="input.mp4",
               output_path="frame_output.json")
vp.process()
```
This creates a file `frame_output.json` with extracted data based on the
selected annotators. To visualize the output, then run:

```
import dvt

vv = dvt.view.VideoViewer()
vv.setup_input(video_path="input.mp4",
               input_path="frame_output.json")
vv.run()
```

This will show the annotations over a running version of the video.

## Frame Annotators

The currently available annotators to process the raw video files are:

- `DiffFrameAnnotator`: determines how different one frame is compared to the
previous frame. Specifically, it down samples the frame to a 32x32 thumbnail
and finds quantiles of the differences in hue, saturation, and value.
- `FlowFrameAnnotator`: follows key-points (points at the corners of detected
edges) across frames using optical flow. Looking at them over time allows for
the analysis of  object and character movement.
- `ObjectCocoFrameAnnotator`: uses the YOLOv2 algorithm to detect and localize
80 classes of objects in the frame.
- `FaceFrameAnnotator`: detects, localizes, and computes 128-dimensional
embeddings of faces in the image. It uses a CNN model which is relatively
robust to scale and 3D-rotations.
- `HistogramFrameAnnotator`: compute histograms of hue, saturation, and values
of the image. Currently it is applied to each sector of a 3x3 grid over the image;
these can be aggregated to have a histogram of the entire image. Also estimates
and outputs the lumosity center of gravity, which can be useful for shot detection.
- `KerasFrameAnnotator`: an annotator that applies any image-based keras model and
output the numeric results. Save the model as an `.h5` model and load it into this
layer. Allows for the inclusion of a preprocessing function and will automatically
resize the image depending on the input shape given in the model.
- `TerminateFrameAnnotator`: a special annotator that can conditionally
terminate the processing of a given frame. Put this in the pipeline between
fast annotators (e.g., the diff and flow annotators) that need to run on every
frame and slower annotators (e.g., object detection) that can be selectively
applied to only a subset of the frames.
- `PngFrameAnnotator`: another special annotator that saves the current frame
as a PNG file. Pair with the Terminate annotator to only save some of the images.

Currently these annotators have many hard-coded tuning parameters. As the
toolkit is built out, we plan to document these and allow for tweaking these
at runtime. We also plan to include frame annotators to extract dominant
colors and to embed frames into a space useful for scene classification.

## Video Annotators

The next stage of development involves producing video-level annotators. These
will take the `frame_output.json` file as an input and produce higher-level
annotations of the video as a whole. Tasks include:

- shot segmentation and classification
- scene segmentation and clustering
- facial recognition
- camera movement classification
- speaker resolution

We have written the `dvt.view.VideoViewer` with an eye towards supporting the
visualization of video level annotations in addition to frame level data.

## Audio and Caption Annotators

The final plan for development of the Distant Viewing Toolkit is to extract
audio features and closed captioning data. These in turn will be feed back into
the video level annotations in order to learn higher order aspects of the
input media.

____
<a href="https://www.neh.gov/" title="National Endowment for the Humanities"><img align="left" src="images/neh_logo_horizlarge.jpg" alt="NEH" class="rpad" width="300px"></a> The Distant Viewing Toolkit is supported by the National Endowment for the Humanities through a Digital Humanities Advancement Grant.
