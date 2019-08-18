
.. highlight:: python

Command Line Options
###################################

The minimal demo shows how to use the command line interface to produce a
visualization of a video file. While the default options in the command line
tool often produce reasonable outputs, is can be useful to modify some of
these values. This tutorial introduces some of the most common options and
explains how to work with them. As with the minimal demo, the code here
assumes that you have installed the dvt toolkit and have
the video file
`video-clip.mp4 <https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4/>`_
in your working directory.

To run the command line tool with the default settings on the video-clip.mp4
file, one can execute the following code in a terminal::

    python3 -m dvt video-clip.mp4

Alternatively, all of the default values can be explicitly set using the
following::

    python3 -m dvt video-clip.mp4 \
        --dirout=dvt-output-data \
        --pipeline-level=2 \
        --diff-cutoff=10 \
        --cut-min-length=30 \
        --frequency=0

What does changing these options do? By default, the output data is places in
a directory called "dvt-output-data" inside of the current working directory.
You can change this in the call above to place the input anywhere on your
machine. The pipeline level determines how much output is constructed. Setting
this value to 0 produces only a JSON metadata file for the video clip;
making it 1 also creates the output images (frames, annotated frames, and
visualizations of the optical flow). Finally, setting it to 2 (the default)
also creates all of the extra files needed to run a local website to visualize
the results.

The last three default options above control how the cuts are determined from
the video file. Making the "diff-cutoff" lower causes the algorithm to be more
aggressive (it makes more cuts) when determining whether a cut occurs; higher
values produce fewer cuts. The minimum cut length determine the minimal length
in frames that a cut can occur for. Finally, the frequency value provides a
different way of determining which frames to annotate. When set to a positive
integer, the command line tool will forgo determining cut breaks and simply
extract one out of every "frequency" frames. The reason for including so many
options for cut detection is that, while the defaults work reasonably well for
recent, high-definition films and scripted television, they can be quite
unreliable when working with other sources. Manipulating the cut-off scores
(or, if that fails, just setting a frequency) allows users to still make use of
the command line interface.

One final option in the command line tool is to do facial recognition for known
people in the input video. To start, create a directory on your machine with
one image per person that you would like to detect. Name the files with the
desired name of the person; for example, if you want to detect images of
Oprah Winfrey, you may add an image titled "oprah-winfrey.png". Then, assuming
your images are in a file in the working directory called "face-images",
you can include these in the pipeline as follows::

    python3 -m dvt video-clip.mp4 \
        --dirout=dvt-output-data \
        --pipeline-level=2 \
        --diff-cutoff=10 \
        --cut-min-length=30 \
        --frequency=0 \
        --path-to-faces=face-images

Note that the annotation process will take slightly longer when detecting
faces, but you will (potentially) have more rich data included in the output.
