
.. highlight:: python


Minimal Demo
#######################

Here we give a quick demo showing how to use the command-line interface in
order to visualize a video file using the distant viewing toolkit. For more
detailed and comprehensive examples of the toolkit, please see the tutorials.

The following demo assumes that you have installed the dvt toolkit and have
the video file
`video-clip.mp4 <https://github.com/distant-viewing/dvt/raw/master/tests/test-data/video-clip.mp4/>`_
in your working directory.

Run the following command to run the default pipeline of annotators from the
distant viewing toolkit::

    python3 -m dvt video-viz video-clip.mp4

This may take several minutes to complete. Some minimal logging information
should display the annotators progress in your terminal. Once finished,
you should have a new directory :code:`dvt-output-data` that contains extracted
metadata and frames from the source material. You can view the extracted
information by starting a local http server::

     python3 -m http.server --directory dvt-output-data

And opening the following: `http://0.0.0.0:8000/ <http://0.0.0.0:8000/>`_.

You can repeat the same process with your own video inputs, though keep in
mind that it may take some time (often several times the length of the input
video file) to finish. The following tutorials further explain other command
line options and additional approaches using the lower-level architecture
provided in the distant viewing toolkit.
