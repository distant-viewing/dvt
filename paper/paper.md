---
title: 'Distant Viewing Toolkit: A Python Package for the Analysis of Visual Culture'
tags:
  - Python
  - digital humanities
  - media studies
  - computational social science
  - time-based media
  - visual culture
  - computer vision
authors:
  - name: Taylor Arnold
    orcid: 0000-0003-0576-0669
    affiliation: 1
  - name: Lauren Tilton
    orcid: 0000-0003-4629-8888
    affiliation: 2
affiliations:
  - name: University of Richmond, Department of Mathematics and Computer Science
    index: 1
  - name: University of Richmond, Department of Rhetoric and Communication Studies
    index: 2
date: 4 August 2019
bibliography: paper.bib
---

# Summary

The Distant Viewing Toolkit is a Python package for the
computational analysis of visual culture. It addresses the challenges of
working with moving images through the automated
extraction and visualization of metadata summarizing the content
(e.g., people/actors, dialogue, scenes, objects) and style (e.g., shot angle,
shot length, lighting, framing, sound) of time-based
media. The software allows users to explore media
such as film and television with formats such as movies, serials, and
news broadcasts.  This toolkit is optimized for two purposes:
(1) scholarly inquiry of visual culture from the humanities and social sciences,
and (2) search and discovery of collections within galleries, libraries, archives, and museums (GLAM).


Many open-source projects provide implementations of state-of-the-art computer
vision algorithms. However, there are limited options for users looking to
quickly build end-to-end pipelines that link together common visual annotations.
Different algorithms require varying dependencies, different input formats, and
produce outputs using different schemas. Because of the rapid pace of
scholarship across the many sub-fields of computer vision, it can be
difficult to determine which algorithms to use as well as a significant amount of work
to manually test every option. These challenges are exacerbated
when working with moving images because most available computer vision
libraries take still images as inputs.   

The Distant Viewing Toolkit addresses these needs by
(1) constructing an object-oriented framework for applying a collection of
algorithms to moving images, (2) packaging together common sound and computer
vision algorithms in order to provide out-of-the-box functionality for common
tasks in the computational analysis of moving images,
and (3) allowing video files alongside still images as an input.
Currently provided algorithms include functionality for: shot detection [@pal2015video],
object detection [@li2018recurrent], face detection [@jiang2017face],
face identification [@cao2018vggface2], color analysis [@karasek2015color],
image similarity [@szegedy2017inception], optical flow [@farneback2003two], and
shot distance analysis [@butler2012television].  

The Distant Viewing Toolkit provides two interfaces.
The Low-level Python API, a direct Python API, provides
for customized and advanced processing of visual data.  
The High-level Command Line Interface is designed to be accessible to users with limited
programming experience. Metadata produced by either interface can also be further
aggregated and analyzed to find patterns across a corpus.
Drawing on theories of exploratory data analysis,  the package includes a
custom JavaScript visualization engine that can be run on a user's machine to
visualize the metadata for search and discovery. Together, these provide tools for
the increasingly popular application of computational methods to the study of visual
culture [@wevers2019visual].  The following sections provide further explaination
of the interfaces and visualization engine followed by a section on the process
and development of the package. Detailed documentation and tutorials are
provided in the package's documentation:
[https://distant-viewing.github.io/dvt/](https://distant-viewing.github.io/dvt/).

![Example index page from the Distant Viewing Toolkit's command line video
visualization. Clicking on an image opens a new page to display the extracted
data from each individual input.](./img/dvt-main-view.png){ width=13.5cm }

![Example video page from the Distant Viewing Toolkit's command line
video visualization from the South Korean movie *The Contact* (1997).
Three shots are shown; scrolling up and down shows the other
detected shots. For each shot, an annotated thumbnail, optical flow, audio tone,
and spectrogram are shown along with metadata such as the start and stop time,
number of detected faces, and the estimated shot length. Clicking on any image
opens a pop-over div with a larger version of the image. The original image can
be shown by clicking on the blue button marked
"Original Image".](./img/dvt-video-view.png){ width=13.5cm }

# Low-level Python API

The full functionality of the Distant Viewing Toolkit is available when using the full Python API.
The toolkit starts by constructing a `DataExtraction`
object that is associated with input data (either a video file or a
collection of still images). Algorithms are then applied to the extraction
object, with results stored as Pandas DataFrames that can be exported as
CSV or JSON files. There are two distinct types of algorithms:

- **annotators**: algorithms that work directly with the source data
but are able to only work with a small subset of frames or still images
- **aggregators**: algorithms that have access to information extracted
from all previously run annotators and aggregators across across the entire
input, but cannot directly access the visual data

The separation of algorithms into these two parts enables the writing of
straightforward, error-free code and closely mirrors the theory of
*Distant Viewing* [@arnold2019distant]:

> Distant viewing is distinguished from other approaches by making explicit
> the interpretive nature of extracting semantic metadata from images.
> In other words, one must 'view' visual materials before studying them.
> Viewing,  which  we  define  as an interpretive action taken by either a
> person or a model, is necessitated by  the  way  in  which  information  is
> transferred  in visual materials. Therefore, in order to view images
> computationally,  a  representation  of  elements  contained within the
> visual material---a code system in semiotics  or,  similarly,  a  metadata
> schema  in  informatics---must  be  constructed.  Algorithms  capable  of
> automatically  converting  raw  images  into the  established  representation
> are  then  needed  to apply  the  approach  at  scale.

The annotator algorithms conduct the process of "viewing" the material whereas
the aggregator algorithms perform a "distant" (e.g., separated from the raw
materials) analysis of the visual inputs.

There are many annotators and aggregators currently available in the toolkit.
Pipelines---pre-bundled sequences of annotators and aggregators---are also
included in the package.  Details of these implementations can be found in the API
documentation. Users can also construct custom Annotator and Aggregator objects,
as described in the documentation's tutorial.

![Schematic of the Distant Viewing Toolkit's internal architecture. Algorithms
are split into two types: annotators that have access to small chunks of the
raw inputs and aggregators that have access to all of the extracted annotations
but not the input data itself.](./img/process.png){ width=13cm }

# High-level Command Line Interface

The command line tools provide a fast way to get started with the toolkit.
Designed for users with no experience programming and minimal knowledge of
machine learning, it is ideal for quickly getting
meaningful results. Users call the command line by directly executing the
Python module (e.g., "python -m dvt"), specifying the desired pipeline, and
pointing to a video file or directory of images. The output data
can be visualized using a local webserver. Figure 1 shows an example landing
page with six annotated video files. Clicking on an image opens a video-specific
page (Figure 2), which visualizes metadata about each cut. While the command line
interface is meant to be easy to run out-of-the-box, it also affords a
high-level of customization through command line options. These are documented
within the toolkit using the **argparse** package. It is also possible to
modify the visualization using custom CSS and JavaScript code. This makes the
command-line interface particularly well-suited for classroom use, following
the task-driven paradigm popular in Digital Humanities pedagogy
[@birnbaum2017task].


# Process and Development

Development of the Distant Viewing Toolkit follows best-practices for
open-source software development [@wilson2014best].
Development of the software is done publicly through our GitHub repository,
which has both a stable master branch and experimental development branch.
The project includes an open-source license (GPL-2), uses the Contributor
Covenant Code of Conduct (v1.4), and provides user templates for submitting issues
and pull requests [@tourani2017code].
We make use of integrated unit testing through the **pytest** package and
TravisCL. The code passes checks for conforming to the common
Python coding styles (checked with **pycodestyle**) and the relatively
aggressive checks provided by **pylint** [@reitz2016hitchhiker]. JavaScript
coding style was verified with the static checkers JSHint and JSLint
[@santos2015using]. Stable
versions of the package are posted to PyPI as both source packages and
pre-compiled wheels to make installation as straightforward as possible.

Because our audiences have different levels of comfort with computational approaches,
we have streamlined the installation process. The package
can be installed using a fresh version of Anaconda Python and packages
available through the pip command on PyPI (both of which can be installed
through GUIs if preferred). We have kept dependencies to a minimum, and have
avoided software that is known to be difficult to install. For example, all of
the included deep-learning algorithms are built using Tensorflow to avoid
requiring multiple deep-learning libraries [@abadi2016tensorflow]. While
Tensorflow can occasionally throw errors, we have found the CPU-version to be
relatively error-free for non-technical users to install on both Windows and
macOS relative to popular alternatives such as Caffe and Torch.

As of version 0.2.0, the core architecture of the toolkit is
stable. We plan to continue to update the available algorithms available in the
toolkit as well as make improvements to the interactive, web-based interface.
We also continue to utilize the toolkit for specific applications with research
partnerships. Our first published example application looks at Network-Era
Sitcoms in the United States and offers a model for the kind of analysis of
visual culture made possible by the Distant Viewing Toolkit [@arnold2019visual].

# Acknowledgements

The Distant Viewing Toolkit is supported through a Digital Humanities
Advancement Grant from the National Endowment for the Humanities
(HAA-261239-18).

# References
