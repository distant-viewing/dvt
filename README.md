
# Distant Viewing Toolkit for the Analysis of Visual Culture

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI version shields.io](https://img.shields.io/pypi/v/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI status shields.io](https://img.shields.io/pypi/status/dvt)](https://pypi.python.org/pypi/dvt/) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01800/status.svg)](https://doi.org/10.21105/joss.01800)

The Distant Viewing Toolkit is a Python package that facilitates the 
computational analysis of still and moving images. The most recent
version of the package focuses on providing a minimal set of functions
that require only a small set of dependencies. Examples of how to make
use of the toolkit are given in the following section.

For more information about setting up the toolkit on your own machine, please
see [INSTALL.md](INSTALL.md). More information about the toolkit and project is
available on the following pages:

* Example analysis using aggregated metadata: ["Visual Style in Two Network Era Sitcoms"](https://culturalanalytics.org/article/11045-visual-style-in-two-network-era-sitcoms)
* Theory of the project: ["Distant Viewing: Analyzing Large Visual Corpora."](https://doi.org/10.1093/llc/fqz013)
* Software Whitepaper: [A Python Package for the Analysis of Visual Culture](https://doi.org/10.21105/joss.01800)

If you have any trouble using the toolkit, please open a
[GitHub issue](https://github.com/distant-viewing/dvt/issues). If you
have additional questions or are interested in collaborating, please contact
us at tarnold2@richmond.edu and ltilton@richmond.edu.

## Example Usage

To use the toolkit on a still image, we first use the `load_image`
function to load the image in Python. Then, we create an annotation
model; below we'll use an annotation that detects and identifies 
faces. Finally, we apply the annotation to the image and save the
results.

```{py}
import dvt

img = dvt.load_image("input/obama2.jpg")
anno_face = dvt.AnnoFaces()
out_face = anno_face.run(img, visualize=True)
```

The object `out_face` is a dictionary with elements that tell us
about the detected faces. These have been designed so that they
can be easily converted to a Panda's data frame, as follows.

```{py}
import pandas as pd

pd.DataFrame(out_face['boxes'])
```
```
   face_id     x  xend    y  yend      prob
0        0   992  1112  292   458  1.000000
1        1   631   749  237   397  1.000000
2        2   422   589  232   470  0.999998
3        3  1124  1247  161   330  0.999997
4        4  1161  1278  719   861  0.921809
```

The algorithm has detected five faces, four with a very high confidence
and a fifth with a reasonable level of confidence. We can look at the output
of the algorithm by saving the annotated image using the `save_image` function.

```{py}
dvt.save_image("faces.png", out_face['img'])
```

Which produces an image like this:

<img src=".images/faces.png" alt="Detected Faces" width="700px"/>

You can see that the algorithm has correctly identified four faces, but that
the fifth is actually a shadow. While many computer vision algorithms will show
you the "perfect" examples that seem to work without errors, keep in mind that
while good, the algorithms still make frequent errors.

Another analysis that we can do with the face detection annoations is use
the output to identify the individuals in an image. Let's run the annotation
over another image consisting of a portrait of President Obama. We can then
see how close this face is to those detected in the family photo.

```{py}
import numpy as np

img_portrait = dvt.load_image("input/obama1.jpg")
out_portrait = anno_face.run(img_portrait)
np.sum(out_portrait['embed'][0] * out_face['embed'], axis=1)
```
```
array([-0.05723718,  0.04770625,  0.86795366,  0.11890081,  0.05184552],
      dtype=float32)
```

You can see that the closest image, by far, is the third face. Looking back
at the metadata, the third face has the lowest x value, and therefore (correctly)
identifies Obama as being on the far left of the frame.

## Other Annotations and Inputs

There are currently four different annotations that extract information from
a still image. These work just like the face annotation above and, include 
the following:

```{py}
anno_keypoints = dvt.AnnoKeypoints()
anno_segment = dvt.AnnoSegmentation()
anno_embed = dvt.AnnoEmbed()
anno_face = dvt.AnnoFaces()
```

There is also a special annotation type that takes a path to a video file and
estimates the location of the video shots.

```{py}
anno_breaks = dvt.AnnoShotBreaks()
out_breaks = anno_breaks.run("input/tm_short.mp4")
```

Finally, there is also a helper function extract frames from a video file. 
The example below shows how to select every 25th frame from a video file
and applies the face annotation to each frame.

```{py}
anno_face = dvt.AnnoFaces()
output = []
for img, frame, msec in dvt.yield_video("input/tm_short.mp4"):
    if (frame % 25) == 0:
        anno = anno_face.run(img, visualize=True)
        if anno:
            anno['frame'] = frame
            output += [anno]
```

We are working on longer tutorials that we hope to release in early 2023. These
will be posted here as soon as they are ready.

------------------

<br>

<a href="https://www.neh.gov/" title="National Endowment for the Humanities"><img align="left" src=".images/neh_logo_horizlarge.jpg" alt="NEH" class="rpad" width="300px"></a> The Distant Viewing Toolkit is supported by the National Endowment for the Humanities through a Digital Humanities Advancement Grant.

<br>

------------------

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

## Contributing

Contributions, including bug fixes and new features, to the toolkit are
welcome. When contributing to this repository, please first discuss the change
you wish to make via a GitHub issue or email with the maintainers of this
repository before making a change. Small bug fixes can be given directly
as pull requests.
