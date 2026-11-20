# Distant Viewing Toolkit for the Analysis of Visual Culture

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI version shields.io](https://img.shields.io/pypi/v/dvt.svg)](https://pypi.python.org/pypi/dvt/) [![PyPI status shields.io](https://img.shields.io/pypi/status/dvt)](https://pypi.python.org/pypi/dvt/) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01800/status.svg)](https://doi.org/10.21105/joss.01800)

The Distant Viewing Toolkit is a Python package that facilitates the
computational analysis of still and moving images. It takes a video, an
image or a directory of images and writes a set of tables about it, each a
parquet file: the shots of a video, the faces and heads in each frame, image
embeddings, the words spoken, and more. Each table is made by one annotator,
and the annotators are listed below.

Version 2 of the toolkit is a rewrite, and nothing written for version 1
runs on it.

More information about the toolkit and project is available on the
following pages:

* Example analysis using aggregated metadata: ["Visual Style in Two Network Era Sitcoms"](https://culturalanalytics.org/article/11045-visual-style-in-two-network-era-sitcoms)
* Theory of the project: ["Distant Viewing: Analyzing Large Visual Corpora."](https://doi.org/10.1093/llc/fqz013)
* Software Whitepaper: [A Python Package for the Analysis of Visual Culture](https://doi.org/10.21105/joss.01800)

If you have any trouble using the toolkit, please open a
[GitHub issue](https://github.com/distant-viewing/dvt/issues). If you
have additional questions or are interested in collaborating, please contact
us at tarnold2@richmond.edu and ltilton@richmond.edu.

## Install

The toolkit needs Python 3.13, and ffmpeg and ffprobe on the path:

```
pip install "dvt[all]"
```

The extras choose which annotators can run. `video` brings the annotators
that read the picture, `face` the face annotator and the tracks, `yolo` the
head detector, and `audio` the annotators that read the sound. `all` brings
every one. Without an extra the package is `dvt.tools` alone, which reads a
media file without a model.

The annotators load their models from one directory, and the models are
downloaded once, before anything is annotated:

```
dvt models fetch
```

All of them come to about 12 GB. `--annotators frames,dino2,face` fetches
only what those annotators load, and `dvt models check` lists which models
are there. The directory is `~/.cache/dvt` unless `--models` or the
environment variable `DVT_MODELS` names another. [MODELS.md](MODELS.md) says
what the directory holds, where each model comes from and under what terms
it is used.

## Use

```
dvt film.mp4 --out out
```

Under `out/` this leaves `annotations/film/` with the tables, `audio/` with
the sound taken from the video, `frames/film/` with the saved frames and
`stems/` with the separated sound. `--annotators a,b,c` runs only those
named, and `--list` prints every annotator, what it writes and what it reads.

An image, or a directory of images, is given the same way. [Images](#images)
below says what is made of one.

The same from Python, where the tables come back as polars data frames by
name:

```python
import dvt

tables = dvt.annotate("film.mp4", out="out", annotators=["metadata", "frames", "shots"])
tables["shots"]
```

With no `out` nothing is kept but the tables returned, and with no
`annotators` every one that fits the input runs. `models` names the model
directory.

## Images

```
dvt poster.jpg --out out
dvt posters/ --out out --annotators dino2,face
```

A path is taken as an image when it ends in `.jpg`, `.jpeg`, `.png`, `.webp`,
`.bmp`, `.tif` or `.tiff`, and as a set of images when it is a directory.
Every image under the directory is read, however deep, in the order of
their sorted paths. Anything else is read as a video.

Five annotators read a single picture, and they are the ones an image has:
`dino2`, `siglip2`, `siglip2_base`, `face` and `yolo`. All five run when
`--annotators` names none, so the models of all five have to be in the model
directory; `dvt models fetch --annotators dino2,face` brings only those of
the two. Naming any other annotator for an image is an error.

The tables go to `out/annotations/poster/` for the image `poster.jpg` and
to `out/annotations/posters/` for the directory `posters/`. The images of a
directory share one set of tables:

| Table                  | Rows                                               |
|------------------------|----------------------------------------------------|
| `images.parquet`       | one an image: its `path`, `width` and `height`     |
| `dino2.parquet`        | one an image, with its embedding                   |
| `siglip2.parquet`      | one an image, with its embedding                   |
| `siglip2_base.parquet` | one an image, with its embedding                   |
| `face.parquet`         | one a face found, with its box and its embedding   |
| `yolo.parquet`         | one a head found, with its box                     |

Each table has a `path` column where that of a video has `frame`. The path
is stated from the directory given, as in `1950s/poster_012.jpg`, and a
single image is stated by its file name. An image that cannot be read stays
in `images.parquet` with no width or height and has no row in the others.
The images need not be of one size.

From Python:

```python
import dvt

tables = dvt.annotate("posters/", annotators=["dino2", "face"])
tables["images"]
tables["dino2"]
```

## Annotators

| Annotator      | Writes                         | What it holds                                              |
|----------------|--------------------------------|------------------------------------------------------------|
| `metadata`     | `metadata.parquet`             | the frame size, the crop, the aspect and the language      |
| `letterbox`    | `letterbox.parquet`            | the black bars around the picture                          |
| `chroma`       | `chroma.parquet`               | colour or black and white                                  |
| `chapters`     | `chapters.parquet`             | the chapters in the file                                   |
| `subtitles`    | `subtitles.parquet`            | the text subtitles in the file                             |
| `frames`       | `frames.parquet`               | one row a frame, with the shot boundary TransNetV2 gives   |
| `shots`        | `shots.parquet`                | the frames cut into shots, and what each shot looks like   |
| `frames_r720`  | `frames/`                      | frames of each shot saved as images                        |
| `dino2`        | `dino2.parquet`                | DINOv2 embeddings of the sampled frames                    |
| `siglip2`      | `siglip2.parquet`              | SigLIP2 embeddings of the sampled frames                   |
| `siglip2_base` | `siglip2_base.parquet`         | the same from the smaller SigLIP2 model                    |
| `face`         | `face.parquet`                 | SCRFD face detection and an ArcFace embedding, every frame |
| `yolo`         | `yolo.parquet`                 | the heads detected in every frame                          |
| `tracks`       | `tracks.parquet`               | head and face tracks within shots, one embedding a track   |
| `audioseg`     | `audioseg_intervals.parquet`   | the sound separated into stems, and where the speech is    |
| `words`        | `words.parquet`                | the words spoken, each with its time                       |
| `praat`        | `praat.parquet`                | phonetic measurement of the speech, frame by frame         |
| `laughter`     | `laughter.parquet`             | laughter heard between lines                               |

`metadata` runs first, and the others are checked against what it records.
`tracks` reads the tables of `shots`, `face` and `yolo`. `words`, `praat` and
`laughter` read the speech that `audioseg` separates.

`audioseg` stops unless it has a CUDA device, so `words`, `praat` and
`laughter` need one too. The annotators of the `video` and `face` extras
also run on a CPU, more slowly.

## Notebooks

If you are interested in learning more about application of computer vision and
the distant viewing toolkit to humanities applications, we offer a self-guided
tutorial through the following Google Colab notebooks. These can be run for free
by anyone with a Google account:

- Distant Viewing Tutorial 1: Documerica and Color Analysis: [[colab](https://colab.research.google.com/github/distant-viewing/dvt/blob/main/tutorials/Distant_Viewing_Tutorial_1_Documerica_and_Color_Analysis.ipynb)], [[notebook](https://github.com/distant-viewing/dvt/blob/main/tutorials/Distant_Viewing_Tutorial_1_Documerica_and_Color_Analysis.ipynb)]
- Distant Viewing Tutorial 2: Shots and Faces in Moving Images: [[colab](https://colab.research.google.com/github/distant-viewing/dvt/blob/main/tutorials/Distant_Viewing_Tutorial_2_Shots_and_Faces_in_Moving_Images.ipynb)], [[notebook](https://github.com/distant-viewing/dvt/blob/main/tutorials/Distant_Viewing_Tutorial_2_Shots_and_Faces_in_Moving_Images.ipynb)]

The colab links open the notebooks in this repository.

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

## License

The toolkit is released under the MIT license; see [LICENSE](LICENSE). The
models the annotators load, and one of the libraries that run them, have
terms of their own, and [MODELS.md](MODELS.md) gives them model by model.
Two matter to anyone whose use is not research or not open source:

- The `face` annotator loads the insightface `buffalo_l` models. The
  insightface code is under the MIT license, but its authors make the models
  "available for non-commercial research purposes only".
- The `yolo` annotator runs on ultralytics, which is under the AGPL-3.0
  license unless a license is bought from Ultralytics, and the head detector
  it loads was trained from an ultralytics model and is under the AGPL-3.0
  too. ultralytics is installed only by the `yolo` and `all` extras.

The aligners that time the words of French, German, Spanish and Italian
speech are under CC BY-NC 4.0.

## Contributing

Contributions, including bug fixes and new features, to the toolkit are
welcome. When contributing to this repository, please first discuss the change
you wish to make via a GitHub issue or email with the maintainers of this
repository before making a change. Small bug fixes can be given directly
as pull requests.
