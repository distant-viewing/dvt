# dvt

The Distant Viewing Toolkit: the annotators, one video in and parquet tables
out. The package is given to people who keep no corpus, so it imports
nothing from the repository around it; `tests/test_layer.py` fails if that
changes. `README.md` says how it is run.

`dvt/tools/` reads a media file without a model. It is all that is installed
without an extra, and the code that keeps the corpus imports it from every
stage, including the ones that are never type checked.

- Do not change what a function in `dvt/tools/` returns or raises for the
  input it is given today. Find its callers across the repository
  (`grep -rn "dvt.tools"`) before changing a signature. A new behaviour is a
  new function beside the old one.
- Nothing in `dvt/tools/` or `dvt/__init__.py` imports more than numpy and
  polars. The registry and the ingest install the package without an extra.
- ffprobe's report and table rows reach the older functions as plain dicts.
  Where a function hands such a dict back to callers that index it, its
  type is `Raw` from `dvt.tools.values` (`dict[str, Any]`), and that is the
  one place `Any` is allowed. A function that reads such a dict takes
  `Mapping[str, object]` and converts each field with `int_or_none`,
  `float_or_none` or `text` from `values`, which give what `int()`,
  `float()` or the old code gave and None where the old `try` caught an
  error. New code reads ffprobe's json through `dvt.tools.probejson`.
- The expected values in `tests/test_filehash.py` and `tests/test_probe.py`
  were taken from the code before the strict checks came in. Every hash in
  the corpus record and every inventory row depends on them; a change that
  moves them is a bug.

The whole package is type checked with basedpyright at the standard level.
The modules listed under `[tool.basedpyright] strict` in `pyproject.toml`,
which is every module but four, and the tests are also formatted and
linted with ruff and checked strictly.

The four stay at the standard level, with their functions annotated:
`transnet.py` and `laughter_net.py`, whose layers call torch's
`nn.Module.__init__`, and `embeddings.py` and `stage_language.py`, which
load through transformers' `from_pretrained`. Each of those is untyped where
strict mode looks. Do not run `ruff format` on the four.

- Run the checks before saying the work is done, and leave them passing.
  Where this repository sits in another as the submodule `src/dvt`, that is
  `bin/check.py src/dvt` from the outer root. On its own it is `uv run ruff
  format` and `uv run ruff check` on the strict modules, then `uv run
  basedpyright` and `uv run pytest`. The checks are in the `dev` group, a default
  group beside `cpu`, which names the `video`, `face` and `audio` extras; the GPU environment is synced with
  `--no-default-groups` and goes without them.
- A table an annotator writes is compared across the whole corpus, so a
  change to what it writes for the same video is a change to the annotator,
  not a tidying. Check old against new on the same input before calling a
  change neutral, and say so when it is not.
- The tests load no trained model. They make small videos with ffmpeg
  (`make_video` in `tests/conftest.py`) and are skipped where it is missing;
  `tests/test_run.py` stands in for the model annotators. The frames stage
  runs TransNetV2 with weights drawn from a seeded generator (the `models`
  fixture): its shot scores mean nothing, but they are the same every run,
  which is what a comparison of old and new needs. The tracks stage runs
  on scenes `tests/scene.py` makes up, and the embedding, YOLO and face
  stages on a fake model. In `tests/test_audio_stages.py` praat measures
  made-up tones, audioseg runs a fake separator that scales its input, and
  words a fake whisperx and voice detector. What surrounds each model is
  what is tested.
- To check the model stages themselves, run every annotator over the same
  few clips on spydur with the old copy of the package and the new one,
  each put first on `PYTHONPATH` in one GPU environment, and compare the
  tables. Run the new copy twice as well: the GPU stages do not give the
  same numbers twice, and the two new runs show by how much.
- scipy's types come from `scipy-stubs` in the dev group, pinned to the
  minor release of scipy the environment holds.
- whisperx, silero-vad, nltk, insightface, onnx2torch, audio-separator,
  parselmouth, soundfile and torchaudio ship no types, so `typings/` holds a
  stub for each. A stub stands in for the whole package, so it declares
  every name used from that library, copied from the installed release.
  Using another name from one of them means adding it to its stub.
  `typings/whisperx/schema.pyi` differs from whisperx on purpose: a word
  whisperx could not align comes back without its start, end and score.
- json from ffprobe, and a row handed on as it was read, is `Raw` from
  `dvt.tools.values`.
- `run.py`, `video.py`, `frames.py`, `embeddings.py`, `yolo.py`, `face.py`,
  `audioseg.py` and `words.py` import `runtime` before anything else,
  inside an `isort: off` block, since
  `runtime` sets the thread count and quiets the libraries before any of
  them loads. Keep it first; ruff
  would otherwise sort it after torch.
- `letterbox.sampled` keeps the frames of the last video it sampled, by
  path; letterbox, chroma and metadata read the same frames of one film.
- ultralytics is installed only in the GPU environment, so `yolo.py` looks
  it up by name with `importlib`, where the checks do not need to find it.
- A new module goes in the strict list, with its tests.
