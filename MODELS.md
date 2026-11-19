# Model provenance

The model directory is `DATA/models` and is never in git (11.9 GB). This file
records where each model came from so it can be rebuilt.

Every model weight in the project lives there — there are no checkpoints
elsewhere. A backup tarball is named for the project rather than for the bare
directory, so it is self-describing wherever it ends up:

```
tar -cf tbm_models.tar --exclude=.DS_Store -C ~/data_tbm models
```

That name is gitignored. Compression is not worth it: gzip -1 gets these files only to ~92% of
their original size (measured on `whisper-large-v3-ct2/model.bin`
and `detect_shots.pt`), since dense float tensors are close to
incompressible. That is ~1 GB saved for a long single-threaded pass over 12 GB,
and it makes the archive slower to verify and restore.

Two constraints shape everything here:

- **The cluster's compute nodes have no internet.** Every model must be staged
  from a login node before a job runs. `bin/submit_job.sh` sets `HF_HUB_OFFLINE=1`
  so a missing model fails loudly instead of hanging on a network call.
- **Caches are pinned to this directory, not `~/.cache`.** See "Cache pinning"
  below — this is opt-in per call site and easy to break.

Revisions below are the commit hashes actually on disk as of 2026-07-29, read
from each cache's `refs/main`. Pin to these rather than re-resolving `main`.

## Auto-downloadable (9.7 GB)

Fetched by library call; no manual step. Confirm with
`dvt.config.check_vendored_models()`.

| Directory | Source | Revision |
|---|---|---|
| `models--google--siglip2-so400m-patch16-naflex` (4.3 GB) | HF `google/siglip2-so400m-patch16-naflex` | `cc24074f717b612951c2dead130904ab9b65a81e` |
| `models--google--siglip2-base-patch16-224` (2.1 GB) | HF `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| `models--facebook--dinov2-large` (1.1 GB) | HF `facebook/dinov2-large` | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |
| `audio-separator` (1.2 GB) | `audio-separator` package, per `download_checks.json` | htdemucs_ft (4 × `.th`) + `vocals_mel_band_roformer.ckpt` |
| `insightface` (601 MB) | insightface `buffalo_l` pack | auto-fetched to `root=MODEL_PATH/insightface` |
| `torch` (1.1 GB) | torch hub — whisperx wav2vec2 aligners | en, fr, de staged (360 MB each) |
| `nltk_data` (11 MB) | `nltk.download("punkt_tab")` | all 20 punkt languages |

## Adding a language

**Whisper itself is not per-language.** `whisper-large-v3-ct2` is a single
multilingual model — its `config.json` carries all 100 `lang_ids` — and
`LANGUAGE` is passed to `load_model`/`transcribe` as a runtime argument.
Changing language downloads no whisper files. What *is* per-language is the
wav2vec2 aligner that converts whisper's segment output into word-level
timestamps, which is all `stage_language.py` fetches.

`dvt/words.py` aligns whatever `LANGUAGE` is set to, and each language needs
**two** assets — easy to miss, because whisperx fetches them at different
points and the second one only fails once a job is already running:

1. **A wav2vec2 aligner.** `en, fr, de, es, it` come from torchaudio and land
   in `DATA/models/torch` via `TORCH_HOME`. The other 37 languages come from
   HuggingFace and need `cache_dir=MODEL_PATH`, or they go to
   `~/.cache/huggingface` and the compute nodes can't see them.
   `align_paths.py` decides which, and is shared by the loader and the stager
   so they cannot disagree.
2. **An nltk punkt sentence splitter.** `whisperx.align()` calls
   `nltk.download("punkt_tab")` on a cache miss, mid-run. All 20 punkt
   languages are now vendored, so this is covered; languages outside those 20
   fall back to English, which is whisperx's own behaviour.

Stage from a **login node** (compute nodes have no internet), once per
language:

```
uv run python -m batch.stage_language en
```

from `src/annotate/base/`, which names this machine's model directory. Anywhere
else, `python -m dvt.stage_language en --models DIR`.

It downloads both assets, then re-loads them with the network disabled so a
gap surfaces there rather than three hours into an array job. Use
`--verify-only` to check what is already staged, and `--align-model <repo id>`
for a language whisperx has no default for.

Staged so far: **en, fr, de**. Adding es or it needs only the aligner (their
punkt is already present); anything else pulls a ~1.2 GB HuggingFace model
into `DATA/models/`.

## CTranslate2 format (2.9 GB)

**`whisper-large-v3-ct2`** is the transcription model — one multilingual
model covering all ~99 languages, not one per language. It is in CTranslate2
format, which HuggingFace does not serve for `openai/whisper-large-v3`, so it
was either converted locally or downloaded already converted.

**Precision is float16, confirmed by arithmetic** rather than inferred:
`model.bin` is 3,087,284,237 bytes, and large-v3's ~1.55 B parameters predict
~3.10 GB at fp16, ~1.55 GB at int8, ~6.2 GB at fp32.

Either route rebuilds an identical directory. Downloading is simpler:

```
# pre-converted
huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir $DATA/models/whisper-large-v3-ct2

# or convert from source
ct2-transformers-converter --model openai/whisper-large-v3 \
    --output_dir $DATA/models/whisper-large-v3-ct2 --quantization float16
```

The directory holds exactly the converter's output set — `config.json`,
`model.bin`, `preprocessor_config.json`, `tokenizer.json`, `vocabulary.json` —
with no `.gitattributes` or `README.md`, which a clone or `snapshot_download`
would normally bring along. That leans toward local conversion, but a
copied-out download looks the same, so the original route is not recoverable
from the files. It does not matter for rebuilding.

Note `dvt/words.py` sets `COMPUTE_TYPE = "int8"`, so CTranslate2 quantizes
these fp16 weights to int8 at load time on every run. Storing an int8 copy
directly would halve this directory to ~1.5 GB and skip that step; keeping
fp16 leaves `COMPUTE_TYPE` free to change without re-converting.

## Needs a recorded source (29 MB)

This cannot be re-obtained from anything written down. It needs its source
filled in.

**`detect_shots.pt` (29 MB)** — TransNetV2 shot-boundary weights, loaded at
`dvt/frames.py` via `torch.load` into the `TransNetV2` class
in `dvt/transnet.py`. The filename records nothing about its origin.

> TODO: record which TransNetV2 port/release these weights came from.

## Trained here (1.2 MB)

**`laughter/laughter_net.pt` (1.1 MB)** and
**`laughter/laughter_classifier.json` (134 KB)**: the laughter stage's
network and classifier, trained in `2026_laugh` in projects-2026. The network is the
checkpoint `19_select.py` chose from `17_train.py` (`model4/chosen.pt`),
loaded into `LaughterNet` in `dvt/laughter_net.py`. The classifier
is `23_classify.py`'s, its 200 trees written out as json by
`28_export.py`, which checks that they score as scikit-learn's
`predict_proba` does before writing them. Both files are made again by
running `28_export.py` in `2026_laugh` and copying `data/export/` into
`models/laughter`.

## Cache pinning

Models land here **only because call sites say so** — the library defaults are
`~/.cache/huggingface` and `~/.cache/torch`.

| Mechanism | Covers | Set in |
|---|---|---|
| `cache_dir=config.models()` | siglip2, siglip2_base, dinov2 | `dvt/embeddings.py` (6 call sites) |
| `config.models()/insightface` | insightface | `dvt/face.py` |
| `model_file_dir=` | audio-separator | `dvt/audioseg.py` |
| direct path | whisper, nltk | `dvt/config.py` |
| `TORCH_HOME` | torch hub | `dvt/config.py` (both machines) + `bin/submit_job.sh` |
| `HF_HUB_CACHE` | HF default cache | `bin/submit_job.sh` only — see below |

`dvt.config.configure()` sets `TORCH_HOME`. torch reads it
inside `torch.hub.get_dir()` at call time, so it works even though callers run
`import torch` before the directory is named.

The HF cache **cannot** be set from `dvt/config.py`: `huggingface_hub`
freezes its cache constants at import time, and every caller imports
transformers first.
It is exported in `bin/submit_job.sh` instead, before Python starts. Locally the
default cache stays at `~/.cache/huggingface`, which is harmless because the
explicit `cache_dir=` arguments override it.

Use `HF_HUB_CACHE`, not `HF_HOME`. `HF_HOME=models` resolves the cache to
`DATA/models/hub/`, which would sit alongside — and diverge from — the
`models--*` directories that `cache_dir=MODEL_PATH` writes directly into
`DATA/models/`.

**Consequence:** any new `from_pretrained` without `cache_dir=MODEL_PATH` will
download to `~/.cache/huggingface`, work fine locally, and fail on the
cluster. `config.assert_torch_home_pinned()` and
`config.check_vendored_models()` exist to catch this before a job dies.
