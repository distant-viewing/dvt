# Models

The annotators load their models from one directory. This file says how to
fill it, what it holds, where each model comes from and under what terms it
is used.

## The model directory

The directory is the one named by `--models`, or by the environment variable
`DVT_MODELS` when that is not given, or `~/.cache/dvt` when neither is set.
From Python it is the `models` argument of `dvt.annotate`, or
`dvt.config.configure(models)`.

```
dvt models fetch --models /path/to/models
dvt models check --models /path/to/models
```

`fetch` downloads every model that is not there and `check` lists which are.
`--annotators frames,dino2,face` keeps either to the models those annotators
load. A download needs the extra that installs its annotator, and `fetch`
names the extra when it is missing. All of the models together are about
12 GB.

An annotator never downloads anything. It reads the directory as it stands
and stops when its model is not there. On a machine with no network, run
`dvt models fetch` somewhere that has one and copy the directory across.
`fetch` refuses to run while `HF_HUB_OFFLINE=1` is set.

## What the directory holds

| Annotator      | Under the model directory                        |   Size | Comes from                             |
|----------------|--------------------------------------------------|-------:|----------------------------------------|
| `frames`       | `detect_shots.pt`                                |  29 MB | `distant-viewing/dvt-models`           |
| `dino2`        | `models--facebook--dinov2-large/`                | 1.1 GB | `facebook/dinov2-large`                |
| `siglip2`      | `models--google--siglip2-so400m-patch16-naflex/` | 4.3 GB | `google/siglip2-so400m-patch16-naflex` |
| `siglip2_base` | `models--google--siglip2-base-patch16-224/`      | 1.4 GB | `google/siglip2-base-patch16-224`      |
| `face`         | `insightface/models/buffalo_l/`                  | 601 MB | the insightface `buffalo_l` pack       |
| `yolo`         | `yolo26l_heads.pt`                               |  53 MB | `distant-viewing/dvt-models`           |
| `audioseg`     | `audio-separator/`                               | 1.2 GB | the `audio-separator` package          |
| `words`        | `whisper-large-v3-ct2/`                          | 2.9 GB | `Systran/faster-whisper-large-v3`      |
| `words`        | `torch/hub/checkpoints/`                         | 360 MB | torchaudio, one aligner a language     |
| `words`        | `nltk_data/tokenizers/punkt_tab/`                |  11 MB | nltk                                   |
| `laughter`     | `laughter/`                                      | 1.2 MB | `distant-viewing/dvt-models`           |

A name with a slash is a repository on the Hugging Face hub. The other
annotators load no model. `tracks` reads the tables of `face` and `yolo`,
and `praat` and `laughter` read the speech that `audioseg` separates.

## Pinned revisions

Every model from the hub is fetched at a fixed commit, so that a table made
next year comes from the weights that made it this year. The embedding
models are pinned in `dvt/pins.py`, and the annotators load the revision
that `fetch` downloads. Whisper and the toolkit's own weights are pinned in
`dvt/fetch.py`.

| Repository                             | Revision                                   |
|----------------------------------------|--------------------------------------------|
| `facebook/dinov2-large`                | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |
| `google/siglip2-so400m-patch16-naflex` | `cc24074f717b612951c2dead130904ab9b65a81e` |
| `google/siglip2-base-patch16-224`      | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| `Systran/faster-whisper-large-v3`      | `edaa852ec7e145841d8ffdb056a99866b5f0a478` |
| `distant-viewing/dvt-models`           | `1a21874c6d0b7daafa7ed0353c6e602d79c1ffd8` |

The insightface pack, the separation models, the torchaudio aligners and
the nltk data have no revision to pin. Each is whatever its library serves
under that name.

## The toolkit's own weights

Four files are kept in `distant-viewing/dvt-models`, since nothing else serves
them in the form the annotators load.

**`detect_shots.pt`** holds the weights of TransNetV2, the shot boundary
network of Souček and Lokoč. They are the weights its authors published at
[soCzech/TransNetV2](https://github.com/soCzech/TransNetV2), under
`inference/transnetv2-weights`, as the PyTorch state that the repository's
`inference-pytorch/convert_weights.py` makes from them. `dvt/transnet.py`
is the network they load into. The copy is kept because that repository
stores its weights in Git LFS, and its LFS quota is so often used up that
the files cannot be relied on to download.

**`yolo26l_heads.pt`** is a head detector. It was trained with ultralytics
8.4.117 for 40 epochs at 640 pixels, starting from the library's
`yolo26l.pt`, on 224,812 frames from 259 films that hold 520,210 head
boxes. Another 14 films were held out for validation and 15 for testing.

**`laughter/laughter_net.pt`** and **`laughter/laughter_classifier.json`**
are the network and the classifier of the laughter annotator, both trained
by the Distant Viewing Lab. The network loads into `LaughterNet` in
`dvt/laughter_net.py`. The classifier is 200 trees written out as json,
checked when they were written to score as scikit-learn's `predict_proba`
does. The lab's training code is its project `2026_laugh`, where
`28_export.py` writes both files.

## Whisper

`whisper-large-v3-ct2` is the transcription model. It is one multilingual
model, not one a language, in the CTranslate2 format that whisperx loads.
`fetch` takes it already converted from `Systran/faster-whisper-large-v3`:
`config.json`, `model.bin`, `preprocessor_config.json`, `tokenizer.json`
and `vocabulary.json`. The same directory can be made from the original
weights:

```
ct2-transformers-converter --model openai/whisper-large-v3 \
    --output_dir /path/to/models/whisper-large-v3-ct2 \
    --quantization float16
```

The weights are stored as float16. `dvt/words.py` sets
`COMPUTE_TYPE = "int8"`, so CTranslate2 quantizes them when it loads them.

## Languages

Whisper transcribes every language with the one model. What a language
needs of its own is the wav2vec2 aligner that gives each word its time, and
an nltk sentence splitter.

`dvt models fetch` brings English. `--language fr`, which may be given more
than once, names others, and is read when the models of `words` are not yet
there. To add a language to a directory that has them:

```
python -m dvt.stage_language fr --models /path/to/models
```

It downloads the aligner and the splitter, then loads both with the network
turned off, so that a gap shows then and not in the middle of a run.
`--verify-only` checks what is there without downloading, and
`--align-model REPO` names an aligner for a language whisperx has no
default for.

The aligners for English, French, German, Spanish and Italian come from
torchaudio, 360 MB each, into `torch/hub/checkpoints/`. Those of the other
languages whisperx knows come from the Hugging Face hub, about 1.2 GB each,
into the model directory. nltk has a splitter for 20 languages and all 20
come with the first download. A language outside them is split as English,
which is what whisperx does.

## Terms

The toolkit is under the MIT license. The models are not all under the same
terms, and some may not be used commercially. Each statement below was read
from the source it names in November 2026.

| Model                               | Terms                                   |
|-------------------------------------|-----------------------------------------|
| TransNetV2, `detect_shots.pt`       | MIT                                     |
| DINOv2 and both SigLIP2 models      | Apache 2.0                              |
| Whisper, as converted by Systran    | MIT                                     |
| the aligner for English             | MIT                                     |
| the aligners for fr, de, es and it  | CC BY-NC 4.0                            |
| the laughter models                 | MIT                                     |
| insightface `buffalo_l`             | non-commercial research only            |
| `yolo26l_heads.pt`, and ultralytics | AGPL-3.0, or a license from Ultralytics |

**insightface.** The library's code is under the MIT license. Its models
are not. The project's README says: "The training data containing the
annotation (and the models trained with these data) are available for
non-commercial research purposes only", and that this holds for the models
its Python library downloads by itself, which is how `buffalo_l` arrives.
For any other use of `buffalo_l` it gives the address
recognition-oss-pack@insightface.ai. The `face` annotator and the face half
of `tracks` rest on this pack.

**ultralytics.** The library is under the AGPL-3.0 license, and Ultralytics
sells an Enterprise License for use of its "software and AI models"
without the AGPL's conditions. `yolo26l_heads.pt` was trained from the
library's own `yolo26l.pt` and is distributed under the AGPL-3.0 as well.
The `yolo` extra installs the library, and the `yolo` annotator imports it.
Nothing else in the toolkit does, so an installation without that extra
holds no AGPL code.

**The aligners.** torchaudio's notes on its wav2vec2 models say that the
English one was published by the authors of wav2vec 2.0 under the MIT
license, and that the French, German, Spanish and Italian ones were
published by the authors of VoxPopuli under CC BY-NC 4.0. An aligner from
the hub has the terms its repository states.

**The separation models.** `audioseg` runs `htdemucs_ft`, from Demucs,
which is under the MIT license, and `vocals_mel_band_roformer.ckpt`, whose
repository `KimberleyJSN/melbandroformer` states the MIT license.

## Where a library would otherwise put a model

transformers and torch hub download to `~/.cache/huggingface` and
`~/.cache/torch` unless told otherwise. The annotators tell them: every
`from_pretrained` passes `cache_dir=config.models()`, insightface and
audio-separator are handed a directory under it, and `config.configure`
sets `TORCH_HOME` to `torch/` under it. A new loader that leaves out
`cache_dir` will work on a machine with a network and fail on one without.

`huggingface_hub` reads the location of its own cache when it is imported,
so `dvt.config` cannot move it afterwards. Set `HF_HUB_CACHE` to the model
directory before Python starts if anything else is to find the models
there. `HF_HOME` is not the same thing: it puts the cache in a `hub/`
directory beneath the one it names.
