"""dvt models fetch: brings the models the annotators load into the model directory.

MODELS names, for each annotator that loads one, what has to be under the
model directory and how it gets there. `dvt models check` says which are
there, and `dvt models fetch` downloads the ones that are not. Nothing here
is imported by an annotator: a run on a machine with no network reads the
directory as it stands.
"""

import argparse
import os
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import config, pins

# The weights that are published nowhere else: the shot boundary network,
# the head detector and the laughter models.
OWN = "distant-viewing/dvt-models"
OWN_REVISION = "1a21874c6d0b7daafa7ed0353c6e602d79c1ffd8"

# whisper-large-v3 in CTranslate2 format, which is what whisperx loads.
WHISPER = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
WHISPER_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)

# What a transformers loader reads. A hub repository can hold the same
# weights in several formats, and only the safetensors are wanted.
HUB_FILES = ("*.json", "*.safetensors", "*.model", "*.txt")

# The two separations audioseg runs, as audio-separator names them.
SEPARATIONS = ("htdemucs_ft.yaml", "vocals_mel_band_roformer.ckpt")

FACE_PACK = "buffalo_l"

# The languages whose aligner is fetched when none is named.
LANGUAGES = ("en",)

OFFLINE = "HF_HUB_OFFLINE"


@dataclass(frozen=True)
class Model:
    # Under the model directory. The model is there when every one is.
    paths: tuple[str, ...]
    # Downloads it, given the languages asked for.
    fetch: Callable[[Sequence[str]], None]
    # The extra of this package that installs what the download needs.
    extra: str
    # Terms a user should read before the weights are used.
    terms: str = ""


def hub_dir(repo: str) -> str:
    """Where the hub's own layout keeps one revision of a repository."""
    return f"models--{repo.replace('/', '--')}/snapshots/{pins.HUB[repo]}"


def own(*files: str) -> Callable[[Sequence[str]], None]:
    def fetch(languages: Sequence[str]) -> None:
        from huggingface_hub import hf_hub_download

        for name in files:
            hf_hub_download(OWN, name, revision=OWN_REVISION, local_dir=config.models())

    return fetch


def hub(repo: str) -> Callable[[Sequence[str]], None]:
    def fetch(languages: Sequence[str]) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo,
            revision=pins.HUB[repo],
            cache_dir=config.models(),
            allow_patterns=list(HUB_FILES),
        )

    return fetch


def face(languages: Sequence[str]) -> None:
    from insightface.utils.storage import ensure_available

    ensure_available("models", FACE_PACK, root=str(config.models() / "insightface"))


def separations(languages: Sequence[str]) -> None:
    from audio_separator.separator import Separator

    config.separator().mkdir(parents=True, exist_ok=True)
    separator = Separator(model_file_dir=str(config.separator()))
    for name in SEPARATIONS:
        separator.download_model_files(name)


def words(languages: Sequence[str]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        WHISPER,
        revision=WHISPER_REVISION,
        local_dir=config.whisper(),
        allow_patterns=list(WHISPER_FILES),
    )
    from .stage_language import stage_aligner, stage_punkt

    for language in languages:
        stage_aligner(language, None)
        stage_punkt(language)


MODELS = {
    "frames": Model(("detect_shots.pt",), own("detect_shots.pt"), "video"),
    "dino2": Model(
        (hub_dir("facebook/dinov2-large"),),
        hub("facebook/dinov2-large"),
        "video",
    ),
    "siglip2": Model(
        (hub_dir("google/siglip2-so400m-patch16-naflex"),),
        hub("google/siglip2-so400m-patch16-naflex"),
        "video",
    ),
    "siglip2_base": Model(
        (hub_dir("google/siglip2-base-patch16-224"),),
        hub("google/siglip2-base-patch16-224"),
        "video",
    ),
    "face": Model(
        (
            f"insightface/models/{FACE_PACK}/det_10g.onnx",
            f"insightface/models/{FACE_PACK}/w600k_r50.onnx",
        ),
        face,
        "face",
        "the insightface models are for non-commercial research only",
    ),
    "yolo": Model(
        ("yolo26l_heads.pt",),
        own("yolo26l_heads.pt"),
        "yolo",
        "trained from an ultralytics model and, like ultralytics, under the AGPL-3.0 license",
    ),
    "audioseg": Model(
        tuple(f"audio-separator/{name}" for name in SEPARATIONS),
        separations,
        "audio",
    ),
    "words": Model(
        ("whisper-large-v3-ct2/model.bin", "nltk_data/tokenizers/punkt_tab"),
        words,
        "audio",
    ),
    "laughter": Model(
        ("laughter/laughter_net.pt", "laughter/laughter_classifier.json"),
        own("laughter/laughter_net.pt", "laughter/laughter_classifier.json"),
        "audio",
    ),
}


def missing(names: Collection[str] | None = None) -> list[str]:
    """The annotators, of those named or of all, whose model is not in the
    model directory. One that loads no model is never missing."""
    root = config.models()
    return [
        name
        for name, model in MODELS.items()
        if (names is None or name in names) and not all((root / p).exists() for p in model.paths)
    ]


def fetch(names: Collection[str] | None = None, languages: Sequence[str] = LANGUAGES) -> list[str]:
    """Downloads what `missing` lists and returns the names it fetched."""
    absent = missing(names)
    for name in absent:
        model = MODELS[name]
        print(f"[{name}] fetching into {config.models()}", flush=True)
        if model.terms:
            print(f"[{name}] note: {model.terms}", flush=True)
        try:
            model.fetch(languages)
        except ModuleNotFoundError as e:
            raise SystemExit(
                f"{name}: {e.name} is not installed. Install the package with "
                f"the {model.extra} extra, dvt[{model.extra}]."
            ) from e
    return absent


def report(names: Collection[str] | None, root: Path) -> list[str]:
    absent = missing(names)
    for name in MODELS:
        if names is None or name in names:
            print(f"{name:<13} {'missing' if name in absent else 'there'}")
    if absent:
        print(f"{len(absent)} missing from {root}; `dvt models fetch` downloads them")
    return absent


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="dvt models", description="The models the annotators load, in one directory."
    )
    ap.add_argument("action", choices=("fetch", "check"))
    ap.add_argument(
        "--models",
        type=Path,
        help=f"the model directory; ${config.MODELS_ENV} if not given, "
        f"and {config.DEFAULT_MODELS} if that is not set",
    )
    ap.add_argument(
        "--annotators",
        help="comma-separated annotators to fetch for, all by default: " + ", ".join(MODELS),
    )
    ap.add_argument(
        "--language",
        action="append",
        help="a language whose words are to be aligned, as a two-letter code; "
        "may be given more than once, en by default",
    )
    args = ap.parse_args(argv)

    names: list[str] | None = None
    if args.annotators:
        names = [n.strip() for n in args.annotators.split(",") if n.strip()]
        unknown = [n for n in names if n not in MODELS]
        if unknown:
            ap.error(f"no model is fetched for: {', '.join(unknown)}")
    if args.models is not None:
        config.configure(args.models)
    root = config.models()

    if args.action == "check":
        return 1 if report(names, root) else 0

    if os.environ.get(OFFLINE) == "1":
        ap.error(f"{OFFLINE}=1 is set; unset it to download")
    root.mkdir(parents=True, exist_ok=True)
    fetch(names, args.language or LANGUAGES)
    return 1 if report(names, root) else 0
