"""Stage one language's alignment assets into the model directory."""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import config

import nltk
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from whisperx.utils import PUNKT_LANGUAGES

from .align_paths import HUGGINGFACE, TORCHAUDIO, resolve_align_model, supported_languages

def checkpoint_dir() -> Path:
    return config.torch_home() / "hub" / "checkpoints"


def stage_aligner(language: str, align_model: str | None) -> tuple[str, str]:
    backend, model_name = resolve_align_model(language, align_model)

    if backend == TORCHAUDIO:
        print(f"  aligner: torchaudio {model_name} -> {checkpoint_dir()}")
        bundle = torchaudio.pipelines.__dict__[model_name]
        bundle.get_model()
    else:
        print(f"  aligner: huggingface {model_name} -> {config.models()}")
        Wav2Vec2Processor.from_pretrained(model_name, cache_dir=config.models())
        Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=config.models())

    return backend, model_name


def stage_punkt(language: str) -> str:
    punkt_lang = PUNKT_LANGUAGES.get(language, "english")
    target = config.nltk_data() / "tokenizers" / "punkt_tab"

    if (target / punkt_lang).exists():
        print(f"  punkt:   {punkt_lang} already present in {config.nltk_data()}")
        return punkt_lang

    print(f"  punkt:   {punkt_lang} -> {config.nltk_data()}")
    with tempfile.TemporaryDirectory() as tmp:
        if not nltk.download("punkt_tab", download_dir=tmp, quiet=True):
            raise RuntimeError(
                "nltk could not download punkt_tab. Check network access; on the "
                "cluster this must be run from a login node."
            )
        source = Path(tmp) / "tokenizers" / "punkt_tab"
        if not source.exists():
            raise RuntimeError(f"punkt_tab downloaded but {source} is missing")
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)

    return punkt_lang


def verify_offline(backend: str, model_name: str, punkt_lang: str) -> list[str]:
    failures: list[str] = []

    if backend == HUGGINGFACE:
        try:
            Wav2Vec2Processor.from_pretrained(
                model_name, cache_dir=config.models(), local_files_only=True
            )
            Wav2Vec2ForCTC.from_pretrained(
                model_name, cache_dir=config.models(), local_files_only=True
            )
            print(f"  ok: {model_name} loads from {config.models()} with no network")
        except Exception as exc:
            failures.append(f"HF aligner {model_name} not loadable offline: {exc}")
    else:
        bundle = torchaudio.pipelines.__dict__[model_name]
        expected = getattr(bundle, "_path", None)
        if expected is None:
            failures.append(
                f"torchaudio bundle {model_name} exposes no _path; cannot "
                f"check its checkpoint is staged"
            )
        elif not (checkpoint_dir() / expected).exists():
            failures.append(
                f"torchaudio checkpoint {expected} missing from {checkpoint_dir()}"
            )
        else:
            print(f"  ok: {expected} present in {checkpoint_dir()}")

    saved = list(nltk.data.path)
    try:
        nltk.data.path = [str(config.nltk_data())]
        nltk.data.find(f"tokenizers/punkt_tab/{punkt_lang}")
        print(f"  ok: punkt '{punkt_lang}' resolves from {config.nltk_data()}")
    except LookupError as exc:
        failures.append(f"punkt '{punkt_lang}' not found under {config.nltk_data()}: {exc}")
    finally:
        nltk.data.path = saved

    return failures


def stage_one(language: str, align_model: str | None, verify_only: bool) -> list[str]:
    print(f"\n=== {language}")
    try:
        if verify_only:
            backend, model_name = resolve_align_model(language, align_model)
            punkt_lang = PUNKT_LANGUAGES.get(language, "english")
        else:
            backend, model_name = stage_aligner(language, align_model)
            punkt_lang = stage_punkt(language)
    except Exception as exc:
        return [f"{language}: staging failed: {type(exc).__name__}: {exc}"]

    return [f"{language}: {f}" for f in
            verify_offline(backend, model_name, punkt_lang)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "language", nargs="?",
        help=f"code, one of: {', '.join(supported_languages())}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="stage every language whisperx has a default aligner for "
             "(41 languages, ~52 GB), continuing past any that fail",
    )
    parser.add_argument(
        "--align-model",
        help="explicit HuggingFace wav2vec2 repo id, for languages whisperx has no default for",
    )
    parser.add_argument(
        "--models",
        help=f"the model directory, when it is not configured already or "
             f"named by ${config.MODELS_ENV}",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check what is already staged without downloading",
    )
    args = parser.parse_args()
    if args.models:
        config.configure(args.models)

    if args.all == bool(args.language):
        parser.error("give a language code or --all, not both or neither")
    if args.all and args.align_model:
        parser.error("--align-model names one repo, so it cannot be used with --all")
    if not args.verify_only and os.environ.get("HF_HUB_OFFLINE") == "1":
        print("HF_HUB_OFFLINE=1 is set; unset it to download.", file=sys.stderr)
        return 2

    languages = supported_languages() if args.all else [args.language]
    verb = "verifying" if args.verify_only else "staging"
    print(f"{verb} {len(languages)} language(s) into {config.models()}")

    failures: list[str] = []
    done: list[str] = []
    for language in languages:
        bad = stage_one(language, args.align_model, args.verify_only)
        failures.extend(bad)
        if not bad:
            done.append(language)

    if len(languages) > 1:
        print(f"\n{'=' * 60}\n{len(done)}/{len(languages)} staged: "
              f"{', '.join(done) or '-'}")
        bad_langs = sorted({f.split(':')[0] for f in failures})
        if bad_langs:
            print(f"{len(bad_langs)} failed: {', '.join(bad_langs)}")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"\n{len(done)} language(s) staged; an offline job can now align them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
