"""dvt PATH --out DIR: every table for one video, image or directory of images."""

import argparse
import sys
from pathlib import Path

# Nothing imported here loads torch or any other library an extra brings:
# the command has to be able to say that one is missing.
from . import annotate, config, fetch
from .annotators import ANNOTATORS, IMAGE_ANNOTATORS


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["models"]:
        return fetch.main(argv[1:])

    ap = argparse.ArgumentParser(
        prog="dvt",
        description="Run the annotators over one video, one image or a directory of images.",
        epilog="`dvt models fetch` downloads the models the annotators load.",
    )
    ap.add_argument(
        "path", type=Path, nargs="?", help="the video, the image or the directory of images"
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="where everything made goes: annotations/<name>/ holds the "
        "tables and, for a video, audio/ the sound, frames/ the saved "
        "frames and stems/ the separated stems",
    )
    ap.add_argument(
        "--models",
        type=Path,
        help=f"the model directory; ${config.MODELS_ENV} if not given, "
        f"and {config.DEFAULT_MODELS} if that is not set",
    )
    ap.add_argument(
        "--tmp",
        type=Path,
        help="scratch space for the separation, a few GB a film; the system's by default",
    )
    ap.add_argument(
        "--annotators",
        help="comma-separated names to run, all by default: "
        + ", ".join(ANNOTATORS)
        + ". An image has "
        + ", ".join(IMAGE_ANNOTATORS),
    )
    ap.add_argument(
        "--list", action="store_true", help="list the annotators and what each writes, and stop"
    )
    args = ap.parse_args(argv)

    if args.list:
        for name, est in ANNOTATORS.items():
            needs = f"  after {', '.join(est.requires)}" if est.requires else ""
            print(f"{name:<13} {est.side:<9} {', '.join(est.outputs)}{needs}")
        return 0

    names = None
    if args.annotators:
        names = [n.strip() for n in args.annotators.split(",") if n.strip()]
    if args.path is None or args.out is None:
        ap.error("a path and --out are both needed")
    config.configure(args.models if args.models is not None else config.models(), args.tmp)
    try:
        annotate(args.path, args.out, names)
    except ModuleNotFoundError as e:
        ap.error(
            f"{e.name} is not installed. The annotators come with the extras of "
            "this package: video, face, yolo, audio or all, as in "
            'pip install "dvt[all]".'
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        ap.error(f"{type(e).__name__}: {e}" if isinstance(e, FileNotFoundError) else str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
