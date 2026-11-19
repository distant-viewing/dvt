"""Third-party import noise."""

import contextlib
import logging
import os
import sys
import warnings
from collections.abc import Generator


def silence() -> None:
    if os.environ.get("TBM_NOISE"):
        return

    logging.getLogger("dinov2").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message=r"\s*torchcodec is not installed correctly",
        category=UserWarning,
    )

    warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider'")
    logging.getLogger("insightface").setLevel(logging.ERROR)

    # onnx2torch's Slice indexes with a list of slices, which torch 2.8 reads
    # as a tuple. 2.9 makes it an error, and pyproject holds torch below 2.9.
    warnings.filterwarnings(
        "ignore",
        message="Using a non-tuple sequence for multidimensional indexing",
        category=UserWarning,
        module=r"onnx2torch\.",
    )
    # insightface's norm_crop calls SimilarityTransform.estimate, deprecated in
    # scikit-image 0.26 and working until 2.2.
    warnings.filterwarnings(
        "ignore",
        message="`estimate` is deprecated",
        category=FutureWarning,
        module=r"insightface\.",
    )
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)

    if not sys.stderr.isatty():
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


_LOAD_REPORT_KEEP = ("MISSING", "MISMATCH", "CONVERSION")


class _UnexpectedOnlyReports(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "LOAD REPORT" not in message:
            return True
        return any(status in message for status in _LOAD_REPORT_KEEP)


@contextlib.contextmanager
def quiet_partial_load() -> Generator[None]:
    if os.environ.get("TBM_NOISE"):
        yield
        return

    log = logging.getLogger("transformers.modeling_utils")
    drop = _UnexpectedOnlyReports()
    log.addFilter(drop)
    try:
        yield
    finally:
        log.removeFilter(drop)


def quiet_loggers() -> None:
    if os.environ.get("TBM_NOISE"):
        return

    roots = ("whisperx", "lightning", "pytorch_lightning")

    names = set(roots) | {
        name
        for name in logging.root.manager.loggerDict
        if name.startswith(tuple(root + "." for root in roots))
    }
    for name in names:
        logging.getLogger(name).setLevel(logging.WARNING)
