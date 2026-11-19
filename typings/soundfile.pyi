from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

class _SoundFileInfo:
    samplerate: int
    frames: int
    channels: int
    duration: float

def info(file: str, verbose: bool = False) -> _SoundFileInfo: ...
@overload
def read(
    file: str,
    frames: int = -1,
    start: int = 0,
    stop: int | None = None,
    dtype: Literal["float64"] = "float64",
    always_2d: bool = False,
) -> tuple[NDArray[np.float64], int]: ...
@overload
def read(
    file: str,
    frames: int = -1,
    start: int = 0,
    stop: int | None = None,
    *,
    dtype: Literal["float32"],
    always_2d: bool = False,
) -> tuple[NDArray[np.float32], int]: ...
def write(
    file: str, data: ArrayLike, samplerate: int, subtype: str | None = None
) -> None: ...

class SoundFile:
    def __init__(
        self,
        file: str,
        mode: str = "r",
        samplerate: int | None = None,
        channels: int | None = None,
        subtype: str | None = None,
    ) -> None: ...
    def write(self, data: ArrayLike) -> None: ...
    def close(self) -> None: ...
