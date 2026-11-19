import numpy as np
from numpy.typing import NDArray

def distance2bbox(
    points: NDArray[np.float32],
    distance: NDArray[np.float32],
    max_shape: tuple[int, int] | None = None,
) -> NDArray[np.float32]: ...
def distance2kps(
    points: NDArray[np.float32],
    distance: NDArray[np.float32],
    max_shape: tuple[int, int] | None = None,
) -> NDArray[np.float32]: ...
