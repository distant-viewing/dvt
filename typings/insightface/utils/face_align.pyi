import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

def norm_crop(
    img: MatLike,
    landmark: NDArray[np.float32],
    image_size: int = 112,
    mode: str = "arcface",
) -> MatLike: ...
