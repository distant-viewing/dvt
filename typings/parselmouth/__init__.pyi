import numpy as np
from numpy.typing import NDArray

from . import praat as praat

class Data: ...

class TimeFrameSampled(Data):
    t1: float
    dt: float
    n_frames: int

class Pitch(TimeFrameSampled):
    # A record per frame; "frequency" is 0 where the frame is unvoiced.
    selected_array: NDArray[np.void]

class Intensity(TimeFrameSampled):
    values: NDArray[np.float64]

class Harmonicity(TimeFrameSampled):
    values: NDArray[np.float64]

class Spectrogram(TimeFrameSampled):
    values: NDArray[np.float64]
    y1: float
    dy: float

class Formant(TimeFrameSampled):
    def get_value_at_time(self, formant_number: int, time: float) -> float: ...
    def get_bandwidth_at_time(self, formant_number: int, time: float) -> float: ...

class Sound(TimeFrameSampled):
    def __init__(
        self, values: NDArray[np.float64], sampling_frequency: float = 44100.0, start_time: float = 0.0
    ) -> None: ...
    def to_pitch_ac(
        self, time_step: float | None = None, pitch_floor: float = 75.0, pitch_ceiling: float = 600.0
    ) -> Pitch: ...
    def to_intensity(
        self, minimum_pitch: float = 100.0, time_step: float | None = None
    ) -> Intensity: ...
    def to_harmonicity_cc(
        self, time_step: float = 0.01, minimum_pitch: float = 75.0
    ) -> Harmonicity: ...
    def to_formant_burg(
        self,
        time_step: float | None = None,
        max_number_of_formants: float = 5.0,
        maximum_formant: float = 5500.0,
    ) -> Formant: ...
    def to_spectrogram(
        self,
        window_length: float = 0.005,
        maximum_frequency: float = 5000.0,
        time_step: float = 0.002,
    ) -> Spectrogram: ...
