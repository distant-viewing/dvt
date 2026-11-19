from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kalman_filter import KalmanFilter


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    score = 0.0
    start_frame = 0
    frame_id = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def reset_ids() -> None:
        """Start the track numbers from one again."""
        BaseTrack._count = 0

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, kalman_filter: "KalmanFilter", frame_id: int) -> None:
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed
