from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import matching
from .basetrack import BaseTrack, TrackState
from .kalman_filter import Floats, KalmanFilter


@dataclass(frozen=True)
class TrackerArgs:
    """What BYTETracker is set up with. embed_gate, when set, keeps a track
    and a detection apart once their embeddings are further than it."""

    track_thresh: float
    track_buffer: int
    match_thresh: float
    mot20: bool = False
    embed_gate: float | None = None


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: ArrayLike, score: float, feat: ArrayLike | None = None) -> None:
        self._tlwh: Floats = np.asarray(tlwh, dtype=float)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: Floats | None = None
        self.covariance: Floats | None = None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.curr_feat: Floats | None = None
        self.smooth_feat: Floats | None = None
        self.feat_alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: ArrayLike) -> None:
        raw: Floats = np.asarray(feat, dtype=float)
        unit = raw / max(float(np.linalg.norm(raw)), 1e-12)
        self.curr_feat = unit
        if self.smooth_feat is None:
            self.smooth_feat = unit
        else:
            smooth = self.feat_alpha * self.smooth_feat + (1 - self.feat_alpha) * unit
            smooth /= max(float(np.linalg.norm(smooth)), 1e-12)
            self.smooth_feat = smooth

    @property
    def kf(self) -> KalmanFilter:
        """The filter activate gave the track."""
        if self.kalman_filter is None:
            raise AttributeError("the track has no Kalman filter until it is activated")
        return self.kalman_filter

    def filter_state(self) -> tuple[Floats, Floats]:
        """The mean and covariance activate gave the track."""
        if self.mean is None or self.covariance is None:
            raise AttributeError("the track has no state until it is activated")
        return self.mean, self.covariance

    def predict(self) -> None:
        mean, covariance = self.filter_state()
        mean_state = mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kf.predict(mean_state, covariance)

    @staticmethod
    def multi_predict(stracks: Sequence["STrack"]) -> None:
        if len(stracks) > 0:
            states = [st.filter_state() for st in stracks]
            multi_mean: Floats = np.asarray([m.copy() for m, _ in states])
            multi_covariance: Floats = np.asarray([c for _, c in states])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance, strict=True)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False) -> None:
        mean, covariance = self.filter_state()
        self.mean, self.covariance = self.kf.update(
            mean, covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track: Self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        mean, covariance = self.filter_state()
        self.mean, self.covariance = self.kf.update(mean, covariance, self.tlwh_to_xyah(new_tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self) -> Floats:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> Floats:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: ArrayLike) -> Floats:
        ret: Floats = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> Floats:
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr: ArrayLike) -> Floats:
        ret: Floats = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: ArrayLike) -> Floats:
        ret: Floats = np.asarray(tlwh, dtype=np.float64).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    def __init__(self, args: TrackerArgs, frame_rate: int = 30) -> None:
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.embed_gate = args.embed_gate
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def gate(self, dists: Floats, tracks: Sequence[STrack], detections: Sequence[STrack]) -> Floats:
        if self.embed_gate is None or dists.size == 0:
            return dists
        dists = dists.copy()
        for i, t in enumerate(tracks):
            if t.smooth_feat is None:
                continue
            for j, d in enumerate(detections):
                if d.curr_feat is None:
                    continue
                if 1.0 - float(np.dot(t.smooth_feat, d.curr_feat)) > self.embed_gate:
                    dists[i, j] = 1.0
        return dists

    def update(
        self,
        output_results: Floats,
        img_info: tuple[int, int],
        img_size: tuple[int, int],
        feats: Sequence[Floats | None] | None = None,
    ) -> list[STrack]:
        """The tracks after one frame's detections, rows of x1, y1, x2, y2
        and score, with the embedding of each where it has one."""
        self.frame_id += 1
        activated_starcks: list[STrack] = []
        refind_stracks: list[STrack] = []
        lost_stracks: list[STrack] = []
        removed_stracks: list[STrack] = []

        if output_results.shape[1] != 5:
            raise ValueError("each detection is x1, y1, x2, y2 and a score")
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        each: NDArray[np.object_] = np.asarray(
            [None] * len(scores) if feats is None else feats, dtype=object
        )

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        feats_keep = each[remain_inds]
        feats_second = each[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, f)
                for (tlbr, s, f) in zip(dets, scores_keep, feats_keep, strict=True)
            ]
        else:
            detections: list[STrack] = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed: list[STrack] = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        dists = self.gate(dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in _pairs(matches):
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, f)
                for (tlbr, s, f) in zip(dets_second, scores_second, feats_second, strict=True)
            ]
        else:
            detections_second: list[STrack] = []
        r_tracked_stracks = [
            strack_pool[i] for i in _ints(u_track) if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = self.gate(dists, r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in _pairs(matches):
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in _ints(u_track):
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in _ints(u_detection)]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        dists = self.gate(dists, unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in _pairs(matches):
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in _ints(u_unconfirmed):
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in _ints(u_detection):
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        return [track for track in self.tracked_stracks if track.is_activated]


def _ints(indices: matching.Indices) -> list[int]:
    """Row or column numbers as ints, to index the track lists by."""
    return [int(i) for i in indices]


def _pairs(matches: NDArray[np.integer]) -> list[tuple[int, int]]:
    """The (track, detection) pairs linear_assignment matched."""
    return [(int(a), int(b)) for a, b in matches]


def joint_stracks(tlista: Sequence[STrack], tlistb: Sequence[STrack]) -> list[STrack]:
    exists: dict[int, int] = {}
    res: list[STrack] = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: Sequence[STrack], tlistb: Sequence[STrack]) -> list[STrack]:
    stracks: dict[int, STrack] = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(
    stracksa: Sequence[STrack], stracksb: Sequence[STrack]
) -> tuple[list[STrack], list[STrack]]:
    pdist = matching.iou_distance(stracksa, stracksb)
    rows, cols = np.where(pdist < 0.15)
    dupa: list[int] = []
    dupb: list[int] = []
    for p, q in zip(_ints(rows), _ints(cols), strict=True):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
