import pytest

from dvt.annotate.core import FrameProcessor, FrameInput
from dvt.annotate.diff import DiffAnnotator
from dvt.aggregate.core import Aggregator
from dvt.aggregate.cut import CutAggregator

from dvt.utils import DictFrame


class TestBase:
    def test_base(self):
        assert Aggregator().aggregate(None) is None


class TestCutAggregator:
    def test_cutoff(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator(quantiles=[40]))

        finput = FrameInput("test-data/video-clip.mp4", bsize=128)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect_all()

        ca = CutAggregator(cut_vals={"h40": 0.2, "q40": 3})
        agg = ca.aggregate(obj_out)

        assert set(agg.keys()) == set(["video", "frame_start", "frame_end"])

    def test_cutoff_empty(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator(quantiles=[40]))

        finput = FrameInput("test-data/video-clip.mp4", bsize=128)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect_all()

        ca = CutAggregator()
        agg = ca.aggregate(obj_out)

        assert agg["frame_start"] == list(range(256))

    def test_cutoff_ignore(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator(quantiles=[40]))

        finput = FrameInput("test-data/video-clip.mp4", bsize=128)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect_all()

        ca = CutAggregator(
            cut_vals={"h40": 0.2, "q40": 3}, ignore_vals={"avg_value": 70}
        )
        agg = ca.aggregate(obj_out)

        assert agg == DictFrame()


if __name__ == "__main__":
    pytest.main([__file__])
