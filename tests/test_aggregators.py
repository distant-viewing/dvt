import pytest

import os.path
import tempfile


from dvt.annotate.core import FrameProcessor, FrameInput
from dvt.annotate.diff import DiffAnnotator
from dvt.annotate.face import FaceAnnotator, FaceDetectDlib, FaceEmbedVgg2
from dvt.annotate.meta import MetaAnnotator
from dvt.annotate.obj import ObjectAnnotator, ObjectDetectRetinaNet
from dvt.annotate.png import PngAnnotator

from dvt.aggregate.core import Aggregator
from dvt.aggregate.cut import CutAggregator
from dvt.aggregate.display import DisplayAggregator
from dvt.aggregate.length import ShotLengthAggregator
from dvt.aggregate.people import PeopleAggregator

from dvt.utils import DictFrame


class TestPeopleAggregator:
    def test_fprint(self):
        face_anno = FaceAnnotator(
            detector=FaceDetectDlib(), embedding=FaceEmbedVgg2(), freq=4
        )

        fpobj = FrameProcessor()
        fpobj.load_annotator(face_anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect_all()

        pa = PeopleAggregator(
            face_names=["person 1", "person 2"],
            fprint=obj_out["face"]["embed"][[0, 1]],
        )
        agg = pa.aggregate(obj_out).todf()

        assert set(agg.keys()) == set(
            [
                "video",
                "frame",
                "top",
                "bottom",
                "right",
                "left",
                "confidence",
                "person",
                "person-dist",
            ]
        )


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


class TestShotLengthAggregator:
    def test_lengths(self):
        fp = FrameProcessor()
        fp.load_annotator(MetaAnnotator())
        fp.load_annotator(FaceAnnotator(detector=FaceDetectDlib(), freq=128))
        fp.load_annotator(
            ObjectAnnotator(detector=ObjectDetectRetinaNet(), freq=128)
        )

        fp.process(FrameInput("test-data/video-clip.mp4"), max_batch=2)
        obj = fp.collect_all()

        sla = ShotLengthAggregator()
        agg = sla.aggregate(obj)
        pdf = agg.todf()

        assert set(agg.keys()) == set(
            [
                "frame",
                "num_faces",
                "num_people",
                "largest_face",
                "largest_body",
                "shot_length",
                "objects",
            ]
        )
        assert agg["shot_length"] == ["5-MCU", "3-MLS", "5-MCU"]


class TestDisplayAggregator:
    def test_lengths(self):
        dname_png = os.path.join(tempfile.mkdtemp(), "png")
        dname_dis = os.path.join(tempfile.mkdtemp(), "dis")

        fp = FrameProcessor()
        fp.load_annotator(PngAnnotator(output_dir=dname_png, freq=128))
        fp.load_annotator(FaceAnnotator(detector=FaceDetectDlib(), freq=128))
        fp.load_annotator(
            ObjectAnnotator(detector=ObjectDetectRetinaNet(), freq=128)
        )

        fp.process(FrameInput("test-data/video-clip.mp4"), max_batch=2)
        obj = fp.collect_all()

        da = DisplayAggregator(input_dir=dname_png, output_dir=dname_dis)
        agg = da.aggregate(obj)
        assert agg == None

        # should have a total of four original png files
        expected_files = set(
            ["frame-{0:06d}.png".format(x) for x in [0, 128, 256, 384]]
        )
        assert set(os.listdir(dname_png)) == expected_files

        # should have only three new png files because, be default, only frames
        # with detected information have data; this can be changed this by
        # providing a the desired list of frames as argument to the aggregate
        # method of DisplayAggregator
        expected_files = set(
            ["frame-{0:06d}.png".format(x) for x in [0, 128, 256]]
        )
        assert set(os.listdir(dname_dis)) == expected_files


if __name__ == "__main__":
    pytest.main([__file__])
