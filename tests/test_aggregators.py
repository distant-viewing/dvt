import pytest

import os.path
import tempfile

from numpy import array
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from dvt.aggregate.cut import CutAggregator
from dvt.aggregate.display import DisplayAggregator
from dvt.aggregate.length import ShotLengthAggregator
from dvt.aggregate.people import PeopleAggregator, make_fprint_from_images
from dvt.utils import pd_col_asarray

class TestPeopleAggregator:

    def test_fprint(self, get_video_annotation):
        de, _ = get_video_annotation

        fprint = pd_col_asarray(de.get_data()["face"], "embed")[[0, 1]]
        pa = PeopleAggregator(
            face_names=["person 1", "person 2"],
            fprint=fprint,
        )
        de.run_aggregator(pa)
        obj = de.get_data()["people"]

        assert set(obj.keys()) == set(
            [
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

    def test_fprint_from_img(self, get_video_annotation):
        de, _ = get_video_annotation

        embed, fnames = make_fprint_from_images("test-data/faces")
        pa = PeopleAggregator(
            face_names=fnames,
            fprint=embed,
        )
        de.run_aggregator(pa)
        obj = de.get_data()["people"]

        assert set(obj.keys()) == set(
            [
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


class TestDisplayAggregator:

    def test_display(self, get_video_annotation):
        de, dname_root = get_video_annotation
        dname_png = os.path.join(dname_root, "png")
        dname_dis = os.path.join(tempfile.mkdtemp(), "dis")

        da = DisplayAggregator(input_dir=dname_png, output_dir=dname_dis)
        de.run_aggregator(da)
        obj = de.get_data()["display"]

        assert_frame_equal(obj, DataFrame())


class TestShotLengthAggregator:

    def test_shot_length(self, get_video_annotation):
        de, _ = get_video_annotation

        sla = ShotLengthAggregator()
        de.run_aggregator(sla)
        obj = de.get_data()["length"]

        assert set(obj.keys()) == set(
            [
                "frame",
                "num_faces",
                "num_people",
                "largest_face",
                "largest_body",
                "shot_length",
                "objects",
                "people"
            ]
        )


class TestCutAggregator:

    def test_shot_length(self, get_video_annotation):
        de, _ = get_video_annotation

        ca = CutAggregator(cut_vals={"q40": 2})
        de.run_aggregator(ca)
        obj = de.get_data()["cut"]

        assert set(obj.keys()) == set(["frame_start", "frame_end"])
