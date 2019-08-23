import pytest

import os.path
import tempfile

from pandas import DataFrame
from pandas.util.testing import assert_frame_equal


class TestAnnotatorVideoFreqOutput:

    def test_video_freq_cielab(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["cielab"]

        assert set(obj.keys()) == set(
            ["frame", "cielab", "dominant_colors"]
        )
        assert len(obj["cielab"].values[0]) == 4096
        assert len(obj["dominant_colors"].values[0]) == 5

    def test_video_freq_diff(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["diff"]

        assert set(obj.keys()) == set([
            "frame", "avg_value", "q40", "h40"
        ])

    def test_video_freq_embed(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["embed"]

        assert set(obj.keys()) == set(
            ["frame", "embed"]
        )
        assert len(obj["embed"].values[0]) == 2048

    def test_video_freq_face(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["face"]

        assert set(obj.keys()) == set([
                "frame",
                "confidence",
                "top",
                "bottom",
                "left",
                "right",
                "embed"
        ])

    def test_video_freq_hofm(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["hofm"]

        assert set(obj.keys()) == set(["frame", "hofm"])

    def test_video_freq_obj(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["obj"]

        assert set(obj.keys()) == set([
            "frame",
            "score",
            "top",
            "bottom",
            "left",
            "right",
            "category",
        ])

    def test_video_freq_opticalflow(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["opticalflow"]

        assert set(obj.keys()) == set(["frame", "opticalflow"])

    def test_video_freq_png(self, get_video_annotation):
        de, _ = get_video_annotation
        obj = de.get_data()["png"]

        assert_frame_equal(obj, DataFrame())


class TestAnnotatorVideoFrameOutput:

    def test_video_frame_cielab(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["cielab"]

        assert set(obj.keys()) == set(
            ["frame", "cielab", "dominant_colors"]
        )
        assert len(obj["cielab"].values[0]) == 4096
        assert len(obj["dominant_colors"].values[0]) == 5

    def test_video_frame_diff(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["diff"]

        assert set(obj.keys()) == set([
            "frame", "avg_value", "q40", "h40"
        ])

    def test_video_frame_embed(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["embed"]

        assert set(obj.keys()) == set(
            ["frame", "embed"]
        )
        assert len(obj["embed"].values[0]) == 2048

    def test_video_frame_face(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["face"]

        assert set(obj.keys()) == set([
                "frame",
                "confidence",
                "top",
                "bottom",
                "left",
                "right",
                "embed"
        ])

    def test_video_frame_hofm(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["hofm"]

        assert set(obj.keys()) == set(["frame", "hofm"])

    def test_video_frame_obj(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["obj"]

        assert set(obj.keys()) == set([
            "frame",
            "score",
            "top",
            "bottom",
            "left",
            "right",
            "category",
        ])

    def test_video_frame_opticalflow(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["opticalflow"]

        assert set(obj.keys()) == set(["frame", "opticalflow"])

    def test_video_frame_png(self, get_video_frame_annotation):
        de, _ = get_video_frame_annotation
        obj = de.get_data()["png"]

        assert_frame_equal(obj, DataFrame())


class TestAnnotatorImageOutput:

    def test_video_frame_cielab(self, get_image_annotation):
        de, _ = get_image_annotation
        obj = de.get_data()["cielab"]

        assert set(obj.keys()) == set(
            ["frame", "cielab", "dominant_colors"]
        )
        assert len(obj["cielab"].values[0]) == 4096
        assert len(obj["dominant_colors"].values[0]) == 5

    def test_video_frame_embed(self, get_image_annotation):
        de, _ = get_image_annotation
        obj = de.get_data()["embed"]

        assert set(obj.keys()) == set(
            ["frame", "embed"]
        )
        assert len(obj["embed"].values[0]) == 2048

    def test_video_frame_face(self, get_image_annotation):
        de, _ = get_image_annotation
        obj = de.get_data()["face"]

        assert set(obj.keys()) == set([
                "frame",
                "confidence",
                "top",
                "bottom",
                "left",
                "right",
                "embed"
        ])

    def test_video_frame_obj(self, get_image_annotation):
        de, _ = get_image_annotation
        obj = de.get_data()["obj"]

        assert set(obj.keys()) == set([
            "frame",
            "score",
            "top",
            "bottom",
            "left",
            "right",
            "category",
        ])

    def test_video_frame_png(self, get_image_annotation):
        de, _ = get_image_annotation
        obj = de.get_data()["png"]

        assert_frame_equal(obj, DataFrame())
