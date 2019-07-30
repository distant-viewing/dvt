import numpy as np
import pytest

from dvt.annotate.core import (
    FrameProcessor,
    FrameInput,
    ImageInput,
    FrameAnnotator,
)
from dvt.annotate.diff import DiffAnnotator
from dvt.utils import DictFrame


class TestImageInput:
    def test_length_input_glob(self):
        iobj = ImageInput(input_paths="test-data/img/*")
        assert len(iobj.paths) == 12

    def test_manual_paths(self):
        iobj = ImageInput(
            input_paths=[
                "test-data/img/frame-000076.png",
                "test-data/img/frame-000506.png",
            ]
        )
        assert len(iobj.paths) == 2

    def test_remove_bad_path(self):
        iobj = ImageInput(
            input_paths=["test-data/img/frame-000076.png", "bad-file-name.png"]
        )
        assert len(iobj.paths) == 1

    def test_frames(self):
        iobj = ImageInput(input_paths=["test-data/img/*"])
        batch = iobj.next_batch()
        frames = batch.get_frames()

        assert frames.shape[0] == 2
        assert np.max(frames[1, :, :, :]) == 0

    def test_vname(self):
        iobj = ImageInput(input_paths=["test-data/img/*"], vname="something")
        batch = iobj.next_batch()

        assert batch.vname == "something"

    def test_continue_bnum(self):
        iobj = ImageInput(
            input_paths=[
                "test-data/img/frame-000076.png",
                "test-data/img/frame-000506.png",
            ]
        )

        batch = iobj.next_batch()
        assert batch.bnum == 0
        assert iobj.continue_read

        batch = iobj.next_batch()
        assert batch.bnum == 1
        assert not iobj.continue_read

    def test_get_frame_names(self):
        iobj = ImageInput(
            input_paths=[
                "test-data/img/frame-000076.png",
                "test-data/img/frame-000506.png",
            ]
        )

        batch = iobj.next_batch()
        fnames = batch.get_frame_names()
        assert fnames == ["test-data/img/frame-000076.png"]

        batch = iobj.next_batch()
        fnames = batch.get_frame_names()
        assert fnames == ["test-data/img/frame-000506.png"]


class TestFrameInput:
    def test_metadata(self):
        fobj = FrameInput("test-data/video-clip.mp4")

        assert fobj.meta["type"] == "video"
        assert fobj.meta["fps"] == 29.97002997002997
        assert fobj.meta["frames"] == 379
        assert fobj.meta["height"] == 480
        assert fobj.meta["width"] == 708
        assert fobj.vname == "video-clip.mp4"

    def test_batch_bsize(self):
        fobj = FrameInput("test-data/video-clip.mp4", bsize=16)
        batch = fobj.next_batch()
        frames = batch.get_batch()

        assert batch.bnum == 0
        assert fobj.continue_read
        assert frames.shape[0] == 16

    def test_continue_bnum(self):
        fobj = FrameInput("test-data/video-clip.mp4", bsize=256)

        batch = fobj.next_batch()
        assert batch.bnum == 0
        assert fobj.continue_read

        batch = fobj.next_batch()
        assert batch.bnum == 1
        assert not fobj.continue_read

        frames = batch.get_frames()
        bwidth = frames[batch.bsize :, :, :, :]
        assert np.max(bwidth) == 0

    def test_get_frame_names(self):
        fobj = FrameInput("test-data/video-clip.mp4", bsize=4)

        batch = fobj.next_batch()
        batch = fobj.next_batch()
        fnames = batch.get_frame_names()

        assert fnames == [4, 5, 6, 7]


class TestFrameBatch:
    def test_batch_size(self):
        fobj = FrameInput("test-data/video-clip.mp4", bsize=4)

        batch = fobj.next_batch()
        bframes = batch.get_batch()
        assert bframes.shape[0] == 4
        assert batch.bsize == 4

    def test_bandwidh_size(self):
        fobj = FrameInput("test-data/video-clip.mp4", bsize=4)

        batch = fobj.next_batch()
        frames = batch.get_frames()
        assert frames.shape[0] == 8


class TestFrameAnnotator:
    def test_clear_cache(self):

        fanno = FrameAnnotator()
        fanno.cache = {"key": "value"}
        fanno.clear()

        assert fanno.cache == {}

    def test_default_return(self):

        fanno = FrameAnnotator()

        assert fanno.annotate(None) is None


class TestFrameProcessor:
    def test_input_pipeline_empty(self):
        fpobj = FrameProcessor()
        assert fpobj.pipeline == {}

    def test_input_pipeline(self):
        fanno = FrameAnnotator()
        fpobj = FrameProcessor({"base": fanno})

        pline = fpobj.pipeline
        assert list(pline.keys()) == ["base"]
        assert issubclass(type(list(pline.values())[0]), FrameAnnotator)

    def test_load_annotators(self):
        fpobj = FrameProcessor()

        fpobj.load_annotator(FrameAnnotator())
        pline = fpobj.pipeline
        assert len(pline) == 1
        assert list(pline.keys()) == ["base"]
        assert issubclass(type(list(pline.values())[0]), FrameAnnotator)

        fobj = FrameAnnotator()
        fobj.name = "other"
        fpobj.load_annotator(fobj)

        pline = fpobj.pipeline
        assert len(pline) == 2
        assert list(pline.keys()) == ["base", "other"]

    def test_clear(self):
        fpobj = FrameProcessor()

        fpobj.load_annotator(FrameAnnotator())
        fpobj.clear()

        assert fpobj.pipeline == {}

    def test_process_empty_output(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(FrameAnnotator())

        finput = FrameInput("test-data/video-clip.mp4", bsize=4)
        fpobj.process(finput)

        assert fpobj.collect("base") == DictFrame()

    def test_process_full_output(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator())

        finput = FrameInput("test-data/video-clip.mp4", bsize=128)
        fpobj.process(finput)

        obj_diff = fpobj.collect("diff")
        assert obj_diff.shape[0] == (128 * 3)

    def test_max_batch(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator())

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=3)

        obj_diff = fpobj.collect("diff")
        assert obj_diff.shape[0] == (3 * 8)

    def test_collect_all(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(FrameAnnotator())
        fpobj.load_annotator(DiffAnnotator())

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)

        output = fpobj.collect_all()

        assert output["base"] == DictFrame()
        assert output["diff"].shape[0] == (2 * 8)


if __name__ == "__main__":
    pytest.main([__file__])
