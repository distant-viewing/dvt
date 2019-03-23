import os.path
import tempfile

import numpy as np
import cv2
import pytest

from dvt.annotate.core import FrameProcessor, FrameInput, ImageInput
from dvt.annotate.diff import DiffAnnotator
from dvt.annotate.embed import EmbedAnnotator, EmbedFrameKerasResNet50
from dvt.annotate.face import (
    FaceAnnotator,
    FaceDetectDlib,
    FaceEmbedDlib,
    FaceEmbedVgg2,
)
from dvt.annotate.meta import MetaAnnotator
from dvt.annotate.object import ObjectAnnotator, ObjectDetectRetinaNet
from dvt.annotate.png import PngAnnotator

from dvt.utils import DictFrame, get_batch


class TestDiffAnno:
    def test_no_quantiles(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator())

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("diff")

        assert set(obj_out.keys()) == set(["video", "frame", "avg_value"])
        assert issubclass(type(obj_out["avg_value"]), np.ndarray)

    def test_quantiles(self):

        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator(quantiles=[40, 50]))

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("diff")
        keys = list(obj_out.keys())

        assert issubclass(type(obj_out["q40"]), np.ndarray)
        assert issubclass(type(obj_out["q50"]), np.ndarray)
        assert issubclass(type(obj_out["h40"]), np.ndarray)
        assert issubclass(type(obj_out["h50"]), np.ndarray)


class TestEmbed:
    def test_embed_resnet(self):
        anno = EmbedAnnotator(freq=4, embedding=EmbedFrameKerasResNet50())
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("embed")

        assert set(obj_out.keys()) == set(["frame", "video", "embed"])
        assert issubclass(type(obj_out["embed"]), np.ndarray)
        assert obj_out["embed"].shape == (4, 2048)


class TestFace:
    def test_face_detector_only(self):
        anno = FaceAnnotator(detector=FaceDetectDlib(), freq=4)
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("face")

        expected_keys = [
            "video",
            "frame",
            "confidence",
            "top",
            "bottom",
            "left",
            "right",
        ]
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out.shape == (8, 7)

    def test_face_detector_cutoff(self):
        anno = FaceAnnotator(detector=FaceDetectDlib(cutoff=1.02), freq=4)
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("face")

        expected_keys = [
            "video",
            "frame",
            "confidence",
            "top",
            "bottom",
            "left",
            "right",
        ]
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out.shape == (7, 7)

    def test_face_dlib_embed(self):
        anno = FaceAnnotator(
            detector=FaceDetectDlib(), embedding=FaceEmbedDlib(), freq=4
        )
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("face")

        expected_keys = [
            "video",
            "frame",
            "confidence",
            "top",
            "bottom",
            "left",
            "right",
            "embed",
        ]
        assert set(obj_out.keys()) == set(expected_keys)
        assert issubclass(type(obj_out["embed"]), np.ndarray)
        assert obj_out["embed"].shape == (8, 128)

    def test_face_vgg2_embed(self):
        anno = FaceAnnotator(
            detector=FaceDetectDlib(), embedding=FaceEmbedVgg2(), freq=4
        )
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("face")

        expected_keys = [
            "video",
            "frame",
            "confidence",
            "top",
            "bottom",
            "left",
            "right",
            "embed",
        ]
        assert set(obj_out.keys()) == set(expected_keys)
        assert issubclass(type(obj_out["embed"]), np.ndarray)
        assert obj_out["embed"].shape == (8, 2048)

    def test_channels(self):

        finput = FrameInput("test-data/video-clip.mp4", bsize=1)
        batch = get_batch(finput, batch_num=0)
        img = batch.get_frames()[0, :, :, :]
        face = DictFrame({"top": 0, "bottom": 96, "left": 0, "right": 96})

        femb = FaceEmbedVgg2()

        femb._iformat = "channels_first"
        emb1 = femb.embed(img, face)
        femb._iformat = "channels_last"
        emb2 = femb.embed(img, face)

        assert (emb1 != emb2).any()


class TestObject:
    def test_object_detection(self):
        anno = ObjectAnnotator(freq=4, detector=ObjectDetectRetinaNet())
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("object")

        expected_keys = [
            "video",
            "frame",
            "score",
            "top",
            "bottom",
            "left",
            "right",
            "class",
        ]
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out.shape == (17, 8)

    def test_object_detection_cutoff(self):
        anno = ObjectAnnotator(freq=4, detector=ObjectDetectRetinaNet(cutoff=0.6))
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("object")

        assert obj_out.shape == (12, 8)


class TestPng:
    def test_png_existing_dir(self):
        dname = tempfile.mkdtemp()  # creates directory

        fpobj = FrameProcessor()
        fpobj.load_annotator(PngAnnotator(output_dir=dname))

        finput = FrameInput("test-data/video-clip.mp4", bsize=4)
        fpobj.process(finput, max_batch=2)

        expected_files = set(["frame-{0:06d}.png".format(x) for x in range(8)])
        obj_out = fpobj.collect("png")
        assert obj_out == DictFrame()
        assert set(os.listdir(dname)) == expected_files

    def test_png_new_dir(self):
        dname = tempfile.mkdtemp()  # creates directory
        dname = os.path.join(dname, "temp2")  # this directory will not exist

        fpobj = FrameProcessor()
        fpobj.load_annotator(PngAnnotator(output_dir=dname))

        finput = FrameInput("test-data/video-clip.mp4", bsize=4)
        fpobj.process(finput, max_batch=2)

        expected_files = set(["frame-{0:06d}.png".format(x) for x in range(8)])
        obj_out = fpobj.collect("png")
        assert obj_out == DictFrame()
        assert set(os.listdir(dname)) == expected_files

    def test_png_image(self):
        dname = tempfile.mkdtemp()  # creates directory

        fpobj = FrameProcessor()
        fpobj.load_annotator(PngAnnotator(output_dir=dname))

        iobj = ImageInput(
            input_paths=[
                "test-data/img/frame-000076.png",
                "test-data/img/frame-000506.png",
            ]
        )
        fpobj.process(iobj, max_batch=2)

        # assert that all of the images in the input exist in output
        expected_files = set(["frame-000076.png", "frame-000506.png"])
        obj_out = fpobj.collect("png")
        assert obj_out == DictFrame()
        assert set(os.listdir(dname)) == expected_files

        # make sure images are the same
        img1 = cv2.imread(os.path.join("test-data/img/frame-000076.png"))
        img2 = cv2.imread(os.path.join(dname, "frame-000076.png"))
        assert np.all(img1 == img2)


class TestMeta:

    def test_meta_output_video(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(MetaAnnotator())

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect("meta")

        expected_keys = ['width', 'type', 'vname', 'fps', 'height', 'frames']
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out['width'] == [708]
        assert obj_out['height'] == [480]
        assert obj_out['type'] == ['video']
        assert obj_out['vname'] == ['video-clip.mp4']
        assert obj_out['fps'] == [29.97002997002997]
        assert obj_out['frames'] == [379]

    def test_meta_output_images(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(MetaAnnotator())

        iobj = ImageInput(
            input_paths=[
                "test-data/img/frame-000076.png",
                "test-data/img/frame-000506.png",
            ]
        )
        fpobj.process(iobj, max_batch=2)
        obj_out = fpobj.collect("meta")

        expected_keys = ['width', 'type', 'vname', 'height']
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out['width'] == [-1]
        assert obj_out['height'] == [-1]
        assert obj_out['type'] == ['image']
        assert obj_out['vname'] == ['']


if __name__ == "__main__":
    pytest.main([__file__])
