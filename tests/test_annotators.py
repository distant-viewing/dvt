import os.path
import tempfile

import numpy as np
import cv2
import pytest

from dvt.annotate.core import FrameProcessor, FrameInput, ImageInput
from dvt.annotate.diff import DiffAnnotator
from dvt.annotate.embed import EmbedAnnotator, EmbedFrameKerasResNet50
from dvt.annotate.face import FaceAnnotator, FaceDetectDlib, FaceEmbedDlib, FaceEmbedVgg2
from dvt.annotate.object import ObjectAnnotator, ObjectDetectRetinaNet
from dvt.annotate.png import PngAnnotator

from dvt.utils import DictFrame

class TestDiffAnno:

    def test_no_quantiles(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator())

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('diff')

        assert set(obj_out.keys()) == set(['video', 'frame', 'avg_value'])
        assert issubclass(type(obj_out['avg_value']), np.ndarray)

    def test_quantiles(self):

        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator(quantiles=[40, 50]))

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('diff')
        keys = list(obj_out.keys())

        assert issubclass(type(obj_out['q40']), np.ndarray)
        assert issubclass(type(obj_out['q50']), np.ndarray)
        assert issubclass(type(obj_out['h40']), np.ndarray)
        assert issubclass(type(obj_out['h50']), np.ndarray)


class TestEmbed:

    def test_embed_resnet(self):
        anno = EmbedAnnotator(freq=4, embedding=EmbedFrameKerasResNet50())
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('embed')

        assert set(obj_out.keys()) == set(['frame', 'video', 'embed'])
        assert issubclass(type(obj_out['embed']), np.ndarray)
        assert obj_out['embed'].shape == (4, 2048)


class TestFace:

    def test_face_detector_only(self):
        anno = FaceAnnotator(detector=FaceDetectDlib(), freq=4)
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('face')

        expected_keys = ['video', 'frame', 'confidence', 'top', 'bottom',
                         'left', 'right']
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out.shape == (8, 7)

    def test_face_detector_cutoff(self):
        anno = FaceAnnotator(detector=FaceDetectDlib(cutoff=1.02), freq=4)
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('face')

        expected_keys = ['video', 'frame', 'confidence', 'top', 'bottom',
                         'left', 'right']
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out.shape == (7, 7)

    def test_face_dlib_embed(self):
        anno = FaceAnnotator(detector=FaceDetectDlib(),
                             embedding=FaceEmbedDlib(), freq=4)
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('face')

        expected_keys = ['video', 'frame', 'confidence', 'top', 'bottom',
                         'left', 'right', 'embed']
        assert set(obj_out.keys()) == set(expected_keys)
        assert issubclass(type(obj_out['embed']), np.ndarray)
        assert obj_out['embed'].shape == (8, 128)

    def test_face_vgg2_embed(self):
        anno = FaceAnnotator(detector=FaceDetectDlib(),
                             embedding=FaceEmbedVgg2(), freq=4)
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('face')

        expected_keys = ['video', 'frame', 'confidence', 'top', 'bottom',
                         'left', 'right', 'embed']
        assert set(obj_out.keys()) == set(expected_keys)
        assert issubclass(type(obj_out['embed']), np.ndarray)
        assert obj_out['embed'].shape == (8, 2048)


class TestObject:

    def test_object_detection(self):
        anno = ObjectAnnotator(freq=4, detector=ObjectDetectRetinaNet())
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('object')

        expected_keys = ['video', 'frame', 'score', 'top', 'bottom',
                         'left', 'right', 'class']
        assert set(obj_out.keys()) == set(expected_keys)
        assert obj_out.shape == (17, 8)

    def test_object_detection_cutoff(self):
        anno = ObjectAnnotator(freq=4,
                               detector=ObjectDetectRetinaNet(cutoff=0.6))
        fpobj = FrameProcessor()
        fpobj.load_annotator(anno)

        finput = FrameInput("test-data/video-clip.mp4", bsize=8)
        fpobj.process(finput, max_batch=2)
        obj_out = fpobj.collect('object')

        assert obj_out.shape == (12, 8)


class TestPng:

    def test_png_existing_dir(self):
        dname = tempfile.mkdtemp() # creates directory

        fpobj = FrameProcessor()
        fpobj.load_annotator(PngAnnotator(output_dir=dname))

        finput = FrameInput("test-data/video-clip.mp4", bsize=4)
        fpobj.process(finput, max_batch=2)

        expected_files = set(['frame-{0:06d}.png'.format(x) for x in range(8)])
        obj_out = fpobj.collect('png')
        assert obj_out == DictFrame()
        assert set(os.listdir(dname)) == expected_files

    def test_png_new_dir(self):
        dname = tempfile.mkdtemp() # creates directory
        dname = os.path.join(dname, "temp2") # this directory will not exist

        fpobj = FrameProcessor()
        fpobj.load_annotator(PngAnnotator(output_dir=dname))

        finput = FrameInput("test-data/video-clip.mp4", bsize=4)
        fpobj.process(finput, max_batch=2)

        expected_files = set(['frame-{0:06d}.png'.format(x) for x in range(8)])
        obj_out = fpobj.collect('png')
        assert obj_out == DictFrame()
        assert set(os.listdir(dname)) == expected_files

    def test_png_image(self):
        dname = tempfile.mkdtemp() # creates directory

        fpobj = FrameProcessor()
        fpobj.load_annotator(PngAnnotator(output_dir=dname))

        iobj = ImageInput(input_paths=["test-data/img/frame-000076.png",
                                       "test-data/img/frame-000506.png"])
        fpobj.process(iobj, max_batch=2)

        # assert that all of the images in the input exist in output
        expected_files = set(['frame-000076.png', 'frame-000506.png'])
        obj_out = fpobj.collect('png')
        assert obj_out == DictFrame()
        assert set(os.listdir(dname)) == expected_files

        # make sure images are the same
        img1 = cv2.imread(os.path.join('test-data/img/frame-000076.png'))
        img2 = cv2.imread(os.path.join(dname, 'frame-000076.png'))
        assert np.all(img1 == img2)


if __name__ == '__main__':
    pytest.main([__file__])
