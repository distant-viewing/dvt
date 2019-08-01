# -*- coding: utf-8 -*-
"""A pipeline for building an interactive website from a video file.
"""

import json
import os
import os.path

import numpy as np

from ..annotate.cielab import CIElabAnnotator
from ..annotate.diff import DiffAnnotator
from ..annotate.core import FrameProcessor, FrameInput, ImageInput
from ..annotate.face import FaceAnnotator, FaceDetectMtcnn
from ..annotate.embed import EmbedAnnotator, EmbedFrameKerasResNet50
from ..annotate.meta import MetaAnnotator
from ..annotate.object import ObjectAnnotator, ObjectDetectRetinaNet
from ..annotate.png import PngAnnotator
from ..aggregate.cut import CutAggregator
from ..utils import setup_tensorflow, _format_time


class WebPipeline:
    """Contains
    """

    def __init__(self, finput, doutput=None):
        # setup tensorflow
        setup_tensorflow()

        # find absolute path to the input and determine where the output will
        # live
        finput = os.path.abspath(finput)
        if doutput is None:
            doutput = os.path.join(os.getcwd(), "dvt-output")
        self.finput = finput
        self.doutput = doutput
        self.cuts = None
        self.cut_data = None

    def run(self):
        if not os.path.exists(self.doutput):
            os.makedirs(self.doutput)

        self._get_cuts()
        self._process_cuts()
        self._make_json()

        _copy_web()

    def _get_cuts(self):
        fpobj = FrameProcessor()
        fpobj.load_annotator(DiffAnnotator(quantiles=[40]))
        fri = FrameInput(self.finput, bsize=128)
        fpobj.process(fri)
        obj_out = fpobj.collect_all()

        ca = CutAggregator(cut_vals={"q40": 4})
        agg = ca.aggregate(obj_out)
        agg["frame_start"] = np.array(agg["frame_start"])
        agg["frame_end"] = np.array(agg["frame_end"])
        agg["mpoint"] = (
            agg["frame_start"] + (agg["frame_end"] - agg["frame_start"]) // 2
        )
        self.cuts = agg

    def _process_cuts(self):
        frames = self.cuts["mpoint"]

        fpobj = FrameProcessor()
        fpobj.load_annotator(MetaAnnotator())
        fpobj.load_annotator(
            PngAnnotator(
                output_dir=os.path.join(self.doutput, "img"), frames=frames
            )
        )
        fpobj.load_annotator(CIElabAnnotator(frames=frames, num_dominant=5))
        fpobj.load_annotator(
            FaceAnnotator(detector=FaceDetectMtcnn(), frames=frames)
        )
        fpobj.load_annotator(
            ObjectAnnotator(detector=ObjectDetectRetinaNet(), frames=frames)
        )

        fri = FrameInput(self.finput, bsize=128)
        fpobj.process(fri)
        self.cut_data = fpobj.collect_all()

    def _make_json(self):
        nframes = len(self.cuts["mpoint"])
        fps = self.cut_data["meta"]["fps"][0]
        num_faces, max_face_size, num_people, obj_list = _process_objects(
            self.cut_data, self.cuts["mpoint"]
        )

        output = []
        for iter in range(nframes):
            frame = self.cuts["mpoint"][iter]

            output += [
                {
                    "frame_start": int(self.cuts["frame_start"][iter]),
                    "frame_end": int(self.cuts["frame_end"][iter]),
                    "time_start": _format_time(
                        float(self.cuts["frame_start"][iter]) / fps * 1000,
                        include_msec=False,
                    ),
                    "time_end": _format_time(
                        float(self.cuts["frame_end"][iter]) / fps * 1000,
                        include_msec=False,
                    ),
                    "img_path": os.path.join(
                        "img",
                        "frame-{0:06d}.png".format(
                            self.cuts["frame_start"][iter]
                        ),
                    ),
                    "dominant_colors": self.cut_data["cielab"][
                        "dominant_colors"
                    ][iter].tolist(),
                    "num_faces": int(num_faces[iter]),
                    "max_face_size": int(max_face_size[iter]),
                    "num_people": int(num_people[iter]),
                    "obj_list": obj_list[iter],
                }
            ]

        with open(os.path.join(self.doutput, "data.json"), "w") as fout:
            json.dump(output, fout, sort_keys=True, indent=4)


def _process_objects(cut_data, frame_set):
    face_array = np.array(cut_data["face"]["frame"])
    obj_array = np.array(cut_data["object"]["frame"])
    face_size = np.array(cut_data["face"]["bottom"]) - np.array(
        cut_data["face"]["top"]
    )
    classes = np.array(cut_data["object"]["class"])
    obj_conf = np.array(cut_data["object"]["score"])

    num_faces = []
    max_face_size = []
    num_people = []
    obj_list = []

    for this_frame in frame_set:
        faces = np.nonzero(face_array == this_frame)[0]
        num_faces.append(len(faces))
        max_face_size.append(np.max(face_size[faces], initial=0))
        objs = np.nonzero((obj_array == this_frame) & (obj_conf > 0.7))[0]
        num_people.append(np.sum(classes[objs] == "person"))
        obj_list.append("; ".join(list(set(classes[objs]))))

    return num_faces, max_face_size, num_people, obj_list


def _copy_web():
    pass
