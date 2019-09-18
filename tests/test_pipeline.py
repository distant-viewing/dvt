from os import makedirs
from os.path import exists, join, abspath
import tempfile

from dvt.pipeline.csv import VideoCsvPipeline
from dvt.pipeline.viz import VideoVizPipeline, ImageVizPipeline


class TestPipelines:
    def test_video_viz(self):
        finput = abspath(join("test-data", "video-clip.mp4"))
        ainput = abspath(join("test-data", "video-clip.wav"))
        sinput = abspath(join("test-data", "video-clip.srt"))

        dname = tempfile.mkdtemp()  # creates directory

        wp = VideoVizPipeline(
            finput=finput,
            dirout=dname,
            path_to_audio=ainput,
            path_to_subtitle=sinput
        )
        wp.run()

        assert (
            wp.dextra.get_data()["cut"]["mpoint"] == [37, 114, 227, 341]
        ).all()
        assert exists(dname)
        assert exists(join(dname, "data"))
        assert exists(join(dname, "data", "toc.json"))
        assert exists(join(dname, "img", "video-clip"))
        assert exists(join(dname, "img", "video-clip", "frames", "frame-000037.png"))
        assert exists(join(dname, "img", "video-clip", "frames", "frame-000114.png"))
        assert exists(join(dname, "img", "video-clip", "frames", "frame-000227.png"))
        assert exists(join(dname, "img", "video-clip", "frames", "frame-000341.png"))

        # test two things: frequency argument works and we can redo a video
        # and it works correctly
        finput = abspath(join("test-data", "video-clip.mp4"))
        wp = VideoVizPipeline(finput, dname, frequency=256)
        wp.run()

        assert exists(join(dname, "img", "video-clip", "frames", "frame-000128.png"))

    def test_video_viz_with_cwd(self):
        finput = abspath(join("test-data", "video-clip.mp4"))
        wp = VideoVizPipeline(finput)

    def test_image_viz(self):
        finput = abspath(join("test-data", "img"))

        dname = tempfile.mkdtemp()  # creates directory

        wp = ImageVizPipeline(finput, dname)
        wp.run()

    def test_video_csv(self):
        finput = abspath(join("test-data", "video-clip.mp4"))
        ainput = abspath(join("test-data", "video-clip.wav"))
        sinput = abspath(join("test-data", "video-clip.srt"))

        dname = tempfile.mkdtemp()  # creates directory

        wp = VideoCsvPipeline(
            finput=finput,
            dirout=dname,
            path_to_audio=ainput,
            path_to_subtitle=sinput
        )
        wp.run()
