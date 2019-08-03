from os import makedirs
from os.path import exists, join, abspath
import tempfile

from dvt.pipeline.video import VideoPipeline


class TestVideoPipeline:
    def test_with_default(self):
        finput = abspath(join("test-data", "video-clip.mp4"))

        dname = tempfile.mkdtemp()  # creates directory

        wp = VideoPipeline(finput, dname)
        wp.make_breaks()
        wp.run()

        assert (wp.cuts["mpoint"] == [37, 114, 227, 341]).all()
        assert exists(dname)
        assert exists(join(dname, "toc.json"))
        assert exists(join(dname, "video-clip"))
        assert exists(join(dname, "video-clip", "img"))
        assert exists(join(dname, "video-clip", "img", "frame-000037.png"))
        assert exists(join(dname, "video-clip", "img", "frame-000114.png"))
        assert exists(join(dname, "video-clip", "img", "frame-000227.png"))
        assert exists(join(dname, "video-clip", "img", "frame-000341.png"))

    def test_with_cwd(self):
        finput = abspath(join("test-data", "video-clip.mp4"))
        wp = VideoPipeline(finput)


if __name__ == "__main__":
    pytest.main([__file__])
