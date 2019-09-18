from os import makedirs
from os.path import exists, join, abspath
import tempfile

import pytest

from dvt.cli import run_cli


class TestCli:
    def test_help(self):
        run_cli()
        run_cli(["python"])

        with pytest.raises(SystemExit):
            run_cli(["python", "video-viz", "-h"])

        with pytest.raises(SystemExit):
            run_cli(["python", "image-viz", "-h"])

        with pytest.raises(SystemExit):
            run_cli(["python", "video-csv", "-h"])

    def test_with_default(self):
        finput = abspath(join("test-data", "video-clip.mp4"))
        dname = tempfile.mkdtemp()  # creates directory

        run_cli(["python", "video-viz", "--dirout", dname, finput])

        assert exists(join(dname, "img"))
        assert exists(join(dname, "img", "video-clip"))
        assert exists(join(dname, "img", "video-clip", "display"))
        assert exists(join(dname, "img", "video-clip", "thumb"))
        assert exists(join(dname, "img", "video-clip", "flow"))
        assert exists(join(dname, "data", "video-clip.json"))

    def test_input_not_found(self):
        finput = abspath(join("test-data", "video-clip-fake.mp4"))
        dname = tempfile.mkdtemp()  # creates directory

        with pytest.raises(FileNotFoundError) as e_info:
            run_cli(["python",  "video-viz", "--dirout", dname, finput])
