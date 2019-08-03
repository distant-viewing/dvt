from os import makedirs
from os.path import exists, join, abspath
import sys
import tempfile

import pytest

from dvt.cli import run_cli


class TestCli:
    def test_with_default(self):
        finput = abspath(join("test-data", "video-clip.mp4"))
        dname = tempfile.mkdtemp()  # creates directory

        sys.argv = ["python", "--dirout", dname, finput]
        run_cli()

        assert exists(join(dname, "video-clip"))
        assert exists(join(dname, "video-clip", "img"))
        assert exists(join(dname, "video-clip", "img-anno"))
        assert exists(join(dname, "video-clip", "img-flow"))
        assert exists(join(dname, "video-clip", "data.json"))

    def test_input_not_found(self):
        finput = abspath(join("test-data", "video-clip-fake.mp4"))
        dname = tempfile.mkdtemp()  # creates directory

        sys.argv = ["python", "--dirout", dname, finput]

        with pytest.raises(FileNotFoundError) as e_info:
            run_cli()

    def test_duplicate_input_base_names(self):
        dname = tempfile.mkdtemp()  # creates directory

        # make two empty files that have have the same basename
        open(join(dname, "file1.mp4"), "a").close()
        open(join(dname, "file1.avi"), "a").close()

        sys.argv = [
            "python",
            "--dirout",
            dname,
            join(dname, "file1.mp4"),
            join(dname, "file1.avi"),
        ]

        with pytest.raises(AssertionError) as e_info:
            run_cli()


if __name__ == "__main__":
    pytest.main([__file__])
