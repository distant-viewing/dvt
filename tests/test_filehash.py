from pathlib import Path

import polars as pl
import pytest

from dvt.tools import filehash

# Hashes the module gave before it took up the strict checks. Every hash in
# the corpus record was made this way, so these must never change.


def pattern(n: int) -> bytes:
    return bytes((i * 7) % 251 for i in range(n))


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (bytes(range(256)) * 10, (2560, "0e27fd7ec8b2d9cc406cd37b399779f5")),
        (pattern(3 * (1 << 20) + 123), (3145851, "5a29823ecbefd8c51805f52f83e70b43")),
        (b"", (0, "dae175c3b0a832421b8a0c0041f9a72e")),
    ],
)
def test_size_and_hash(tmp_path: Path, content: bytes, expected: tuple[int, str]) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(content)
    assert filehash.size_and_hash(path) == expected


def test_size_and_hash_of_nothing(tmp_path: Path) -> None:
    assert filehash.size_and_hash(tmp_path / "missing") == (None, None)


def test_middle_of_a_large_file_is_not_read(tmp_path: Path) -> None:
    content = bytearray(pattern(3 * (1 << 20)))
    path = tmp_path / "f.bin"
    path.write_bytes(content)
    before = filehash.size_and_hash(path)
    content[len(content) // 2] ^= 0xFF
    path.write_bytes(content)
    assert filehash.size_and_hash(path) == before


def test_audio_paths(tmp_path: Path) -> None:
    m4a = tmp_path / "film.m4a"
    marker = filehash.noaudio_path_for(m4a)
    assert marker.name == "film.m4a.noaudio"
    assert filehash.audio_path_for(m4a) is None
    marker.write_bytes(filehash.NOAUDIO_CONTENT)
    assert filehash.audio_path_for(m4a) == marker
    m4a.write_bytes(b"x")
    assert filehash.audio_path_for(m4a) == m4a


def record(tmp_path: Path, mp4: Path, m4a: Path) -> Path:
    mp4_bytes, mp4_hash = filehash.size_and_hash(mp4)
    m4a_bytes, m4a_hash = filehash.size_and_hash(m4a)
    meta = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {"mp4_bytes": [mp4_bytes], "mp4_hash": [mp4_hash], "m4a_bytes": [m4a_bytes],
         "m4a_hash": [m4a_hash]},
        schema={"mp4_bytes": pl.Int64, "mp4_hash": pl.String, "m4a_bytes": pl.Int64,
                "m4a_hash": pl.String},
    ).write_parquet(meta)  # fmt: skip
    return meta


def test_check_sources(tmp_path: Path) -> None:
    mp4, m4a = tmp_path / "f.mp4", tmp_path / "f.m4a"
    mp4.write_bytes(b"video")
    m4a.write_bytes(b"audio")
    meta = record(tmp_path, mp4, m4a)
    assert filehash.check_sources(meta, mp4, m4a) == []

    mp4.write_bytes(b"video, changed")
    m4a.unlink()
    verdicts = [v for v, _ in filehash.check_sources(meta, mp4, m4a)]
    assert verdicts == ["mp4 changed", "m4a gone"]
    assert set(verdicts) <= set(filehash.BLOCKING_VERDICTS)


def test_check_sources_unrecorded(tmp_path: Path) -> None:
    mp4, m4a = tmp_path / "f.mp4", tmp_path / "f.m4a"
    mp4.write_bytes(b"video")
    meta = record(tmp_path, tmp_path / "none.mp4", tmp_path / "none.m4a")
    found = filehash.check_sources(meta, mp4, m4a)
    assert [v for v, _ in found] == ["mp4 not recorded", "m4a not recorded"]
    assert "is there now, 5 bytes" in found[0][1]


def test_check_sources_predates_hashes(tmp_path: Path) -> None:
    meta = tmp_path / "metadata.parquet"
    pl.DataFrame({"title": ["x"]}).write_parquet(meta)
    assert filehash.check_sources(meta, tmp_path / "a", tmp_path / "b") == [
        ("not recorded", "metadata.parquet predates the hash columns")
    ]
