import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dvt.tools import probejson as probe
from dvt.tools.probejson import Stream

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=20,
)


@given(json_values)
def test_streams_never_raises_on_any_json(value: object) -> None:
    probe.streams(value)
    probe.streams({"streams": value})
    probe.streams({"streams": [value]})


@given(st.text())
def test_load_never_raises(text: str) -> None:
    probe.load(text)


def test_load_wants_an_object() -> None:
    assert probe.load("[]") is None
    assert probe.load("3") is None
    assert probe.load("{}") == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3), ("3", 3), ("3.5", None), (True, None), (None, None), ("x", None)],
)
def test_as_int(value: object, expected: int | None) -> None:
    assert probe.as_int(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3.0), (2.5, 2.5), ("2.5", 2.5), (False, None), ("N/A", None), (None, None)],
)
def test_as_float(value: object, expected: float | None) -> None:
    assert probe.as_float(value) == expected


@pytest.mark.parametrize(
    ("tag", "seconds"),
    [
        ("01:30:00.000000000", 5400.0),
        ("00:00:01.5", 1.5),
        ("1:2", None),
        ("a:b:c", None),
    ],
)
def test_parse_duration_tag(tag: str, seconds: float | None) -> None:
    assert probe.parse_duration_tag(tag) == seconds


def test_stream_fields() -> None:
    raw = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "field_order": "tt",
                    "bit_rate": "5000000",
                    "duration": "60.5",
                },
                {"codec_type": "audio", "channels": 6, "tags": {"language": "fra"}},
                {"codec_type": "audio", "tags": {"language": ""}},
            ]
        }
    )
    assert probe.streams(probe.load(raw)) == [
        Stream(0, "video", "h264", None, 1920, 1080, "tt", 5_000_000, 60.5, None),
        Stream(1, "audio", channels=6, language="fra"),
        Stream(2, "audio"),
    ]


def test_duration_falls_back_to_matroska_tag() -> None:
    data = {"streams": [{"tags": {"DURATION-eng": "00:01:00.000"}}]}
    assert probe.streams(data)[0].duration == 60.0


def test_duration_field_wins_over_tag() -> None:
    data = {"streams": [{"duration": "10", "tags": {"DURATION": "00:01:00.000"}}]}
    assert probe.streams(data)[0].duration == 10.0
