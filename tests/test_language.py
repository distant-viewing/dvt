import pytest
from hypothesis import given
from hypothesis import strategies as st

from dvt.tools import language


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("fre", "fra"), (" FRA ", "fra"), ("", "und"), (None, "und"), ("eng", "eng"), ("xyz", "xyz")],
)
def test_normalise(tag: str | None, expected: str) -> None:
    assert language.normalise(tag) == expected


@given(st.one_of(st.none(), st.text(max_size=6)))
def test_normalise_is_idempotent(tag: str | None) -> None:
    once = language.normalise(tag)
    assert language.normalise(once) == once


@pytest.mark.parametrize(
    ("tag", "silent"), [(None, True), ("zxx", True), ("ZXX", True), ("eng", False), ("", False)]
)
def test_silent(tag: str | None, silent: bool) -> None:
    assert language.silent(tag) is silent


@pytest.mark.parametrize(
    ("tag", "two"), [("fre", "fr"), ("zxx", None), ("mul", None), (None, None), ("xyz", None)]
)
def test_code(tag: str | None, two: str | None) -> None:
    assert language.code(tag) == two


@pytest.mark.parametrize(
    ("tag", "text"),
    [
        ("zxx", "no linguistic content (zxx)"),
        ("mul", "no single language (mul)"),
        (None, "language not yet decided (und)"),
        ("ger", "de from the tag deu"),
        ("xyz", "tagged xyz, which is not a language this pipeline knows"),
    ],
)
def test_describe(tag: str | None, text: str) -> None:
    assert language.describe(tag) == text


def a(tag: str | None, channels: object) -> dict[str, object]:
    stream: dict[str, object] = {"channels": channels}
    if tag is not None:
        stream["tags"] = {"language": tag}
    return stream


@pytest.mark.parametrize(
    ("streams", "pick"),
    [
        ([a("fra", 2), a("fra", 6)], 1),  # the 5.1 mix of the film's language
        ([a("fra", 2), a("eng", 6)], 0),  # a dub is never taken
        ([a("fra", 2), a(None, 6)], 1),  # untagged company is
        ([a(None, 2), a("eng", 6), a(None, 6)], 2),
        ([a("fra", 1), a("fra", 6)], 0),  # a mono first stream stands
        ([a("fra", "x"), a("fra", None)], 0),
        ([a("fra", 6), a("fra", 6)], 0),  # ties go to the earlier stream
    ],
)
def test_pick_stream(streams: list[dict[str, object]], pick: int) -> None:
    assert language.pick_stream(streams) == pick


def test_tags_differ() -> None:
    assert not language.tags_differ([a("fra", 2), a("fre", 2), a(None, 2)])
    assert language.tags_differ([a("fra", 2), a("eng", 2)])
