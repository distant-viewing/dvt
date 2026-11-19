"""The language of a file is the tag on its first audio stream, and nothing else."""

from collections.abc import Mapping, Sequence

from .values import int_or_none, mapping, text

# The three sentinels ISO 639-2 provides. zxx is no linguistic content, a
# silent film or a score with no speech. mul is more than one language, what
# the curated tables used to call mixed. und is not yet decided.
SILENT = "zxx"
MIXED = "mul"
UNDECIDED = "und"
SENTINELS = frozenset({SILENT, MIXED, UNDECIDED})

# ISO 639-2 has two three-letter codes for twenty languages, a terminology
# form (fra) and a bibliographic form (fre). The MP4 media header holds the
# terminology form, and so does the inventory; a tag in the other form is
# read as the same language.
BIBLIOGRAPHIC = {
    "alb": "sqi",
    "arm": "hye",
    "baq": "eus",
    "bur": "mya",
    "chi": "zho",
    "cze": "ces",
    "dut": "nld",
    "fre": "fra",
    "geo": "kat",
    "ger": "deu",
    "gre": "ell",
    "ice": "isl",
    "mac": "mkd",
    "mao": "mri",
    "may": "msa",
    "per": "fas",
    "rum": "ron",
    "slo": "slk",
    "tib": "bod",
    "wel": "cym",
}

# The two-letter ISO 639-1 code whisperx takes, by the three-letter tag.
ISO_639_1 = {
    "ara": "ar",
    "ben": "bn",
    "bod": "bo",
    "cat": "ca",
    "ces": "cs",
    "cym": "cy",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "eus": "eu",
    "fas": "fa",
    "fin": "fi",
    "fra": "fr",
    "glg": "gl",
    "heb": "he",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "hye": "hy",
    "ind": "id",
    "isl": "is",
    "ita": "it",
    "jpn": "ja",
    "kat": "ka",
    "kor": "ko",
    "lav": "lv",
    "lit": "lt",
    "mal": "ml",
    "mkd": "mk",
    "mri": "mi",
    "msa": "ms",
    "mya": "my",
    "nld": "nl",
    "nno": "nn",
    "nor": "no",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "sqi": "sq",
    "swe": "sv",
    "tam": "ta",
    "tel": "te",
    "tgl": "tl",
    "tha": "th",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "vie": "vi",
    "wol": "wo",
    "zho": "zh",
}


def normalise(tag: str | None) -> str:
    """The tag as the inventory records it: lower case, terminology form,
    and und where the stream carries no tag at all."""
    tag = (tag or "").strip().lower()
    if not tag:
        return UNDECIDED
    return BIBLIOGRAPHIC.get(tag, tag)


def silent(tag: str | None) -> bool:
    """Whether a file is silent: its audio is tagged zxx, or it has no audio
    stream at all, which the inventory records as no tag."""
    return tag is None or normalise(tag) == SILENT


def code(tag: str | None) -> str | None:
    """The two-letter code a transcriber takes, or None: for a sentinel,
    for no tag, and for a tag this table does not know."""
    tag = normalise(tag)
    return None if tag in SENTINELS else ISO_639_1.get(tag)


def describe(tag: str | None) -> str:
    """Why a file is or is not transcribable, in a few words."""
    tag = normalise(tag)
    if tag == SILENT:
        return "no linguistic content (zxx)"
    if tag == MIXED:
        return "no single language (mul)"
    if tag == UNDECIDED:
        return "language not yet decided (und)"
    two = ISO_639_1.get(tag)
    if two is None:
        return f"tagged {tag}, which is not a language this pipeline knows"
    return f"{two} from the tag {tag}"


def _tag_of(stream: Mapping[str, object]) -> str | None:
    tags = mapping(stream.get("tags"))
    return text(tags.get("language")) if tags else None


def _channels_of(stream: Mapping[str, object]) -> int:
    return int_or_none(stream.get("channels")) or 0


def pick_stream(streams: Sequence[Mapping[str, object]]) -> int:
    """Which audio stream is the film's: the index into `streams`, ffprobe's
    audio streams in order.

    The first stream sets the language, by the acquire convention that the
    film comes first and a dub or a commentary after it. Among the streams
    with that tag or none, the one with the most channels wins, so a 5.1 mix
    beats a stereo mix and a stream tagged otherwise is never taken. A first
    stream with no tag keeps only untagged company, so an untagged file
    cannot yield audio in a language the inventory does not know about. A
    mono first stream is taken as it is."""
    base = normalise(_tag_of(streams[0]))
    if _channels_of(streams[0]) <= 1:
        return 0
    eligible = [(n, s) for n, s in enumerate(streams) if normalise(_tag_of(s)) in (base, UNDECIDED)]
    return max(eligible, key=lambda ns: (_channels_of(ns[1]), -ns[0]))[0]


def tags_differ(streams: Sequence[Mapping[str, object]]) -> bool:
    """Whether the file's audio streams carry more than one real tag."""
    tags = {normalise(_tag_of(s)) for s in streams} - {UNDECIDED}
    return len(tags) > 1
