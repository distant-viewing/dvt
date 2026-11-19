from typing import NotRequired, TypedDict

# whisperx declares a word's start, end and score as always there; a word
# it could not align comes back without them.
class SingleWordSegment(TypedDict):
    word: str
    start: NotRequired[float]
    end: NotRequired[float]
    score: NotRequired[float]

class SingleCharSegment(TypedDict):
    char: str
    start: float
    end: float
    score: float

class SingleSegment(TypedDict):
    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float]

class SingleAlignedSegment(TypedDict):
    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float]
    words: list[SingleWordSegment]
    chars: list[SingleCharSegment] | None

class TranscriptionResult(TypedDict):
    segments: list[SingleSegment]
    language: str

class AlignedTranscriptionResult(TypedDict):
    segments: list[SingleAlignedSegment]
    word_segments: list[SingleWordSegment]
