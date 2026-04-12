"""Pure-core functions for Whisper dictation logic.

No I/O, no model, no audio, no clipboard — plain data in, plain data out.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# TranscriptionMode  (streaming: transcribe while recording; batch: wait until stop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamingMode:
    """Transcribe in chunks while recording and paste each chunk immediately."""


@dataclass(frozen=True)
class BatchMode:
    """Accumulate all audio, then transcribe and paste once after recording stops."""


TranscriptionMode = StreamingMode | BatchMode


# ---------------------------------------------------------------------------
# TranscriptionOutcome  (was: TranscriptionPlan with boolean mode flag)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkipTranscription:
    """No new audio to process; caller should not call mlx_whisper."""

    new_transcribed_samples: int
    new_committed_offset: int


@dataclass(frozen=True)
class DoTranscription:
    """Caller should slice audio[start_idx:] and pass to mlx_whisper."""

    start_idx: int
    overlap_in_chunk_seconds: float
    new_transcribed_samples: int
    keep_from_idx: int
    new_committed_offset: int


TranscriptionOutcome = SkipTranscription | DoTranscription


def plan_transcription_window(
    *,
    committed_offset: int,
    transcribed_samples: int,
    snapshot_len: int,
    overlap_samples: int,
    sample_rate: int,
    min_chunk_seconds: float = 0.5,
) -> TranscriptionOutcome:
    """Compute slice indices and overlap metadata for the next transcription pass.

    Args:
        committed_offset: Absolute sample index of the first sample in the current
            audio buffer (samples before this have been trimmed away).
        transcribed_samples: Absolute sample index up to which text has been emitted.
        snapshot_len: Number of samples in the current buffer snapshot.
        overlap_samples: How many samples to rewind for Whisper context.
        sample_rate: Audio sample rate (Hz).
        min_chunk_seconds: Minimum chunk duration required to attempt transcription.

    Returns:
        SkipTranscription if there is nothing new to process; DoTranscription
        with slice indices and overlap metadata otherwise.
    """
    min_samples = int(min_chunk_seconds * sample_rate)
    start_abs = max(committed_offset, transcribed_samples - overlap_samples)
    start_idx = start_abs - committed_offset
    total_absolute = committed_offset + snapshot_len
    chunk_len = snapshot_len - start_idx

    if total_absolute <= start_abs or chunk_len < min_samples:
        return SkipTranscription(
            new_transcribed_samples=transcribed_samples,
            new_committed_offset=committed_offset,
        )

    overlap_in_chunk_seconds = (transcribed_samples - start_abs) / sample_rate
    new_transcribed_samples = total_absolute
    keep_from_abs = max(committed_offset, new_transcribed_samples - overlap_samples)
    keep_from_idx = keep_from_abs - committed_offset

    return DoTranscription(
        start_idx=start_idx,
        overlap_in_chunk_seconds=overlap_in_chunk_seconds,
        new_transcribed_samples=new_transcribed_samples,
        keep_from_idx=keep_from_idx,
        new_committed_offset=keep_from_abs,
    )


# ---------------------------------------------------------------------------
# RecorderEvent  (was: string discriminator "started"/"stopped"/"error:<msg>")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingStarted:
    pass


@dataclass(frozen=True)
class RecordingStopped:
    pass


@dataclass(frozen=True)
class RecordingError:
    message: str


RecorderEvent = RecordingStarted | RecordingStopped | RecordingError


# ---------------------------------------------------------------------------
# KeyAction  (was: classify_key_event returning bare strings)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartRecording:
    pass


@dataclass(frozen=True)
class StopRecording:
    pass


@dataclass(frozen=True)
class IgnoreKeyPress:
    pass


KeyAction = StartRecording | StopRecording | IgnoreKeyPress


def classify_key_event(
    is_recording: bool, time_since_last: float, tap_threshold: float
) -> KeyAction:
    """Determine the action for a hotkey press.

    Args:
        is_recording: Whether audio is currently being captured.
        time_since_last: Seconds elapsed since the previous hotkey press.
        tap_threshold: Maximum inter-tap interval (exclusive) to count as a
            double-tap.

    Returns:
        StartRecording — begin recording (double-tap while idle).
        StopRecording  — end recording (any tap while recording).
        IgnoreKeyPress — first tap while idle; wait for potential second tap.
    """
    if is_recording:
        return StopRecording()
    if time_since_last < tap_threshold:
        return StartRecording()
    return IgnoreKeyPress()


# ---------------------------------------------------------------------------
# WhisperSegment  (was: list[dict] with untyped "start"/"text" access)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhisperSegment:
    start: float
    text: str

    @staticmethod
    def from_dict(d: dict[str, object]) -> "WhisperSegment":
        return WhisperSegment(
            start=float(d["start"]),  # type: ignore[arg-type]
            text=str(d["text"]),
        )


def join_new_segments(segments: list[WhisperSegment], overlap_in_chunk_seconds: float) -> str:
    """Extract and join text from segments that fall after the overlap window.

    Args:
        segments: Whisper result segments.
        overlap_in_chunk_seconds: Seconds of already-emitted audio re-fed to
            Whisper for context. Segments whose ``start`` is within the overlap
            (with a 50 ms tolerance) are skipped.

    Returns:
        Concatenated text of qualifying segments.
    """
    threshold = overlap_in_chunk_seconds - 0.05
    return "".join(seg.text for seg in segments if seg.start >= threshold)
