"""Pure-core functions for Whisper dictation logic.

No I/O, no model, no audio, no clipboard — plain data in, plain data out.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionPlan:
    should_transcribe: bool
    start_idx: int
    overlap_in_chunk_seconds: float
    new_transcribed_samples: int
    keep_from_idx: int
    new_committed_offset: int


def plan_transcription_window(
    *,
    committed_offset: int,
    transcribed_samples: int,
    snapshot_len: int,
    overlap_samples: int,
    sample_rate: int,
    min_samples: int,
) -> TranscriptionPlan:
    """Compute slice indices and overlap metadata for the next transcription pass.

    Args:
        committed_offset: Absolute sample index of the first sample in the current
            audio buffer (samples before this have been trimmed away).
        transcribed_samples: Absolute sample index up to which text has been emitted.
        snapshot_len: Number of samples in the current buffer snapshot.
        overlap_samples: How many samples to rewind for Whisper context.
        sample_rate: Audio sample rate (Hz).
        min_samples: Minimum chunk length required to attempt transcription.

    Returns:
        TranscriptionPlan — if ``should_transcribe`` is False, all other fields are
        zero/default and the caller should skip this pass.
    """
    start_abs = max(committed_offset, transcribed_samples - overlap_samples)
    start_idx = start_abs - committed_offset
    total_absolute = committed_offset + snapshot_len
    chunk_len = snapshot_len - start_idx

    if total_absolute <= start_abs or chunk_len < min_samples:
        return TranscriptionPlan(
            should_transcribe=False,
            start_idx=0,
            overlap_in_chunk_seconds=0.0,
            new_transcribed_samples=transcribed_samples,
            keep_from_idx=0,
            new_committed_offset=committed_offset,
        )

    overlap_in_chunk_seconds = (transcribed_samples - start_abs) / sample_rate
    new_transcribed_samples = total_absolute
    keep_from_abs = max(committed_offset, new_transcribed_samples - overlap_samples)
    keep_from_idx = keep_from_abs - committed_offset

    return TranscriptionPlan(
        should_transcribe=True,
        start_idx=start_idx,
        overlap_in_chunk_seconds=overlap_in_chunk_seconds,
        new_transcribed_samples=new_transcribed_samples,
        keep_from_idx=keep_from_idx,
        new_committed_offset=keep_from_abs,
    )


# ---------------------------------------------------------------------------
# Target 1: recorder event → UI state
# ---------------------------------------------------------------------------

_IDLE_BUTTON = "Start Recording (⌥⌥)"
_RECORDING_BUTTON = "Stop Recording (⌥)"


@dataclass(frozen=True)
class RecorderUIState:
    is_recording: bool
    menubar_title: str
    status_title: str
    record_button_title: str
    notification_message: str | None  # None → no notification


def parse_recorder_event(event: str) -> RecorderUIState:
    """Map a recorder event string to a UI update descriptor.

    Args:
        event: One of ``"started"``, ``"stopped"``, or ``"error:<message>"``.

    Returns:
        RecorderUIState with all fields needed to update the menubar UI.
    """
    if event == "started":
        return RecorderUIState(
            is_recording=True,
            menubar_title="🔴",
            status_title="Status: Recording & Transcribing...",
            record_button_title=_RECORDING_BUTTON,
            notification_message=None,
        )
    if event == "stopped":
        return RecorderUIState(
            is_recording=False,
            menubar_title="💬",
            status_title="Status: Ready",
            record_button_title=_IDLE_BUTTON,
            notification_message=None,
        )
    # error:<message>
    message = event[len("error:") :]
    return RecorderUIState(
        is_recording=False,
        menubar_title="💬",
        status_title="Status: Ready",
        record_button_title=_IDLE_BUTTON,
        notification_message=message,
    )


# ---------------------------------------------------------------------------
# Target 2: double-tap detection
# ---------------------------------------------------------------------------


def classify_key_event(is_recording: bool, time_since_last: float, tap_threshold: float) -> str:
    """Determine the action for a hotkey press.

    Args:
        is_recording: Whether audio is currently being captured.
        time_since_last: Seconds elapsed since the previous hotkey press.
        tap_threshold: Maximum inter-tap interval (exclusive) to count as a
            double-tap.

    Returns:
        ``"start"`` — begin recording (double-tap while idle).
        ``"stop"``  — end recording (any tap while recording).
        ``"ignore"`` — first tap while idle; wait for potential second tap.
    """
    if is_recording:
        return "stop"
    if time_since_last < tap_threshold:
        return "start"
    return "ignore"


# ---------------------------------------------------------------------------
# Target 3: segment filter + text join
# ---------------------------------------------------------------------------


def join_new_segments(segments: list[dict], overlap_in_chunk_seconds: float) -> str:  # type: ignore[type-arg]
    """Extract and join text from segments that fall after the overlap window.

    Args:
        segments: Whisper result segments, each with ``"start"`` (float) and
            ``"text"`` (str) keys.
        overlap_in_chunk_seconds: Seconds of already-emitted audio re-fed to
            Whisper for context. Segments whose ``start`` is within the overlap
            (with a 50 ms tolerance) are skipped.

    Returns:
        Concatenated text of qualifying segments.
    """
    threshold = overlap_in_chunk_seconds - 0.05
    return "".join(seg["text"] for seg in segments if seg["start"] >= threshold)


# ---------------------------------------------------------------------------
# Target 4: LaunchAgent plist construction
# ---------------------------------------------------------------------------


def build_launchagent_plist(label: str, executable: str, script_path: str) -> dict:  # type: ignore[type-arg]
    """Build the LaunchAgent plist dictionary.

    Args:
        label: The launchd service label (e.g. ``"com.whisper.dictation"``).
        executable: Absolute path to the Python interpreter.
        script_path: Absolute path to the entry-point script.

    Returns:
        A dict suitable for serialisation with :func:`plistlib.dump`.
    """
    return {
        "Label": label,
        "ProgramArguments": [executable, script_path],
        "RunAtLoad": True,
        "KeepAlive": False,
    }
