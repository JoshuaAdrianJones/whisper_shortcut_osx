from launchagent import build_launchagent_plist
from whisper_core import (
    BatchMode,
    DoTranscription,
    IgnoreKeyPress,
    SkipTranscription,
    StartRecording,
    StopRecording,
    StreamingMode,
    WhisperSegment,
    classify_key_event,
    join_new_segments,
    plan_transcription_window,
)


def test_given_fresh_buffer_starts_at_zero_with_no_overlap() -> None:
    plan = plan_transcription_window(
        committed_offset=0,
        transcribed_samples=0,
        snapshot_len=16000 * 2,  # 2s
        overlap_samples=16000,  # 1s
        sample_rate=16000,
    )
    assert isinstance(plan, DoTranscription)
    assert plan.start_idx == 0
    assert plan.overlap_in_chunk_seconds == 0.0
    assert plan.new_transcribed_samples == 32000
    assert plan.keep_from_idx == 16000  # retain 1s tail
    assert plan.new_committed_offset == 16000


def test_given_chunk_smaller_than_min_samples_skips() -> None:
    plan = plan_transcription_window(
        committed_offset=0,
        transcribed_samples=0,
        snapshot_len=4000,
        overlap_samples=16000,
        sample_rate=16000,
    )
    assert isinstance(plan, SkipTranscription)


def test_given_prior_transcription_rewinds_by_overlap() -> None:
    plan = plan_transcription_window(
        committed_offset=0,
        transcribed_samples=48000,
        snapshot_len=64000,
        overlap_samples=16000,
        sample_rate=16000,
    )
    assert isinstance(plan, DoTranscription)
    assert plan.start_idx == 32000  # 48000 - 16000
    assert plan.overlap_in_chunk_seconds == 1.0  # the rewound second
    assert plan.new_transcribed_samples == 64000


def test_skip_when_no_new_audio_past_committed() -> None:
    # committed_offset equals total_absolute — nothing new
    plan = plan_transcription_window(
        committed_offset=32000,
        transcribed_samples=32000,
        snapshot_len=0,
        overlap_samples=16000,
        sample_rate=16000,
    )
    assert isinstance(plan, SkipTranscription)
    assert plan.new_committed_offset == 32000
    assert plan.new_transcribed_samples == 32000


def test_committed_offset_shifts_indices() -> None:
    # Buffer trimmed: committed_offset=16000, buffer holds samples 16000..48000
    plan = plan_transcription_window(
        committed_offset=16000,
        transcribed_samples=16000,  # nothing transcribed past trim point yet
        snapshot_len=32000,  # 2s of audio in the buffer
        overlap_samples=16000,
        sample_rate=16000,
    )
    assert isinstance(plan, DoTranscription)
    assert plan.start_idx == 0  # start_abs == committed_offset, so idx=0
    assert plan.overlap_in_chunk_seconds == 0.0
    assert plan.new_transcribed_samples == 48000
    assert plan.new_committed_offset == 32000
    assert plan.keep_from_idx == 16000  # keep last 1s of the 2s buffer


def test_overlap_capped_at_committed_offset() -> None:
    # transcribed_samples - overlap_samples < committed_offset:
    # start_abs must not go below committed_offset
    plan = plan_transcription_window(
        committed_offset=8000,
        transcribed_samples=8000,
        snapshot_len=24000,
        overlap_samples=16000,  # would rewind to -8000 without cap
        sample_rate=16000,
        min_chunk_seconds=0.25,
    )
    assert isinstance(plan, DoTranscription)
    assert plan.start_idx == 0  # capped at committed_offset
    assert plan.overlap_in_chunk_seconds == 0.0


# ---------------------------------------------------------------------------
# classify_key_event
# ---------------------------------------------------------------------------


def test_given_not_recording_and_quick_double_tap_returns_start() -> None:
    action = classify_key_event(is_recording=False, time_since_last=0.3, tap_threshold=0.5)
    assert isinstance(action, StartRecording)


def test_given_not_recording_and_slow_tap_returns_ignore() -> None:
    action = classify_key_event(is_recording=False, time_since_last=0.6, tap_threshold=0.5)
    assert isinstance(action, IgnoreKeyPress)


def test_given_recording_returns_stop_regardless_of_timing() -> None:
    action = classify_key_event(is_recording=True, time_since_last=0.1, tap_threshold=0.5)
    assert isinstance(action, StopRecording)


def test_given_recording_slow_tap_still_returns_stop() -> None:
    action = classify_key_event(is_recording=True, time_since_last=5.0, tap_threshold=0.5)
    assert isinstance(action, StopRecording)


def test_given_tap_exactly_at_threshold_returns_ignore() -> None:
    # time_since_last < tap_threshold is the condition, so equal means ignore
    action = classify_key_event(is_recording=False, time_since_last=0.5, tap_threshold=0.5)
    assert isinstance(action, IgnoreKeyPress)


# ---------------------------------------------------------------------------
# join_new_segments
# ---------------------------------------------------------------------------


def test_given_no_overlap_returns_all_segment_text() -> None:
    segments = [
        WhisperSegment(start=0.0, text="Hello "),
        WhisperSegment(start=1.0, text="world"),
    ]
    result = join_new_segments(segments, overlap_in_chunk_seconds=0.0)
    assert result == "Hello world"


def test_given_overlap_filters_segments_before_threshold() -> None:
    segments = [
        WhisperSegment(start=0.0, text="old "),
        WhisperSegment(start=0.5, text="also old "),
        WhisperSegment(start=1.0, text="new"),
    ]
    # threshold = 1.0 - 0.05 = 0.95; segments at 0.0 and 0.5 are excluded
    result = join_new_segments(segments, overlap_in_chunk_seconds=1.0)
    assert result == "new"


def test_given_segment_just_above_threshold_is_included() -> None:
    segments = [WhisperSegment(start=0.96, text="yes")]
    result = join_new_segments(segments, overlap_in_chunk_seconds=1.0)
    assert result == "yes"


def test_given_segment_just_below_threshold_is_excluded() -> None:
    segments = [WhisperSegment(start=0.94, text="no")]
    result = join_new_segments(segments, overlap_in_chunk_seconds=1.0)
    assert result == ""


def test_given_empty_segments_returns_empty_string() -> None:
    result = join_new_segments([], overlap_in_chunk_seconds=0.5)
    assert result == ""


# ---------------------------------------------------------------------------
# build_launchagent_plist
# ---------------------------------------------------------------------------


def test_given_args_returns_correct_plist_structure() -> None:
    plist = build_launchagent_plist(
        label="com.example.app",
        executable="/usr/bin/python3",
        script_path="/Users/josh/app.py",
    )
    assert plist["Label"] == "com.example.app"
    assert plist["ProgramArguments"] == [
        "/usr/bin/python3",
        "/Users/josh/app.py",
    ]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is False


def test_given_different_args_values_are_reflected() -> None:
    plist = build_launchagent_plist(
        label="com.other.label",
        executable="/opt/homebrew/bin/python3",
        script_path="/tmp/script.py",
    )
    assert plist["Label"] == "com.other.label"
    assert plist["ProgramArguments"][0] == "/opt/homebrew/bin/python3"
    assert plist["ProgramArguments"][1] == "/tmp/script.py"


# ---------------------------------------------------------------------------
# TranscriptionMode ADT
# ---------------------------------------------------------------------------


def test_streaming_mode_instantiates() -> None:
    mode = StreamingMode()
    assert isinstance(mode, StreamingMode)


def test_batch_mode_instantiates() -> None:
    mode = BatchMode()
    assert isinstance(mode, BatchMode)


def test_streaming_mode_equality() -> None:
    assert StreamingMode() == StreamingMode()


def test_batch_mode_equality() -> None:
    assert BatchMode() == BatchMode()


def test_modes_are_not_equal() -> None:
    assert StreamingMode() != BatchMode()


def test_streaming_mode_is_frozen() -> None:
    import dataclasses

    import pytest

    mode = StreamingMode()
    with pytest.raises(dataclasses.FrozenInstanceError):
        mode.x = 1  # type: ignore[misc]


def test_batch_mode_is_frozen() -> None:
    import dataclasses

    import pytest

    mode = BatchMode()
    with pytest.raises(dataclasses.FrozenInstanceError):
        mode.x = 1  # type: ignore[misc]
