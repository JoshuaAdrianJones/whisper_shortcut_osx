# Module Depth Review: whisper_shortcut_osx

**Date:** 2026-04-12
**Scope:** All project source files — `whisper_core.py`, `whisper_menubar.py`, `test_whisper_core.py`
**Summary:** 8 public boundaries reviewed across 2 modules; 3 shallow findings identified. The highest-priority action is collapsing the LaunchAgent lifecycle (currently split across 3 functions in 2 files) into a single encapsulated abstraction. Two smaller opportunities follow.

---

## Depth Inventory

| Module | Location | Public Methods | Total Params | Depth Ratio | Signals Triggered |
|--------|----------|---------------|-------------|-------------|-------------------|
| `plan_transcription_window` | `whisper_core.py:36` | 1 | 6 | 2.9:1 | `high_param_count`, `missing_defaults` |
| `build_launchagent_plist` + `_launchagent_loaded` + `toggle_start_at_login` | `whisper_core.py:193`, `whisper_menubar.py:319`, `whisper_menubar.py:433` | 3 | 9 | 3.6:1 | `temporal_decomposition`, `information_leakage` |
| `WhisperMenuBarApp._start_listening` | `whisper_menubar.py:389` | 1 | 0 | 1:1 | `pass_through_method` |

> `WhisperDictationApp` (4 public methods, ~150 lines of implementation) and the ADT dataclasses (`RecordingStarted`, `DoTranscription`, etc.) are excluded — both are deep relative to their interfaces.

---

## Deepening Opportunities

### Opportunity 1: Collapse LaunchAgent lifecycle into one encapsulated boundary

**Modules involved:**
- `build_launchagent_plist` (`whisper_core.py:193`)
- `_launchagent_loaded` (`whisper_menubar.py:319`)
- `toggle_start_at_login` (`whisper_menubar.py:433`)

**Signals:** `temporal_decomposition`, `information_leakage`

**Current state:** A caller invoking "add to login items" must orchestrate five distinct concerns:

1. Know `LAUNCHAGENT_LABEL` and `LAUNCHAGENT_PATH` as module-level constants
2. Call `build_launchagent_plist(label, executable, script_path)` to construct the dict
3. Write the plist file with `plistlib.dump`
4. Call `launchctl bootstrap gui/<uid> <path>` — knowing the `bootstrap` verb
5. Call `launchctl bootout gui/<uid>/<label>` to reverse — knowing the `bootout` verb (different from `bootstrap`)

Removal is the mirror image: unlink the file, then `bootout`. After every mutation the caller re-queries `_launchagent_loaded` to refresh UI state, which means it must also know how to check state. The 50-line `toggle_start_at_login` method is pure orchestration of what should be hidden implementation.

**Proposed abstraction:** A `LaunchAgentController` class or a module `launchagent.py` with three functions:

```python
def install(label: str, executable: str, script_path: str) -> None: ...
def uninstall(label: str) -> None: ...
def is_active(label: str) -> bool: ...
```

`install` absorbs: plist construction, file path derivation, `plistlib.dump`, `launchctl bootstrap`.
`uninstall` absorbs: `launchctl bootout`, `os.unlink`.
`is_active` absorbs: `launchctl print`, returncode interpretation, uid lookup.

`toggle_start_at_login` collapses to ~6 lines:

```python
def toggle_start_at_login(self, sender: Any) -> None:
    if sender.state:
        launchagent.uninstall(LAUNCHAGENT_LABEL)
    else:
        launchagent.install(LAUNCHAGENT_LABEL, sys.executable, os.path.abspath(__file__))
    sender.state = launchagent.is_active(LAUNCHAGENT_LABEL)
```

**What gets hidden:** `LAUNCHAGENT_PATH` derivation, `os.getuid()`, launchctl verb selection (`bootstrap` vs `bootout`), plist schema, file I/O, error notifications for each subprocess step.

**Effort:** M

**Risk:** The error-notification side-effects currently live inside `toggle_start_at_login` (two `rumps.notification` calls on failure). Moving them inside the new module creates a UI-layer dependency in what should be pure logic. Resolution: have `install`/`uninstall` raise exceptions; let `toggle_start_at_login` catch and notify. No BREAKING changes — this is private code.

---

### Opportunity 2: Eliminate derived parameter `min_samples` from `plan_transcription_window`

**Modules involved:** `plan_transcription_window` (`whisper_core.py:36`), `_transcribe_new_audio` (`whisper_menubar.py:188`)

**Signals:** `high_param_count`, `missing_defaults`

**Current state:** The caller (`_transcribe_new_audio`) always computes `min_samples` as:

```python
min_samples = int(0.5 * self.sample_rate)
```

This is a leaked policy: the "minimum chunk is half a second" rule lives in the caller, not in the function that uses it. `plan_transcription_window` already receives `sample_rate`; it can compute the minimum itself.

The 6-parameter signature (all keyword-only) sits at a 2.9:1 depth ratio — right at the shallow threshold.

**Proposed abstraction:** Remove `min_samples` as a parameter; compute it internally:

```python
def plan_transcription_window(
    *,
    committed_offset: int,
    transcribed_samples: int,
    snapshot_len: int,
    overlap_samples: int,
    sample_rate: int,
    min_chunk_seconds: float = 0.5,   # policy lives here, not with callers
) -> TranscriptionOutcome:
    min_samples = int(min_chunk_seconds * sample_rate)
    ...
```

Param count drops from 6 → 5 (or 5 + 1 with default). The `min_samples` derivation disappears from `_transcribe_new_audio`. Callers who need a non-default minimum can still override `min_chunk_seconds`.

**What gets hidden:** The policy that 0.5 s is the minimum viable chunk length.

**Effort:** S

**Risk:** **BREAKING** for `test_whisper_core.py` — test calls pass `min_samples=8000` or `min_samples=4000` directly and will need updating to `min_chunk_seconds=0.5` / `min_chunk_seconds=0.25`. Behaviorally identical after the update.

---

### Opportunity 3: Delete pass-through method `_start_listening`

**Modules involved:** `WhisperMenuBarApp._start_listening` (`whisper_menubar.py:389`)

**Signals:** `pass_through_method`

**Current state:**

```python
def _start_listening(self) -> None:
    if self.whisper_app:
        self.whisper_app.start_listening()
```

Exists solely so that `threading.Thread(target=self._start_listening, daemon=True).start()` has a callable. The method adds one interface name and one conditional branch that contributes nothing — if `whisper_app` is `None` here it means `_initialize_whisper` failed and the thread would never be started. The guard is redundant.

**Proposed abstraction:** Inline the thread target:

```python
# whisper_menubar.py:379 — replace:
threading.Thread(target=self._start_listening, daemon=True).start()
# with:
threading.Thread(target=self.whisper_app.start_listening, daemon=True).start()
```

Delete `_start_listening` entirely. Move the `assert self.whisper_app is not None` to just before the thread spawn if mypy requires it (it already is non-None at that point in `_initialize_whisper`).

**What gets hidden:** Nothing new — this is pure interface reduction.

**Effort:** S (trivial)

**Risk:** None. Private method, no external callers.

---

## Implementation Sequence

| Priority | Opportunity | Effort | Rationale |
|----------|-------------|--------|-----------|
| 1 | Delete `_start_listening` pass-through | S | Zero risk, zero migration cost, immediately removes dead interface surface |
| 2 | Remove `min_samples` from `plan_transcription_window` | S | Policy moves to where it belongs; test updates are mechanical |
| 3 | Collapse LaunchAgent lifecycle | M | Largest cognitive-load reduction; requires new module/class and exception-based error propagation |

---

## Appendix: Detection Signal Reference

**pass_through_method** — A method whose body only forwards arguments to another method with a similar signature. Adds interface surface without functionality. Detection: ≤3 line body consisting of a forwarded call.

**high_param_count** — A function requiring many parameters pushes decisions onto callers that the module could absorb. Detection: >4 params on a public method (not counting `self`).

**information_leakage** — A design decision (data format, algorithm detail, constant) appears in two or more modules, forcing callers to know implementation details. Detection: identical constants, derived values, or data structures duplicated across files.

**temporal_decomposition** — Code split by *when* operations happen (read → transform → write) rather than by what knowledge they encapsulate. Related logic scatters across multiple modules with intermediate data passing. Detection: 2+ functions that operate on the same logical resource in sequence, each knowing part of the protocol.

**missing_defaults** — A module forces callers to supply values the module could compute internally. Every required parameter is interface cost. Detection: constructor or function requires values with obvious sensible defaults (timeouts, thresholds, derived constants).
