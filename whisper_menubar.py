#!/usr/bin/env python3
"""
macOS Whisper Dictation Menubar App
Local speech-to-text using Whisper AI with keyboard shortcuts.
"""

import logging
import logging.handlers
import os
import plistlib
import subprocess
import sys
import threading
import time
from typing import Any

from whisper_core import (
    build_launchagent_plist,
    classify_key_event,
    join_new_segments,
    parse_recorder_event,
    plan_transcription_window,
)

# Third-party imports
try:
    import mlx_whisper
    import numpy as np
    import pyperclip
    import rumps
    import sounddevice as sd
    from pynput import keyboard
    from pynput.keyboard import Controller, Key
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: uv sync")
    sys.exit(1)


def _setup_logger() -> logging.Logger:
    log_dir = os.path.expanduser("~/Library/Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "whisper_dictation.log")
    lg = logging.getLogger("whisper_dictation")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        lg.addHandler(handler)
    return lg


logger = _setup_logger()


class WhisperDictationApp:
    """Core dictation functionality using Whisper AI"""

    def __init__(self) -> None:
        # Configuration
        self.model_repo = "mlx-community/whisper-large-v3-turbo"
        self.sample_rate = 16000
        self.chunk_interval = 3  # seconds between streaming transcriptions
        self.overlap_seconds = 1  # overlap between chunks for context

        # State variables
        self.recording = False
        self.audio_data: list[np.ndarray] = []
        self._audio_lock = threading.Lock()
        self._committed_offset = 0  # absolute samples dropped from head
        self._transcribed_samples = 0  # absolute samples emitted so far
        self._stop_event = threading.Event()
        self._streaming_done = threading.Event()
        self._streaming_done.set()
        self._original_clipboard: str | None = None
        self._stop_worker_thread: threading.Thread | None = None

        # Invoked on state transitions: "started", "stopped", "error:<msg>"
        self.state_callback: Any = None

        # Initialize components
        self.stream: sd.InputStream | None = None
        self.listener = None

        # Setup
        self._initialize_whisper()

    def _emit(self, event: str) -> None:
        cb = self.state_callback
        if cb is None:
            return
        try:
            cb(event)
        except Exception:
            logger.exception("state_callback raised")

    def _initialize_whisper(self) -> None:
        """Warm up mlx-whisper by running a silent pass (downloads/caches the model)."""
        try:
            mlx_whisper.transcribe(
                np.zeros(self.sample_rate, dtype=np.float32),
                path_or_hf_repo=self.model_repo,
            )
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {e}") from e

    def _record_audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Callback for audio recording"""
        if not self.recording:
            return
        with self._audio_lock:
            self.audio_data.append(indata[:, 0].copy())

    def start_recording(self) -> None:
        """Start audio recording with streaming transcription"""
        if self.recording:
            return

        # Wait for any prior stop worker so we don't race on audio_data
        prev = self._stop_worker_thread
        if prev is not None and prev.is_alive():
            prev.join(timeout=30)

        self.recording = True
        self.audio_data = []
        self._committed_offset = 0
        self._transcribed_samples = 0
        self._stop_event.clear()
        self._streaming_done.clear()

        # Save clipboard once at the start of recording
        try:
            self._original_clipboard = pyperclip.paste()
        except Exception:
            logger.exception("clipboard read failed")
            self._original_clipboard = None

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._record_audio_callback,
            )
            assert self.stream is not None
            self.stream.start()

            threading.Thread(target=self._streaming_transcribe_loop, daemon=True).start()
            self._emit("started")
        except Exception as e:
            logger.exception("failed to start input stream")
            self.recording = False
            self.stream = None
            self._streaming_done.set()
            self._emit(f"error:{e}")

    def _streaming_transcribe_loop(self) -> None:
        """Background thread that transcribes audio chunks while recording"""
        overlap_samples = int(self.overlap_seconds * self.sample_rate)

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.chunk_interval)
            self._transcribe_new_audio(overlap_samples)

        # Final transcription of any remaining audio
        self._transcribe_new_audio(overlap_samples)
        self._streaming_done.set()

    def _transcribe_new_audio(self, overlap_samples: int) -> None:
        """Transcribe audio that hasn't been processed yet"""
        with self._audio_lock:
            if len(self.audio_data) == 0:
                return
            snapshot = list(self.audio_data)
            committed_offset = self._committed_offset

        audio_array = np.concatenate(snapshot)
        min_samples = int(0.5 * self.sample_rate)

        plan = plan_transcription_window(
            committed_offset=committed_offset,
            transcribed_samples=self._transcribed_samples,
            snapshot_len=len(audio_array),
            overlap_samples=overlap_samples,
            sample_rate=self.sample_rate,
            min_samples=min_samples,
        )

        if not plan.should_transcribe:
            return

        chunk = audio_array[plan.start_idx :]

        try:
            result = mlx_whisper.transcribe(chunk, path_or_hf_repo=self.model_repo)
            new_text = join_new_segments(result.get("segments", []), plan.overlap_in_chunk_seconds)

            if new_text.strip():
                self._paste_text(new_text.strip())

            self._transcribed_samples = plan.new_transcribed_samples
            tail = audio_array[plan.keep_from_idx :]

            with self._audio_lock:
                # Preserve any frames the callback appended during transcription.
                new_entries = self.audio_data[len(snapshot) :]
                self.audio_data = [tail] + new_entries
                self._committed_offset = plan.new_committed_offset
        except Exception:
            logger.exception("transcription failed")

    def _paste_text(self, text: str) -> None:
        """Paste text at cursor position without saving/restoring clipboard"""
        try:
            pyperclip.copy(text)
            kbd = Controller()
            time.sleep(0.1)
            with kbd.pressed(Key.cmd):
                kbd.press("v")
                kbd.release("v")
            time.sleep(0.05)
        except Exception:
            logger.exception("paste failed")

    def stop_recording(self) -> None:
        """Stop audio recording; finalization happens on a worker thread."""
        if not self.recording:
            return

        self.recording = False
        t = threading.Thread(target=self._stop_worker, daemon=True)
        self._stop_worker_thread = t
        t.start()

    def _stop_worker(self) -> None:
        try:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    logger.exception("failed to close stream")
                self.stream = None

            # Signal the streaming thread to do its final pass and wait.
            self._stop_event.set()
            self._streaming_done.wait(timeout=30)

            # Restore original clipboard synchronously so a quick re-start
            # doesn't capture the transcript as "_original_clipboard".
            if self._original_clipboard is not None:
                try:
                    pyperclip.copy(self._original_clipboard)
                except Exception:
                    logger.exception("clipboard restore failed")
        finally:
            self._emit("stopped")

    def start_listening(self) -> None:
        """Start listening for double-tap of right Option key"""
        last_tap_time: list[float] = [0.0]
        tap_threshold = 0.5

        def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            try:
                if key == keyboard.Key.alt_r:
                    now = time.time()
                    action = classify_key_event(
                        is_recording=self.recording,
                        time_since_last=now - last_tap_time[0],
                        tap_threshold=tap_threshold,
                    )
                    if action == "start":
                        threading.Thread(target=self.start_recording, daemon=True).start()
                    elif action == "stop":
                        self.stop_recording()
                    last_tap_time[0] = now
            except Exception:
                logger.exception("hotkey handler failed")

        with keyboard.Listener(on_press=on_press) as listener:
            self.listener = listener
            try:
                listener.join()
            except KeyboardInterrupt:
                pass

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.recording:
            self.stop_recording()
        if self.listener:
            self.listener.stop()


LAUNCHAGENT_LABEL = "com.whisper.dictation"
LAUNCHAGENT_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHAGENT_LABEL}.plist")


def _launchagent_loaded() -> bool:
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHAGENT_LABEL}"],
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        logger.exception("launchctl print failed")
        return False


class WhisperMenuBarApp(rumps.App):  # type: ignore[misc]
    """Menubar interface for Whisper Dictation"""

    def __init__(self) -> None:
        super().__init__(
            "Whisper",
            quit_button="Quit",
        )
        self.title = "💬"

        # Initialize whisper app
        self.whisper_app: WhisperDictationApp | None = None
        self.is_initialized = False
        self.is_recording = False

        # Create menu items
        self.status_item = rumps.MenuItem("Status: Ready", callback=None)
        self.record_button = rumps.MenuItem("Start Recording (⌥⌥)", callback=self.toggle_recording)
        self.login_item = rumps.MenuItem("Start at Login", callback=self.toggle_start_at_login)
        self.login_item.state = _launchagent_loaded()

        self.menu = [
            self.status_item,
            None,
            self.record_button,
            None,
            self.login_item,
            None,
            "About",
        ]

        # Initialize whisper in background
        threading.Thread(target=self._initialize_whisper, daemon=True).start()

    def _initialize_whisper(self) -> None:
        """Initialize the WhisperDictationApp in background"""
        try:
            self.status_item.title = "Status: Loading model..."
            self.whisper_app = WhisperDictationApp()
            self.whisper_app.state_callback = self._on_recorder_state
            self.is_initialized = True
            self.status_item.title = "Status: Ready"
            rumps.notification(
                title="Whisper Dictation",
                subtitle="Ready to use",
                message="Double-tap right Option (⌥) to start recording",
            )
            threading.Thread(target=self._start_listening, daemon=True).start()
        except Exception as e:
            logger.exception("whisper init failed")
            self.status_item.title = f"Status: Error - {str(e)[:30]}"
            rumps.notification(
                title="Whisper Dictation",
                subtitle="Initialization failed",
                message=str(e),
            )

    def _start_listening(self) -> None:
        """Start the keyboard listener in background"""
        if self.whisper_app:
            self.whisper_app.start_listening()

    def _on_recorder_state(self, event: str) -> None:
        """State callback from WhisperDictationApp — drives menubar UI."""
        state = parse_recorder_event(event)
        self.is_recording = state.is_recording
        self.record_button.title = state.record_button_title
        self.status_item.title = state.status_title
        self.title = state.menubar_title
        if state.notification_message is not None:
            rumps.notification(
                title="Whisper Dictation",
                subtitle="Recording failed",
                message=state.notification_message,
            )

    def toggle_recording(self, _: Any) -> None:
        """Toggle recording; UI state is driven by the recorder callback."""
        if not self.is_initialized:
            rumps.alert("Whisper Not Ready", "Please wait for the model to finish loading.")
            return

        assert self.whisper_app is not None
        if self.is_recording:
            self.status_item.title = "Status: Finishing..."
            threading.Thread(target=self.whisper_app.stop_recording, daemon=True).start()
        else:
            threading.Thread(target=self.whisper_app.start_recording, daemon=True).start()

    def toggle_start_at_login(self, sender: Any) -> None:
        """Toggle auto-start at login via LaunchAgent.

        Note: the plist pins the current Python interpreter and script path. If
        the venv or script moves, toggle this off and on again to refresh.
        """
        uid = os.getuid()
        if sender.state:
            # Currently enabled — bootout and remove the plist
            subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}/{LAUNCHAGENT_LABEL}"],
                capture_output=True,
            )
            try:
                os.unlink(LAUNCHAGENT_PATH)
            except OSError:
                logger.exception("failed to remove LaunchAgent plist")
        else:
            plist = build_launchagent_plist(
                label=LAUNCHAGENT_LABEL,
                executable=sys.executable,
                script_path=os.path.abspath(__file__),
            )
            os.makedirs(os.path.dirname(LAUNCHAGENT_PATH), exist_ok=True)
            try:
                with open(LAUNCHAGENT_PATH, "wb") as f:
                    plistlib.dump(plist, f)
            except Exception as e:
                logger.exception("failed to write LaunchAgent plist")
                rumps.notification(
                    title="Whisper Dictation",
                    subtitle="Start at Login failed",
                    message=str(e),
                )
                sender.state = _launchagent_loaded()
                return

            proc = subprocess.run(
                ["launchctl", "bootstrap", f"gui/{uid}", LAUNCHAGENT_PATH],
                capture_output=True,
            )
            if proc.returncode != 0:
                err = proc.stderr.decode(errors="replace").strip()
                logger.error("launchctl bootstrap failed: %s", err)
                rumps.notification(
                    title="Whisper Dictation",
                    subtitle="Start at Login failed",
                    message=err or "launchctl bootstrap returned non-zero",
                )

        sender.state = _launchagent_loaded()

    @rumps.clicked("About")  # type: ignore[untyped-decorator]
    def about(self, _: Any) -> None:
        """Show about dialog"""
        rumps.alert(
            title="Whisper Dictation",
            message=(
                "A local speech-to-text app using OpenAI Whisper\n\n"
                "Controls:\n"
                "• Double-tap right Option (⌥⌥) to start recording\n"
                "• Single tap right Option (⌥) to stop\n"
                "• Or use the menubar button\n\n"
                "All processing happens locally on your device."
            ),
        )


if __name__ == "__main__":
    WhisperMenuBarApp().run()
