#!/usr/bin/env python3
"""
macOS Whisper Dictation Menubar App
A local speech-to-text application that uses Whisper AI with keyboard shortcuts.
"""

import sys
import os
import threading
import time
import plistlib

# Third-party imports
try:
    import rumps
    import sounddevice as sd
    import numpy as np
    from faster_whisper import WhisperModel
    import pyperclip
    from pynput import keyboard
    from pynput.keyboard import Controller, Key
except ImportError as e:
    print(f"Missing required package: {e}")
    print(
        "Install with: pip install faster-whisper pynput pyperclip sounddevice numpy rumps"
    )
    sys.exit(1)


class WhisperDictationApp:
    """Core dictation functionality using Whisper AI"""

    def __init__(self):
        # Configuration
        self.model_size = "small"  # tiny, base, small, medium, large
        self.sample_rate = 16000
        self.chunk_interval = 3  # seconds between streaming transcriptions
        self.overlap_seconds = 1  # overlap between chunks for context

        # State variables
        self.recording = False
        self.audio_data = []
        self._audio_lock = threading.Lock()
        self._transcribed_samples = 0
        self._stop_event = threading.Event()
        self._streaming_done = threading.Event()
        self._original_clipboard = None

        # Initialize components
        self.model = None
        self.stream = None
        self.listener = None

        # Setup
        self._initialize_whisper()

    def _initialize_whisper(self):
        """Initialize the Whisper model"""
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {e}")

    def _record_audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording"""
        if not self.recording:
            return
        with self._audio_lock:
            self.audio_data.append(indata[:, 0].copy())

    def start_recording(self):
        """Start audio recording with streaming transcription"""
        if self.recording:
            return

        self.recording = True
        self.audio_data = []
        self._transcribed_samples = 0
        self._stop_event.clear()
        self._streaming_done.clear()

        # Save clipboard once at the start of recording
        try:
            self._original_clipboard = pyperclip.paste()
        except Exception:
            self._original_clipboard = None

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._record_audio_callback,
            )
            self.stream.start()

            # Start streaming transcription thread
            threading.Thread(
                target=self._streaming_transcribe_loop, daemon=True
            ).start()
        except Exception:
            self.recording = False

    def _streaming_transcribe_loop(self):
        """Background thread that transcribes audio chunks while recording"""
        overlap_samples = int(self.overlap_seconds * self.sample_rate)

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.chunk_interval)
            self._transcribe_new_audio(overlap_samples)

        # Final transcription of any remaining audio
        self._transcribe_new_audio(overlap_samples)
        self._streaming_done.set()

    def _transcribe_new_audio(self, overlap_samples):
        """Transcribe audio that hasn't been processed yet"""
        with self._audio_lock:
            if len(self.audio_data) == 0:
                return
            audio_array = np.concatenate(self.audio_data)

        total_samples = len(audio_array)

        # Start from overlap before the last transcribed position for context
        start = max(0, self._transcribed_samples - overlap_samples)

        if total_samples <= start:
            return

        chunk = audio_array[start:]

        # Need a minimum amount of audio to transcribe (~0.5s)
        min_samples = int(0.5 * self.sample_rate)
        if len(chunk) < min_samples:
            return

        try:
            segments, _ = self.model.transcribe(chunk, vad_filter=True)
            text = "".join(segment.text for segment in segments)

            if text.strip():
                self._paste_text(text.strip())

            self._transcribed_samples = total_samples
        except Exception:
            pass

    def _paste_text(self, text):
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
            pass

    def stop_recording(self):
        """Stop audio recording and finalize transcription"""
        if not self.recording:
            return

        self.recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Signal the streaming thread to do its final pass and wait
        self._stop_event.set()
        self._streaming_done.wait(timeout=30)

        # Restore original clipboard
        if self._original_clipboard is not None:
            try:
                threading.Timer(
                    1.0, lambda: pyperclip.copy(self._original_clipboard)
                ).start()
            except Exception:
                pass

    def start_listening(self):
        """Start listening for double-tap of right Option key"""
        last_tap_time = [0]
        tap_threshold = 0.5

        def on_press(key):
            try:
                if key == keyboard.Key.alt_r:
                    now = time.time()
                    time_since_last = now - last_tap_time[0]

                    if not self.recording and time_since_last < tap_threshold:
                        threading.Thread(
                            target=self.start_recording, daemon=True
                        ).start()
                    elif self.recording:
                        self.stop_recording()

                    last_tap_time[0] = now
            except Exception:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            self.listener = listener
            try:
                listener.join()
            except KeyboardInterrupt:
                pass

    def cleanup(self):
        """Cleanup resources"""
        if self.recording:
            self.stop_recording()
        if self.listener:
            self.listener.stop()


LAUNCHAGENT_LABEL = "com.whisper.dictation"
LAUNCHAGENT_PATH = os.path.expanduser(
    f"~/Library/LaunchAgents/{LAUNCHAGENT_LABEL}.plist"
)


class WhisperMenuBarApp(rumps.App):
    """Menubar interface for Whisper Dictation"""

    def __init__(self):
        super(WhisperMenuBarApp, self).__init__(
            "Whisper",
            quit_button="Quit",
        )
        self.title = "💬"

        # Initialize whisper app
        self.whisper_app = None
        self.is_initialized = False
        self.is_recording = False

        # Create menu items
        self.status_item = rumps.MenuItem("Status: Ready", callback=None)
        self.record_button = rumps.MenuItem(
            "Start Recording (⌥⌥)", callback=self.toggle_recording
        )
        self.login_item = rumps.MenuItem(
            "Start at Login", callback=self.toggle_start_at_login
        )
        self.login_item.state = os.path.exists(LAUNCHAGENT_PATH)

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

    def _initialize_whisper(self):
        """Initialize the WhisperDictationApp in background"""
        try:
            self.status_item.title = "Status: Loading model..."
            self.whisper_app = WhisperDictationApp()
            self.is_initialized = True
            self.status_item.title = "Status: Ready"
            rumps.notification(
                title="Whisper Dictation",
                subtitle="Ready to use",
                message="Double-tap right Option (⌥) to start recording",
            )
            threading.Thread(target=self._start_listening, daemon=True).start()
        except Exception as e:
            self.status_item.title = f"Status: Error - {str(e)[:30]}"
            rumps.notification(
                title="Whisper Dictation",
                subtitle="Initialization failed",
                message=str(e),
            )

    def _start_listening(self):
        """Start the keyboard listener in background"""
        if self.whisper_app:
            self.whisper_app.start_listening()

    def toggle_recording(self, _):
        """Toggle recording on/off"""
        if not self.is_initialized:
            rumps.alert(
                "Whisper Not Ready", "Please wait for the model to finish loading."
            )
            return

        if self.is_recording:
            self.status_item.title = "Status: Finishing..."
            self.whisper_app.stop_recording()
            self.is_recording = False
            self.record_button.title = "Start Recording (⌥⌥)"
            self.status_item.title = "Status: Ready"
            self.title = "💬"
        else:
            threading.Thread(
                target=self.whisper_app.start_recording, daemon=True
            ).start()
            self.is_recording = True
            self.record_button.title = "Stop Recording (⌥)"
            self.status_item.title = "Status: Recording & Transcribing..."
            self.title = "🔴"

    def toggle_start_at_login(self, sender):
        """Toggle auto-start at login via LaunchAgent"""
        if sender.state:
            # Currently enabled — remove the plist
            try:
                os.unlink(LAUNCHAGENT_PATH)
            except OSError:
                pass
            sender.state = False
        else:
            # Currently disabled — write the plist
            plist = {
                "Label": LAUNCHAGENT_LABEL,
                "ProgramArguments": [
                    sys.executable,
                    os.path.abspath(__file__),
                ],
                "RunAtLoad": True,
                "KeepAlive": False,
            }
            os.makedirs(os.path.dirname(LAUNCHAGENT_PATH), exist_ok=True)
            with open(LAUNCHAGENT_PATH, "wb") as f:
                plistlib.dump(plist, f)
            sender.state = True

    @rumps.clicked("About")
    def about(self, _):
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
