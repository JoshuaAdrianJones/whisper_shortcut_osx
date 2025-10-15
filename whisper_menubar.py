#!/usr/bin/env python3
"""
macOS Whisper Dictation Menubar App
A local speech-to-text application that uses Whisper AI with keyboard shortcuts.
"""

import sys
import os
import threading
import wave
import tempfile
import time

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

        # State variables
        self.recording = False
        self.audio_data = []

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

    def _record_audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if not self.recording:
            return
        self.audio_data.append(indata[:, 0].copy())

    def start_recording(self):
        """Start audio recording"""
        if self.recording:
            return

        self.recording = True
        self.audio_data = []

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._record_audio_callback,
            )
            self.stream.start()
        except Exception:
            self.recording = False

    def stop_recording(self):
        """Stop audio recording and process"""
        if not self.recording:
            return

        self.recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if len(self.audio_data) == 0:
            return

        # Combine audio data
        audio_array = np.concatenate(self.audio_data)

        # Save to temporary file and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            self._save_audio_to_file(audio_array, tmp_file.name)

            try:
                segments, info = self.model.transcribe(tmp_file.name)

                # Extract text
                text = ""
                for segment in segments:
                    text += segment.text

                if text.strip():
                    self._insert_text(text.strip())

            except Exception:
                pass
            finally:
                os.unlink(tmp_file.name)

    def _save_audio_to_file(self, audio_data, filename):
        """Save audio data to WAV file"""
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def _insert_text(self, text):
        """Insert text at cursor position using clipboard"""
        try:
            original_clipboard = pyperclip.paste()
            pyperclip.copy(text)

            kbd = Controller()
            time.sleep(0.1)

            # Press Cmd+V to paste
            with kbd.pressed(Key.cmd):
                kbd.press("v")
                kbd.release("v")

            # Restore original clipboard after delay
            threading.Timer(1.0, lambda: pyperclip.copy(original_clipboard)).start()

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
        self.menu = [
            self.status_item,
            None,
            self.record_button,
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
            self.whisper_app.stop_recording()
            self.is_recording = False
            self.record_button.title = "Start Recording (⌥⌥)"
            self.status_item.title = "Status: Processing..."
            self.title = "💬"
        else:
            threading.Thread(
                target=self.whisper_app.start_recording, daemon=True
            ).start()
            self.is_recording = True
            self.record_button.title = "Stop Recording (⌥)"
            self.status_item.title = "Status: Recording..."
            self.title = "🔴"
            threading.Timer(
                1.0,
                lambda: setattr(self.status_item, "title", "Status: Ready")
                if not self.is_recording
                else None,
            ).start()

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
