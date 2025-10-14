#!/usr/bin/env python3
"""
macOS Whisper Dictation App
A local speech-to-text application that uses Whisper AI with keyboard shortcuts
Specifically designed for macOS with proper permissions handling.

Requirements:
- pip install faster-whisper pynput pyperclip sounddevice webrtcvad numpy wave
- macOS accessibility permissions for the Terminal/Python app
- Microphone permissions

Usage:
1. Run the script: python3 whisper_dictation.py
2. Press Cmd+Shift+S to start/stop recording
3. The transcribed text will be inserted at the cursor location
"""

import sys
import os
import threading
import wave
import tempfile
from pynput import keyboard
import time

# Third-party imports
try:
    import sounddevice as sd
    import numpy as np
    from faster_whisper import WhisperModel
    import pyperclip
    from pynput import keyboard
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install faster-whisper pynput pyperclip sounddevice numpy")
    sys.exit(1)


class WhisperDictationApp:
    def __init__(self, silent=False):
        # Configuration
        self.model_size = "base"  # tiny, base, small, medium, large
        self.sample_rate = 16000
        self.silent = silent

        # State variables
        self.recording = False
        self.audio_data = []

        # Initialize components
        self.model = None
        self.stream = None
        self.listener = None

        # Setup
        if not silent:
            self._check_permissions()
        self._initialize_whisper()

    def _check_permissions(self):
        """Check and guide user through macOS permissions setup"""
        print("=== macOS Permissions Check ===")
        print("This app requires the following permissions:")
        print("1. Accessibility: To insert text and detect keyboard shortcuts")
        print("2. Microphone: To record audio for transcription")
        print("\nTo grant permissions:")
        print("• Go to System Preferences > Security & Privacy > Privacy")
        print("• Add your Terminal app (or IDE) to 'Accessibility' and 'Microphone'")
        print("• If using Python directly, you may need to add Python to both lists")
        print("\nPress Enter to continue once permissions are granted...")
        input()

    def _initialize_whisper(self):
        """Initialize the Whisper model"""
        if not self.silent:
            print(f"Loading Whisper {self.model_size} model...")
        try:
            # Use CPU by default, change to "cuda" if you have compatible GPU
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",  # Quantization for better CPU performance
            )
            if not self.silent:
                print("✓ Whisper model loaded successfully")
        except Exception as e:
            if not self.silent:
                print(f"✗ Failed to load Whisper model: {e}")
            raise

    def _record_audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if not self.recording:
            return

        if status:
            print(f"Audio status: {status}")

        # Store audio data
        self.audio_data.append(indata[:, 0].copy())

    def start_recording(self):
        """Start audio recording"""
        if self.recording:
            return

        if not self.silent:
            print("\n🎙️  Recording started - speak now...")
            print("   (Press Cmd+Shift+S again to stop)")

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

        except Exception as e:
            if not self.silent:
                print(f"✗ Failed to start recording: {e}")
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
            if not self.silent:
                print("\nNo audio recorded")
            return

        if not self.silent:
            print("\n🔄 Processing audio...")

        # Combine audio data
        audio_array = np.concatenate(self.audio_data)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            self._save_audio_to_file(audio_array, tmp_file.name)

            # Transcribe with Whisper
            try:
                if not self.silent:
                    print("🤖 Transcribing with Whisper...")
                segments, info = self.model.transcribe(tmp_file.name)

                # Extract text
                text = ""
                for segment in segments:
                    text += segment.text

                if text.strip():
                    if not self.silent:
                        print(f"📝 Transcribed: {text.strip()}")
                    self._insert_text(text.strip())
                else:
                    if not self.silent:
                        print("🤷 No speech detected in audio")

            except Exception as e:
                if not self.silent:
                    print(f"✗ Transcription failed: {e}")
            finally:
                # Clean up temp file
                os.unlink(tmp_file.name)

    def _save_audio_to_file(self, audio_data, filename):
        """Save audio data to WAV file"""
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def _insert_text(self, text):
        """Insert text at cursor position using multiple methods"""
        try:
            # Method 1: Clipboard + paste (most reliable)
            original_clipboard = pyperclip.paste()  # Backup current clipboard
            pyperclip.copy(text)

            # Use pynput.keyboard to paste (more reliable than osascript for LaunchAgent)
            from pynput.keyboard import Controller, Key

            kbd = Controller()

            # Small delay to ensure clipboard is ready
            time.sleep(0.1)

            # Press Cmd+V to paste
            with kbd.pressed(Key.cmd):
                kbd.press("v")
                kbd.release("v")

            # Restore original clipboard after a short delay
            threading.Timer(1.0, lambda: pyperclip.copy(original_clipboard)).start()

            if not self.silent:
                print("✅ Text inserted successfully")

        except Exception as e:
            if not self.silent:
                print(f"✗ Failed to insert text: {e}")
                print(f"📋 Text copied to clipboard: {text}")

    def _on_hotkey_pressed(self):
        """Handle hotkey press"""
        if self.recording:
            print("\n⏹️  Stopping recording...")
            self.stop_recording()
        else:
            threading.Thread(target=self.start_recording, daemon=True).start()

    def start_listening(self):
        """Start listening for double-tap of right Option key to record"""
        if not self.silent:
            print("🎧 Listening for right Option key double-tap to start recording...")
            print("🔁 Double-tap → start recording")
            print("⏹️  Single tap (while recording) → stop recording")
            print("🛑 Ctrl+C to quit")

        last_tap_time = [0]
        tap_threshold = 0.5  # seconds between taps to count as a double tap

        def on_press(key):
            try:
                if key == keyboard.Key.alt_r:  # right Option key
                    now = time.time()
                    time_since_last = now - last_tap_time[0]

                    if not self.recording and time_since_last < tap_threshold:
                        # Double-tap detected → start recording
                        if not self.silent:
                            print("\n🎙️ Double-tap detected → start recording")
                        threading.Thread(
                            target=self.start_recording, daemon=True
                        ).start()
                    elif self.recording:
                        # Single tap while recording → stop
                        if not self.silent:
                            print("\n⏹️ Single tap detected → stop recording")
                        self.stop_recording()

                    last_tap_time[0] = now
            except Exception as e:
                if not self.silent:
                    print(f"⚠️ Listener error: {e}")

        with keyboard.Listener(on_press=on_press) as listener:
            self.listener = listener
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")

    def cleanup(self):
        """Cleanup resources"""
        if self.recording:
            self.stop_recording()
        if self.listener:
            self.listener.stop()


def main():
    """Main function"""
    print("🎤 macOS Whisper Dictation App")
    print("=" * 40)

    try:
        app = WhisperDictationApp()
        app.start_listening()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    finally:
        try:
            app.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
