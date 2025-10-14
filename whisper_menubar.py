#!/usr/bin/env python3
import sys
import rumps
import threading
from whisper_dictation import WhisperDictationApp  # import your existing class

# Enable stdout/stderr for debugging
sys.stdout = open("/tmp/whispermenubar.out.log", "a", buffering=1)
sys.stderr = open("/tmp/whispermenubar.err.log", "a", buffering=1)
print("=== Whisper MenuBar Starting ===", flush=True)


class WhisperMenuBarApp(rumps.App):
    def __init__(self):
        super(WhisperMenuBarApp, self).__init__(
            "Whisper",  # App name
            quit_button="Quit",
        )
        self.title = "💬"  # Set menubar title

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
            None,  # Separator
            self.record_button,
            None,  # Separator
            "About",
        ]

        # Initialize whisper in a thread to avoid blocking
        threading.Thread(target=self._initialize_whisper, daemon=True).start()

    def _initialize_whisper(self):
        """Initialize the WhisperDictationApp in background"""
        try:
            self.status_item.title = "Status: Loading model..."
            self.whisper_app = WhisperDictationApp(silent=True)
            self.is_initialized = True
            self.status_item.title = "Status: Ready"
            rumps.notification(
                title="Whisper Dictation",
                subtitle="Ready to use",
                message="Double-tap right Option (⌥) to start recording",
            )
            # Start listening for keyboard shortcuts
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
            # Reset status after a delay
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
