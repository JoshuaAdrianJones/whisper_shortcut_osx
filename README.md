# macOS Whisper Dictation App

A local speech-to-text application using OpenAI's Whisper model with keyboard shortcuts for macOS.

> **Note**: This project was created using [Claude Code](https://claude.ai/code) to provide a free alternative to paid dictation apps.

## Features

- **Local Processing**: Complete privacy - no data sent to external servers
- **Menubar App**: Runs in your macOS menubar with visual status indicators
- **Keyboard Shortcuts**: Double-tap right Option (⌥) to start recording, single tap to stop
- **macOS Integration**: Automatic text insertion at cursor position in any application
- **Performance Optimized**: Uses faster-whisper with CPU optimization

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install faster-whisper pynput pyperclip sounddevice numpy rumps
```

### 2. macOS Permissions Setup

#### Accessibility Permissions
1. Open **System Settings** → **Privacy & Security** → **Accessibility**
2. Click the **+** button and add:
   - **Terminal** (if running from Terminal)
   - **Your IDE** (if running from VS Code, PyCharm, etc.)
   - Or **Python** executable (if running directly)

#### Microphone Permissions
1. In **Privacy & Security**, select **Microphone**
2. Add the same applications as above
3. Ensure they're checked/enabled

## Usage

### Running the Menubar App (Recommended)

```bash
python3 whisper_menubar.py
```

The app appears in your menubar with a 🎤 icon that changes to 🔴 when recording.

### Running the Command-Line Version

```bash
python3 whisper_dictation.py
```

### Controls

- **Double-tap right Option (⌥⌥)**: Start recording
- **Single tap right Option (⌥)**: Stop recording
- **Menubar Menu**: Click the 🎤 icon to start/stop or view status
- **Ctrl+C**: Quit the application

### How It Works

1. Double-tap the right Option (⌥) key to start recording
2. Speak naturally - the 🎤 icon changes to 🔴 while recording
3. Single tap the right Option key to stop
4. Text is automatically transcribed and inserted at your cursor position

## Configuration

You can modify these settings in `whisper_dictation.py`:

```python
# Model size (tiny, base, small, medium, large)
self.model_size = "base"  # Default: fast with good accuracy

# Audio settings
self.sample_rate = 16000  # Audio sample rate
```

### Model Selection
- `tiny`: Fastest, least accurate
- `base`: Good balance (current default)
- `small`: Better accuracy, slower
- `medium/large`: Best accuracy, much slower

## Troubleshooting

### Common Issues

#### Menubar icon not showing
- Make sure you're running `python3 whisper_menubar.py`
- Check that rumps is properly installed: `pip show rumps`
- View logs at `/tmp/whispermenubar.err.log` and `/tmp/whispermenubar.out.log`

#### "Permission denied" or hotkeys not working
- **Solution**: Ensure accessibility permissions are granted
- **Check**: System Settings → Privacy & Security → Accessibility

#### Microphone not detected
- **Solution**: Grant microphone permissions
- **Check**: System Settings → Privacy & Security → Microphone

#### Text not inserting
- **Solution**: The app falls back to clipboard if direct insertion fails
- **Workaround**: Manually paste with Cmd+V if needed

#### Poor transcription quality
- **Solutions**:
  - Use a better model: Change `model_size` to "small" or "medium"
  - Use an external microphone and reduce background noise
  - Speak clearly and at moderate pace

#### "No module named" errors
- **Solution**: Install missing dependencies
```bash
pip install [missing_package_name]
```

## Advanced Configuration

### Language Support

Whisper supports multiple languages. To specify a language, modify the transcription call in `whisper_dictation.py`:
```python
segments, info = self.model.transcribe(
    audio_file,
    language="en",  # or "es", "fr", "de", etc.
)
```

### Auto-Start on Login (Optional)

To launch the menubar app automatically at login, create a LaunchAgent plist file at `~/Library/LaunchAgents/com.user.whispermenubar.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.whispermenubar</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/your/whisper_menubar.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>/tmp/whispermenubar.err.log</string>
    <key>StandardOutPath</key>
    <string>/tmp/whispermenubar.out.log</string>
</dict>
</plist>
```

Then load it with:
```bash
launchctl load ~/Library/LaunchAgents/com.user.whispermenubar.plist
```

## System Requirements

- **macOS**: 10.14 (Mojave) or later
- **Python**: 3.8 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1-5GB for Whisper models
- **Microphone**: Built-in or external microphone

## Security and Privacy

- **Local Processing**: All speech processing happens on your device
- **No Network**: No data is sent to external servers
- **Temporary Files**: Audio files are automatically deleted after processing
- **Clipboard**: Original clipboard content is restored after text insertion

## Contributing

Feel free to open issues or submit pull requests! This project was created to provide a free alternative to paid dictation apps.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
