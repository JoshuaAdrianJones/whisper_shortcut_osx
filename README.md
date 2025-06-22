# macOS Whisper Dictation App

A local speech-to-text application using OpenAI's Whisper model with keyboard shortcuts for macOS.

> **Note**: This project was created using [Claude Code](https://claude.ai/code) to provide a free alternative to paid dictation apps.

## Features

- **Local Processing**: Complete privacy - no data sent to external servers
- **Manual Recording Control**: Start/stop recording with keyboard shortcut
- **macOS Integration**: Native text insertion that works with any application
- **Performance Optimized**: Uses faster-whisper with CPU optimization
- **Keyboard Shortcut**: Cmd+Shift+S to toggle recording
- **Simple Operation**: Press once to start, press again to stop

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install faster-whisper pynput pyperclip sounddevice webrtcvad numpy
```

### 2. macOS Permissions Setup

#### Accessibility Permissions
1. Open **System Preferences** → **Security & Privacy** → **Privacy**
2. Select **Accessibility** from the left sidebar
3. Click the lock icon and enter your password
4. Click the **+** button and add:
   - **Terminal** (if running from Terminal)
   - **Your IDE** (if running from VS Code, PyCharm, etc.)
   - Or **Python** executable (if running directly)

#### Microphone Permissions
1. In the same **Privacy** section, select **Microphone**
2. Add the same applications as above
3. Ensure they're checked/enabled

### 3. Test Audio Setup (Optional)

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

## Usage

### Running the App

```bash
python3 whisper_dictation.py
```

### Controls

- **Cmd+Shift+S**: Start/stop recording
- **Ctrl+C**: Quit the application

### Recording Behavior

1. Press the hotkey to start recording
2. Speak naturally - recording continues until you press the hotkey again
3. Press the hotkey again to stop recording
4. Text is automatically processed and inserted at your cursor position

## Configuration

You can modify these settings in the script:

```python
# Model size (tiny, base, small, medium, large)
self.model_size = "small"  # Default: good balance of speed/accuracy

# Audio settings
self.sample_rate = 16000  # Audio sample rate
```

## Troubleshooting

### Common Issues

#### "Permission denied" or hotkeys not working
- **Solution**: Ensure accessibility permissions are granted
- **Check**: System Preferences → Security & Privacy → Privacy → Accessibility

#### Microphone not detected
- **Solution**: Grant microphone permissions
- **Check**: System Preferences → Security & Privacy → Privacy → Microphone

#### Text not inserting
- **Solution**: The app falls back to clipboard if direct insertion fails
- **Workaround**: Manually paste with Cmd+V if needed

#### Poor transcription quality
- **Solutions**:
  - Use a better model: Change `model_size` to "small" or "medium"
  - Improve audio quality: Use external microphone, reduce background noise
  - Speak clearly and at moderate pace

#### "No module named" errors
- **Solution**: Install missing dependencies
```bash
pip install [missing_package_name]
```

## Performance Tips

### Model Selection
- `tiny`: Fastest, least accurate
- `base`: Good balance
- `small`: Better accuracy, slower (current default)
- `medium/large`: Best accuracy, much slower

### Audio Quality
- Use external microphone when possible
- Minimize background noise
- Speak at normal volume and pace

## Advanced Configuration

### Custom Hotkey

To change the hotkey combination, modify this line in the script:
```python
'<cmd>+<shift>+s': self._on_hotkey_pressed
```

Example alternatives:
- `'<cmd>+<shift>+<space>'`: Cmd+Shift+Space
- `'<ctrl>+<alt>+<space>'`: Ctrl+Option+Space
- `'<cmd>+<alt>+<space>'`: Cmd+Option+Space

### Language Support

Whisper supports multiple languages. To specify a language:
```python
segments, info = self.model.transcribe(
    audio_file,
    language="en",  # or "es", "fr", "de", etc.
)
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