"""LaunchAgent lifecycle management for macOS."""

import logging
import os
import plistlib
import subprocess
from typing import Any

logger = logging.getLogger("whisper_dictation")

_LAUNCHAGENTS_DIR = os.path.expanduser("~/Library/LaunchAgents")


def _plist_path(label: str) -> str:
    return os.path.join(_LAUNCHAGENTS_DIR, f"{label}.plist")


def build_launchagent_plist(label: str, executable: str, script_path: str) -> dict[str, Any]:
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


def install(label: str, executable: str, script_path: str) -> None:
    """Write the plist and bootstrap the LaunchAgent.

    Raises:
        OSError: if writing the plist file fails.
        RuntimeError: if ``launchctl bootstrap`` returns non-zero.
    """
    path = _plist_path(label)
    plist = build_launchagent_plist(label, executable, script_path)
    os.makedirs(_LAUNCHAGENTS_DIR, exist_ok=True)
    with open(path, "wb") as f:
        plistlib.dump(plist, f)

    uid = os.getuid()
    proc = subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", path],
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(err or "launchctl bootstrap returned non-zero")


def uninstall(label: str) -> None:
    """Bootout the LaunchAgent and remove its plist.

    ``bootout`` failure is logged but not re-raised (the service may already be
    unloaded). Plist removal failure is also logged but not re-raised.
    """
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/{label}"],
        capture_output=True,
    )
    try:
        os.unlink(_plist_path(label))
    except OSError:
        logger.exception("failed to remove LaunchAgent plist")


def is_active(label: str) -> bool:
    """Return True if the LaunchAgent is currently loaded."""
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        logger.exception("launchctl print failed")
        return False
