"""
RoboPacerV2 Dashboard - file browser for picking a model (.hef)
==================================================================
Backs the "Alege model" window on the main page: browse any folder on the
Pi (read-only listing) and pick a .hef file, like a desktop file-open
dialog. Only folders and .hef files are listed - that's all that can be
picked.

resolve_hef() is what /api/start trusts, never the client: the path must be
absolute, resolve to a real readable file, and end in .hef.
"""

import os

MAX_ENTRIES = 1000
_HIDE_AT_ROOT = {"proc", "sys", "dev"}  # pseudo-filesystems: thousands of entries, never a model


def resolve_hef(path):
    """Real path of a readable .hef file, or None."""
    if not isinstance(path, str) or not os.path.isabs(path) or "\x00" in path:
        return None
    real = os.path.realpath(path)
    if real.lower().endswith(".hef") and os.path.isfile(real) and os.access(real, os.R_OK):
        return real
    return None


def list_dir(path):
    """({"path", "parent", "entries", "truncated"}, None) or (None, error).
    entries: folders first, then .hef files, each sorted by name; hidden
    (dot) files are skipped."""
    if not isinstance(path, str) or not os.path.isabs(path) or "\x00" in path:
        return None, "invalid_path"
    real = os.path.realpath(path)
    if not os.path.isdir(real):
        return None, "not_a_directory"
    dirs, hefs = [], []
    try:
        with os.scandir(real) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                if real == "/" and entry.name in _HIDE_AT_ROOT:
                    continue
                try:
                    if entry.is_dir():  # follows symlinks
                        dirs.append({"name": entry.name, "type": "dir"})
                    elif entry.name.lower().endswith(".hef") and entry.is_file():
                        hefs.append({"name": entry.name, "type": "hef",
                                     "size": entry.stat().st_size})
                except OSError:
                    continue
    except PermissionError:
        return None, "permission_denied"
    except OSError:
        return None, "unreadable"
    key = lambda e: e["name"].lower()
    entries = sorted(dirs, key=key) + sorted(hefs, key=key)
    truncated = len(entries) > MAX_ENTRIES
    return {
        "path": real,
        "parent": None if real == "/" else os.path.dirname(real),
        "entries": entries[:MAX_ENTRIES],
        "truncated": truncated,
    }, None
