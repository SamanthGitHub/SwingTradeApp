"""Crash-safe file persistence helpers.

Every JSON/text store in the app (watchlists, alerts, portfolio state, API-usage ledger,
trade journal, caches) writes through here. The write goes to a temp file **in the same
directory** and is swapped in with ``os.replace``, which is atomic on both POSIX and
Windows — a crash mid-write can never leave a half-written (corrupt) file, and a reader
in another session/tab always sees either the old or the new content, never a mix.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (same-directory temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Serialize ``obj`` to JSON and write it to ``path`` atomically."""
    atomic_write_text(Path(path), json.dumps(obj, indent=indent, default=str))


def read_json(path: Path, default: Any = None) -> Any:
    """Load JSON from ``path``; returns ``default`` on a missing, empty or corrupt file
    (a truncated file from a pre-atomic-write crash must never take the app down)."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default
