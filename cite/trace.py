"""Append every question and its full result to a daily JSON lines file, so any answer can be replayed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import Result


def write_trace(result: Result, directory: str | Path = "data/traces") -> Path:
    now = datetime.now(timezone.utc)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{now:%Y-%m-%d}.jsonl"
    with file.open("a") as f:
        f.write(json.dumps({"time": now.isoformat(timespec="seconds"), **result.to_dict()}) + "\n")
    return file
