from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a formatted JSON file."""
    output_path = Path(path)
    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return its contents."""
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)