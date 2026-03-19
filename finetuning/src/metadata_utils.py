from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_metadata_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix(path.suffix + ".metadata.json")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_counts(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def atomic_to_excel(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_excel(tmp_path, index=False, **kwargs)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def read_excel_with_retry(
    path: str | Path,
    *,
    attempts: int = 5,
    delay_seconds: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_excel(path, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay_seconds * attempt)
    assert last_error is not None
    raise last_error
