"""Path and small file-loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias


PathInput: TypeAlias = str | Path


def to_abs_path(path: PathInput, *, base_dir: PathInput | None = None) -> Path:
    """Resolve one path-like input into an absolute path."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    anchor = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
    return (anchor / candidate).resolve()


def resolve_import_path(import_ref: PathInput, *, parent_config_path: PathInput) -> Path:
    """Resolve one `import_config` reference relative to the parent config path."""
    parent_config = to_abs_path(parent_config_path)
    ref = Path(import_ref).expanduser()
    if ref.is_absolute():
        candidates = [to_abs_path(ref)]
    else:
        candidates = [to_abs_path(parent_config.parent / ref)]
        candidates.extend(to_abs_path(parent / ref) for parent in parent_config.parents)
        candidates.append(to_abs_path(ref))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    for resolved in deduped:
        if resolved.exists():
            return resolved
    return deduped[0]


def load_dotenv(dotenv_path: PathInput = ".env") -> None:
    """Load key-value pairs from `.env` into the environment if not already set."""
    import os

    path = to_abs_path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
