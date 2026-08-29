"""Per-task library-usage knobs (yaml). Mirrors ``utils/mdl/configs``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from ..counter import LibraryUsageConfig

CONFIG_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_usage_config(name: str) -> LibraryUsageConfig:
    """Load a yaml config by name. Falls back to defaults when missing."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        return LibraryUsageConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = LibraryUsageConfig()
    return LibraryUsageConfig(
        lib_package_name=data.get("lib_package_name", defaults.lib_package_name),
        code_extensions=tuple(data.get("code_extensions") or defaults.code_extensions),
        ignore_dirs=tuple(data.get("ignore_dirs") or defaults.ignore_dirs),
    )


__all__ = ["load_usage_config", "CONFIG_DIR"]
