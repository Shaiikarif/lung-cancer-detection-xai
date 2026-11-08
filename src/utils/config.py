"""Configuration loader module for PulmoAI framework."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration YAML file.

    Args:
        config_path: Relative or absolute path to the config file.
                     Defaults to configs/config.yaml in the project root.

    Returns:
        Dict containing project configuration settings.
    """
    root = get_project_root()
    if config_path is None:
        config_path = os.getenv("PULMO_CONFIG", str(root / "configs" / "config.yaml"))

    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure paths are resolved relative to root
    return config
