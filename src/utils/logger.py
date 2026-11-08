"""Logging setup for PulmoAI framework."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "pulmo_ai", level: int = logging.INFO) -> logging.Logger:
    """Return a pre-configured logger with clean formatting.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
