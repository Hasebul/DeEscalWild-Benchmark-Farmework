"""
logger.py — Application-wide logger configuration.
"""

import logging
import sys


def get_logger(name: str = "youtube_scraper") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:                         # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
