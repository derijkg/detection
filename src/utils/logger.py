# detection/src/utils/logger.py

import sys
import logging
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_DIR = Path("/home/gderijck/detection/outputs/logs")


def setup_logger(
    name: str = "detection",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configures a logger with console stream output and optional log file output.
    All relative log file paths default to /home/gderijck/detection/outputs/logs/.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger is instantiated multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = DEFAULT_LOG_DIR / log_path
        
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger