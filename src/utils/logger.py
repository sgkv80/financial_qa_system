"""
logger.py

This module sets up a centralized logging utility for the Financial QA System.
It provides a reusable logger with both console and file handlers. The logging 
level and file location can be configured via YAML configuration files.

Classes:
    LoggerManager:
        A class for setting up and managing a project-wide logger.

Functions:
    get_logger(name: str) -> logging.Logger:
        Returns a logger instance with a given name.

Example:
    >>> from utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Logger is working!")
"""

import logging
import os
from logging.handlers import RotatingFileHandler


class LoggerManager:
    """
    A manager for creating and configuring loggers for the project.

    Attributes:
        log_file (str): The path to the log file.
        log_level (str): The logging level (e.g., "DEBUG", "INFO").
        max_bytes (int): Maximum size (in bytes) of each log file before rotation.
        backup_count (int): Number of backup log files to keep.

    Methods:
        get_logger(name: str) -> logging.Logger:
            Creates or retrieves a logger with specified settings.
    """

    def __init__(self, log_file: str = "logs/system.log", log_level: str = "INFO",
                 max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3):
        """
        Initialize the LoggerManager with file path and settings.

        Args:
            log_file (str): Path to the log file.
            log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
            max_bytes (int): Maximum log file size in bytes before rotation.
            backup_count (int): Number of rotated logs to keep.
        """
        self.log_file = log_file
        self.log_level = log_level.upper()
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Create and return a logger with console and file handlers.

        Args:
            name (str): The name of the logger (usually `__name__`).

        Returns:
            logging.Logger: A configured logger instance.

        Example:
            >>> logger = LoggerManager().get_logger(__name__)
            >>> logger.info("Hello from the logger!")
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.log_level, logging.INFO))

        if not logger.handlers:  # Avoid duplicate handlers
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))

            # File handler with rotation
            file_handler = RotatingFileHandler(
                self.log_file, maxBytes=self.max_bytes, backupCount=self.backup_count
            )
            file_handler.setFormatter(logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger


def get_logger(name: str) -> logging.Logger:
    """
    A shortcut function to quickly get a logger instance with default settings.

    Args:
        name (str): The name of the logger (usually `__name__`).

    Returns:
        logging.Logger: A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.warning("This is a warning")
    """
    manager = LoggerManager()
    return manager.get_logger(name)
