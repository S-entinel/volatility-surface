"""
Centralized logging utility for the Volatility Surface program.

Provides consistent logging format across all modules with complete type hints:
[timestamp] [module_name] [level] message
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Fetching market data")
        2025-12-09 10:30:45 - module.name - INFO - Fetching market data
    """
    logger: logging.Logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create console handler
    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter: logging.Formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def set_log_level(logger: logging.Logger, level: int) -> None:
    """
    Change the log level of an existing logger.
    
    Args:
        logger: Logger instance to modify
        level: New logging level (e.g., logging.DEBUG, logging.WARNING)
        
    Returns:
        None
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> set_log_level(logger, logging.DEBUG)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def get_logger(name: str) -> Optional[logging.Logger]:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name to retrieve
        
    Returns:
        Logger instance if it exists, None otherwise
        
    Example:
        >>> logger = get_logger('src.data.market_data')
        >>> if logger:
        ...     logger.info("Using existing logger")
    """
    logger = logging.getLogger(name)
    return logger if logger.handlers else None