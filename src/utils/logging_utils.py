"""
Logging utilities for the RAG Intelligent Agent

This module provides logging functionality for the application.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any

# Default log level if not specified in config
DEFAULT_LOG_LEVEL = "INFO"

# Map string log levels to logging constants
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Create the application logger
app_logger = logging.getLogger("rag_agent")


def setup_logger(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Set up and configure the application logger.
    
    Args:
        config: Optional configuration dictionary with the following keys:
            - log_level: String log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            - debug: Boolean indicating whether debug mode is enabled
            
    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    app_logger.handlers = []
    
    # Get log level from config or use default
    log_level_str = config.get("log_level", DEFAULT_LOG_LEVEL) if config else DEFAULT_LOG_LEVEL
    log_level = LOG_LEVELS.get(log_level_str.upper(), logging.INFO)
    
    # Set the log level for the logger
    app_logger.setLevel(log_level)
    
    # Create console handler with formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    app_logger.addHandler(console_handler)
    
    # If in debug mode, also log to file
    is_debug = config.get("debug", False) if config else False
    if is_debug:
        # Create logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        # Create file handler which logs even debug messages
        file_handler = logging.FileHandler("logs/rag_agent.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        app_logger.addHandler(file_handler)
    
    # If this is the first time setup, log a startup message
    app_logger.info("Logger initialized with level %s", log_level_str)
    
    return app_logger 