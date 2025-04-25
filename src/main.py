#!/usr/bin/env python3
"""
RAG Intelligent Agent - Main Entry Point

This module serves as the main entry point for the RAG Intelligent Agent application,
handling command-line arguments, configuration loading, and application initialization.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config

# Configure logging
def setup_logging(log_level):
    """Configure the application logging based on the provided log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RAG Intelligent Agent")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(Path(__file__).parent.parent / "config" / "default.py"),
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--ui", 
        action="store_true",
        help="Launch the Streamlit UI"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    try:
        # Configuration is loaded automatically when 'config' is imported
        logger.info("Configuration loaded successfully")
        
        # Check that required API keys are available using the imported config instance
        if not config.get('GOOGLE_API_KEY'):
            logger.error("Google API key not found in configuration")
            sys.exit(1)
            
        if not config.get('QDRANT_API_KEY'):
            logger.error("Qdrant API key not found in configuration")
            sys.exit(1)
        
        # Launch UI if requested
        if args.ui:
            logger.info("Launching Streamlit UI...")
            os.system(f"streamlit run {Path(__file__).parent}/ui/streamlit_app.py")
        else:
            logger.info("Starting in CLI mode (not implemented yet)")
            print("CLI mode not implemented yet. Use --ui to launch the Streamlit interface.")
            
    except Exception as e:
        logger.exception(f"Error in main application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 