"""
Configuration management for the RAG Intelligent Agent.
Handles loading environment variables and providing access to configuration values.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Import default configuration
from config.default import (
    GOOGLE_API_KEY, 
    QDRANT_API_KEY, 
    QDRANT_URL,
    MAX_CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    SUPPORTED_FORMATS,
    MAX_DOCUMENT_SIZE_MB,
    EMBEDDING_MODEL, 
    LLM_MODEL, 
    TOP_K_RESULTS,
    RELEVANCE_THRESHOLD,
    DEBUG, 
    LOG_LEVEL,
    CACHE_EXPIRY
)

# Set up logger
logger = logging.getLogger(__name__)

# Define required configuration keys
REQUIRED_CONFIG = [
    'GOOGLE_API_KEY',
    'QDRANT_API_KEY',
    'QDRANT_URL'
]

# Configuration schema with descriptions for documentation
CONFIG_SCHEMA = {
    'GOOGLE_API_KEY': 'API key for Google Generative AI (Gemini)',
    'QDRANT_API_KEY': 'API key for Qdrant Cloud vector database',
    'QDRANT_URL': 'URL for your Qdrant Cloud instance',
    'MAX_CHUNK_SIZE': 'Maximum size of document chunks in characters',
    'CHUNK_OVERLAP': 'Overlap between adjacent chunks in characters',
    'SUPPORTED_FORMATS': 'List of supported document formats',
    'MAX_DOCUMENT_SIZE_MB': 'Maximum document size in megabytes',
    'EMBEDDING_MODEL': 'Name of the embedding model to use',
    'LLM_MODEL': 'Name of the LLM model to use for response generation',
    'TOP_K_RESULTS': 'Number of document chunks to retrieve for each query',
    'RELEVANCE_THRESHOLD': 'Minimum relevance score for retrieved chunks',
    'DEBUG': 'Enable debug mode (True/False)',
    'LOG_LEVEL': 'Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
    'CACHE_EXPIRY': 'Cache expiry time in seconds'
}


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class Config:
    """Configuration manager for the application."""
    
    _instance = None
    _config = {}
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one config instance exists."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        if not self._initialized:
            self._load_config()
            self._initialized = True
    
    def _load_config(self):
        """Load configuration from environment variables and default values."""
        # Load environment variables from .env file if it exists
        dotenv_path = Path('.env')
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.info("Loaded environment variables from .env file")
        else:
            logger.warning(".env file not found. Using environment variables and defaults.")
            
        # Load configuration values
        self._config = {
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY', GOOGLE_API_KEY),
            'QDRANT_API_KEY': os.getenv('QDRANT_API_KEY', QDRANT_API_KEY),
            'QDRANT_URL': os.getenv('QDRANT_URL', QDRANT_URL),
            'MAX_CHUNK_SIZE': int(os.getenv('MAX_CHUNK_SIZE', MAX_CHUNK_SIZE)),
            'CHUNK_OVERLAP': int(os.getenv('CHUNK_OVERLAP', CHUNK_OVERLAP)),
            'SUPPORTED_FORMATS': os.getenv('SUPPORTED_FORMATS', SUPPORTED_FORMATS),
            'MAX_DOCUMENT_SIZE_MB': float(os.getenv('MAX_DOCUMENT_SIZE_MB', MAX_DOCUMENT_SIZE_MB)),
            'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', EMBEDDING_MODEL),
            'LLM_MODEL': os.getenv('LLM_MODEL', LLM_MODEL),
            'TOP_K_RESULTS': int(os.getenv('TOP_K_RESULTS', TOP_K_RESULTS)),
            'RELEVANCE_THRESHOLD': float(os.getenv('RELEVANCE_THRESHOLD', RELEVANCE_THRESHOLD)),
            'DEBUG': self._parse_bool(os.getenv('DEBUG', str(DEBUG))),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', LOG_LEVEL),
            'CACHE_EXPIRY': int(os.getenv('CACHE_EXPIRY', CACHE_EXPIRY))
        }
        
        # Handle special case for SUPPORTED_FORMATS if it's passed as a string
        if isinstance(self._config['SUPPORTED_FORMATS'], str):
            self._config['SUPPORTED_FORMATS'] = [
                fmt.strip() for fmt in self._config['SUPPORTED_FORMATS'].split(',')
            ]
            
        # Set log level based on configuration
        logging.basicConfig(
            level=getattr(logging, self._config['LOG_LEVEL']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Validate required configuration
        self._validate_config()
    
    def _parse_bool(self, value: str) -> bool:
        """Parse string to boolean."""
        return value.lower() in ('true', 't', 'yes', 'y', '1')
    
    def _validate_config(self):
        """Validate that all required configuration is present."""
        missing = []
        for key in REQUIRED_CONFIG:
            if self._config.get(key) is None:
                missing.append(key)
        
        if missing:
            error_msg = f"Missing required configuration: {', '.join(missing)}. "
            
            # Add specific guidance for missing Qdrant configuration
            if 'QDRANT_URL' in missing or 'QDRANT_API_KEY' in missing:
                error_msg += "\nPlease check your Qdrant Cloud settings in the .env file. "
                error_msg += "For Qdrant Cloud, ensure QDRANT_URL is the full URL including https:// "
                error_msg += "and QDRANT_API_KEY contains your API key."
            
            error_msg += "\nPlease check your .env file or environment variables."
            raise ConfigurationError(error_msg)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def get_schema(self) -> Dict[str, str]:
        """Get the configuration schema with descriptions."""
        return CONFIG_SCHEMA
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get API keys configuration."""
        return {
            'GOOGLE_API_KEY': self._config.get('GOOGLE_API_KEY'),
            'QDRANT_API_KEY': self._config.get('QDRANT_API_KEY')
        }
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration."""
        return {
            'QDRANT_URL': self._config.get('QDRANT_URL'),
            'QDRANT_API_KEY': self._config.get('QDRANT_API_KEY')
        }
    
    def get_document_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration."""
        return {
            'MAX_CHUNK_SIZE': self._config.get('MAX_CHUNK_SIZE'),
            'CHUNK_OVERLAP': self._config.get('CHUNK_OVERLAP'),
            'SUPPORTED_FORMATS': self._config.get('SUPPORTED_FORMATS'),
            'MAX_DOCUMENT_SIZE_MB': self._config.get('MAX_DOCUMENT_SIZE_MB')
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG model configuration."""
        return {
            'EMBEDDING_MODEL': self._config.get('EMBEDDING_MODEL'),
            'LLM_MODEL': self._config.get('LLM_MODEL'),
            'TOP_K_RESULTS': self._config.get('TOP_K_RESULTS'),
            'RELEVANCE_THRESHOLD': self._config.get('RELEVANCE_THRESHOLD')
        }


# Create a singleton instance
config = Config()


# Utility functions for accessing configuration
def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    return config.get(key, default)


def get_api_key(service: str) -> Optional[str]:
    """Get an API key for a specific service."""
    if service.lower() == 'google':
        return config.get('GOOGLE_API_KEY')
    elif service.lower() == 'qdrant':
        return config.get('QDRANT_API_KEY')
    return None


def get_document_processing_config() -> Dict[str, Any]:
    """Get document processing configuration."""
    return config.get_document_processing_config()


def get_rag_config() -> Dict[str, Any]:
    """Get RAG model configuration."""
    return config.get_rag_config()


def get_qdrant_config() -> Dict[str, Any]:
    """Get Qdrant configuration."""
    return config.get_qdrant_config()


def get_all_config() -> Dict[str, Any]:
    """Get all configuration values."""
    return config.get_all()


# For testing purposes
def test_config():
    """Test the configuration system by validating all required values are present."""
    try:
        config = Config()
        config._validate_config()
        print("✅ Configuration validation passed.")
        
        # Print some configuration values for verification
        print("\nConfiguration values:")
        print(f"Google API Key: {'*' * 8 if config.get('GOOGLE_API_KEY') else 'Not configured'}")
        print(f"Qdrant API Key: {'*' * 8 if config.get('QDRANT_API_KEY') else 'Not configured'}")
        print(f"Qdrant URL: {config.get('QDRANT_URL')}")
        print(f"Debug Mode: {config.get('DEBUG')}")
        print(f"Log Level: {config.get('LOG_LEVEL')}")
        print(f"Supported Formats: {config.get('SUPPORTED_FORMATS')}")
        
        return True
    except ConfigurationError as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    

if __name__ == "__main__":
    # This allows you to test the configuration by running this module directly
    test_config() 