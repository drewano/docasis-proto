"""
Test script to verify the configuration management system.

This script tests loading environment variables, validating required configuration,
and accessing configuration values through the Config class and utility functions.

Usage:
    python -m config.test_config
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    Config, 
    ConfigurationError,
    get_config,
    get_api_key,
    get_document_processing_config,
    get_rag_config,
    get_qdrant_config,
    get_all_config
)


def test_config_without_env():
    """Test config without environment variables."""
    print("\n🔍 Testing configuration without .env file...")
    
    # Backup any existing environment variables
    backup_google_api_key = os.environ.get('GOOGLE_API_KEY')
    backup_qdrant_api_key = os.environ.get('QDRANT_API_KEY')
    
    # Remove environment variables for testing
    if 'GOOGLE_API_KEY' in os.environ:
        del os.environ['GOOGLE_API_KEY']
    if 'QDRANT_API_KEY' in os.environ:
        del os.environ['QDRANT_API_KEY']
    
    # Create a new Config instance
    try:
        config = Config()
        print("✅ Default configuration loaded successfully")
        print(f"  Debug mode: {config.get('DEBUG')}")
        print(f"  Log level: {config.get('LOG_LEVEL')}")
        print(f"  Chunk size: {config.get('MAX_CHUNK_SIZE')}")
        
        # This should fail because required config is missing
        try:
            config._validate_config()
            print("❌ Validation should have failed but didn't")
        except ConfigurationError as e:
            print(f"✅ Validation failed as expected: {e}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
    
    # Restore environment variables
    if backup_google_api_key:
        os.environ['GOOGLE_API_KEY'] = backup_google_api_key
    if backup_qdrant_api_key:
        os.environ['QDRANT_API_KEY'] = backup_qdrant_api_key


def test_config_with_env():
    """Test config with temporary .env file."""
    print("\n🔍 Testing configuration with temporary .env file...")
    
    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_env:
        temp_env.write("GOOGLE_API_KEY=test_google_api_key\n")
        temp_env.write("QDRANT_API_KEY=test_qdrant_api_key\n")
        temp_env.write("QDRANT_URL=https://test-qdrant-url.qdrant.tech\n")
        temp_env.write("DEBUG=true\n")
        temp_env.write("LOG_LEVEL=DEBUG\n")
        temp_env.write("MAX_CHUNK_SIZE=500\n")
        temp_env.write("SUPPORTED_FORMATS=pdf,docx,txt,csv\n")
        temp_env_path = temp_env.name
    
    # Backup the original .env file if it exists
    original_env_path = Path('.env')
    backup_env_path = None
    if original_env_path.exists():
        backup_env_path = Path('.env.backup')
        original_env_path.rename(backup_env_path)
    
    # Move the temporary .env file to .env
    Path(temp_env_path).rename(original_env_path)
    
    try:
        # Reset Config singleton to force reloading configuration
        Config._instance = None
        Config._initialized = False
        
        # Test the config class
        config = Config()
        try:
            config._validate_config()
            print("✅ Configuration validation passed")
        except ConfigurationError as e:
            print(f"❌ Configuration validation failed: {e}")
        
        # Test utility functions
        google_api_key = get_api_key('google')
        qdrant_api_key = get_api_key('qdrant')
        doc_config = get_document_processing_config()
        all_config = get_all_config()
        
        print("\n📝 Configuration values:")
        print(f"  Google API Key: {'*' * 8 if google_api_key else 'Not configured'}")
        print(f"  Qdrant API Key: {'*' * 8 if qdrant_api_key else 'Not configured'}")
        print(f"  Debug Mode: {get_config('DEBUG')}")
        print(f"  Log Level: {get_config('LOG_LEVEL')}")
        print(f"  Chunk Size: {doc_config.get('MAX_CHUNK_SIZE')}")
        print(f"  Supported Formats: {doc_config.get('SUPPORTED_FORMATS')}")
        
        # Test that custom values were loaded correctly
        assert get_config('DEBUG') is True, "DEBUG should be True"
        assert get_config('LOG_LEVEL') == "DEBUG", "LOG_LEVEL should be DEBUG"
        assert get_config('MAX_CHUNK_SIZE') == 500, "MAX_CHUNK_SIZE should be 500"
        assert 'csv' in get_config('SUPPORTED_FORMATS'), "csv should be in SUPPORTED_FORMATS"
        
        print("\n✅ All configuration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error in configuration tests: {e}")
    finally:
        # Clean up and restore original .env file if it existed
        original_env_path.unlink(missing_ok=True)
        if backup_env_path and backup_env_path.exists():
            backup_env_path.rename(original_env_path)


if __name__ == "__main__":
    print("🧪 Testing Configuration Management System 🧪")
    print("============================================")
    
    test_config_without_env()
    test_config_with_env()
    
    print("\n============================================")
    print("🧪 Configuration Testing Complete 🧪") 