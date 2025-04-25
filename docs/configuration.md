# Configuration System Documentation

This document describes the configuration system for the RAG Intelligent Agent, including environment variables, configuration files, and how to use them properly.

## Overview

The RAG Intelligent Agent uses a robust configuration system that:
- Loads environment variables from `.env` files
- Provides default configurations in Python code
- Validates required configuration values
- Offers a clean API for accessing configuration throughout the application

## Environment Variables

The application uses the following environment variables which should be defined in a `.env` file (not committed to version control):

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | Yes | Google API key for Gemini and text-embedding-004 | `your_google_api_key` |
| `QDRANT_API_KEY` | Yes | API key for Qdrant Cloud | `your_qdrant_api_key` |
| `QDRANT_URL` | Yes | URL for your Qdrant Cloud instance | `https://your-cluster.qdrant.io` |
| `LOG_LEVEL` | No | Logging level (default: INFO) | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CHUNK_SIZE` | No | Document chunk size (default: 1000) | `500` |
| `CHUNK_OVERLAP` | No | Document chunk overlap (default: 200) | `100` |
| `MAX_DOCUMENTS` | No | Maximum documents to retrieve (default: 5) | `10` |

## Configuration Files

### 1. `.env` File

Create a `.env` file in the root directory with your specific configuration:

```
GOOGLE_API_KEY=your_google_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-cluster.qdrant.io
LOG_LEVEL=INFO
```

### 2. `config/default.py`

This file contains default configuration values that apply when environment variables are not provided. The structure is:

```python
# default.py
DEFAULT_CONFIG = {
    "app": {
        "name": "RAG Intelligent Agent",
        "version": "1.0.0",
    },
    "qdrant": {
        "collection_name": "documents",
        "timeout": 10.0,
    },
    "document_processing": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    # Other default configurations...
}
```

## Using Configuration in Code

The configuration system is implemented in `src/utils/config.py` and provides a clean interface for accessing configuration values:

```python
from src.utils.config import get_config

# Get a specific configuration value
api_key = get_config("google.api_key")  # From GOOGLE_API_KEY env var

# Get a value with a default fallback
chunk_size = get_config("document_processing.chunk_size", default=1000)

# Check if a configuration exists
has_qdrant_config = has_config("qdrant.api_key")
```

## Configuration Validation

The configuration system validates required values on application startup. If required configuration is missing, the application will log an error and exit with a helpful message about which configuration values are missing.

## Best Practices

1. **Never commit `.env` files** to version control
2. Use `.env.example` as a template showing required variables without actual values
3. Use descriptive configuration keys in a hierarchical structure
4. Provide sensible defaults for non-critical configuration
5. Always validate required configuration early in application startup
6. Access configuration through the provided API rather than using `os.environ` directly

## Testing with Configuration

For testing, you can use the configuration API's testing helpers:

```python
from src.utils.config import mock_config, reset_config

def test_something():
    # Set up test configuration
    with mock_config({"qdrant.url": "http://test-url"}):
        # Test code here
        pass
    # Configuration is reset after the context manager
```

This allows testing with specific configuration values without modifying environment variables. 