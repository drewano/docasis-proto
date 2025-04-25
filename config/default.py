"""
Default configuration values for the RAG Intelligent Agent.
These values can be overridden by environment variables.
"""

# API Keys and Endpoints
GOOGLE_API_KEY = None  # Required for Google Generative AI (Gemini)
QDRANT_API_KEY = None  # Required for Qdrant Cloud
QDRANT_URL = None  # Required - Qdrant Cloud URL should be set in .env file

# Document Processing Settings
MAX_CHUNK_SIZE = 1000  # Maximum characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks in characters
SUPPORTED_FORMATS = ["pdf", "docx", "txt"]  # Supported document formats
MAX_DOCUMENT_SIZE_MB = 25  # Maximum document size in MB

# RAG Model Settings
EMBEDDING_MODEL = "models/text-embedding-004"  # Google's embedding model
LLM_MODEL = "gemini-2.0-flash"    # Gemini model for generation
TOP_K_RESULTS = 5  # Number of chunks to retrieve for each query
RELEVANCE_THRESHOLD = 0.7  # Minimum relevance score for retrieved chunks

# Application Settings
DEBUG = False
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
CACHE_EXPIRY = 3600  # Cache expiry time in seconds (1 hour) 