# System Architecture

This document outlines the high-level architecture of the RAG Intelligent Agent system.

## Architecture Overview

The RAG Intelligent Agent follows a modular architecture with clear separation of concerns. The system is designed to be extensible, maintainable, and testable.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI Interface                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG Pipeline Module                        │
└───────┬───────────────────────┬──────────────────────┬──────────┘
        │                       │                      │
        ▼                       ▼                      ▼
┌───────────────┐      ┌────────────────┐     ┌────────────────┐
│  Document     │      │   Embedding    │     │     Vector     │
│  Processing   │──────│   Module       │─────│     Storage    │
│  Module       │      │                │     │     Module     │
└───────────────┘      └────────────────┘     └────────────────┘
        │                       │                      │
        ▼                       ▼                      ▼
┌───────────────┐      ┌────────────────┐     ┌────────────────┐
│   Docling     │      │  Google Text   │     │    Qdrant      │
│   Library     │      │   Embedding    │     │     Cloud      │
└───────────────┘      └────────────────┘     └────────────────┘
```

## Component Descriptions

### 1. Streamlit UI Interface

- **Purpose**: Provides the user interface for document upload, query input, and results display
- **Location**: `src/ui/streamlit_app.py`
- **Key Features**:
  - Three-panel layout (document management, query interface, relevance dashboard)
  - Document upload with progress indicators
  - Query input and response display
  - Relevance visualization

### 2. RAG Pipeline Module

- **Purpose**: Orchestrates the retrieval and generation process
- **Location**: `src/models/rag_model.py`
- **Key Features**:
  - Integration with LangChain
  - Query processing
  - Document retrieval
  - Response generation
  - Context management

### 3. Document Processing Module

- **Purpose**: Handles document parsing, transformation, and chunking
- **Location**: `src/data/document_processor.py`
- **Key Features**:
  - Support for multiple document formats (PDF, DOCX, TXT)
  - Text extraction
  - Document chunking with overlap
  - Metadata extraction

### 4. Embedding Module

- **Purpose**: Generates vector embeddings for document chunks
- **Location**: `src/models/embedding.py`
- **Key Features**:
  - Integration with Google's text-embedding-004 model
  - Batch processing
  - Caching mechanism
  - Error handling for API failures

### 5. Vector Storage Module

- **Purpose**: Manages interactions with Qdrant Cloud
- **Location**: `src/data/vector_store.py`
- **Key Features**:
  - Collection management
  - Vector upload and retrieval
  - Similarity search
  - Connection pooling

## Data Flow

1. **Document Ingestion Flow**:
   - User uploads document → Document Processing Module parses and chunks the document
   - Chunks sent to Embedding Module → Vector embeddings generated
   - Embeddings sent to Vector Storage Module → Stored in Qdrant Cloud

2. **Query Processing Flow**:
   - User inputs query → RAG Pipeline preprocesses query
   - Query converted to embedding → Similar document chunks retrieved from Vector Storage
   - Retrieved chunks used as context → LLM generates response
   - Response and relevance data displayed to user

## Extensibility

The architecture is designed for extensibility:

- **New Document Types**: Add new document processors to the Document Processing Module
- **Alternative Embedding Models**: Swap the embedding provider in the Embedding Module
- **Different Vector Databases**: Create new implementations of the Vector Storage interface
- **UI Customization**: Modify the Streamlit UI without affecting the core functionality

## Security Considerations

- API keys are managed through environment variables
- Document data is processed locally when possible
- Cloud services (Google AI, Qdrant) are accessed securely
- No user data is stored beyond the current session unless explicitly saved 