# RAG Intelligent Agent

A Retrieval-Augmented Generation (RAG) system that enhances the capabilities of Large Language Models (LLMs) by providing them with accurate, up-to-date, and relevant information from a document database.

## Features

- **Document Management**: Upload, process, and index documents using Docling and Qdrant
- **RAG Pipeline**: Enhance Gemini LLM responses with relevant document context using LangChain
- **Interactive UI**: Intuitive Streamlit interface for document management and interaction
- **Relevance Dashboard**: Evaluate response quality with document citations and relevance metrics
- **Document Cleanup**: Simple mechanism to delete all documents from the database

## Tech Stack

- **Language**: Python 3.10+
- **RAG Framework**: LangChain
- **Vector Database**: Qdrant Cloud
- **Document Processing**: Docling
- **LLM**: Google Gemini Flash (version 20)
- **Embeddings**: Google text-embedding-004
- **UI**: Streamlit

## System Architecture

The system follows a modular architecture with the following components:

1. **Document Processing Module**: Uses Docling to parse, transform, and chunk documents
2. **Embedding Module**: Processes document chunks using text-embedding-004
3. **Vector Storage Module**: Manages the interaction with Qdrant Cloud
4. **RAG Pipeline Module**: Orchestrates the retrieval and generation process using LangChain
5. **UI Module**: Provides the Streamlit interface for user interaction
6. **Relevance Dashboard Module**: Displays document relevance metrics for generated responses

## Prerequisites

- Python 3.10 or higher
- Qdrant Cloud account
- Google AI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-intelligent-agent.git
   cd rag-intelligent-agent
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_URL=your_qdrant_url
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run src/ui/streamlit_app.py
   ```

2. Use the left panel to upload and manage documents
3. Enter natural language queries in the center panel
4. View document relevance metrics in the right panel

## Project Structure

```
.
├── src/                    # Source code
│   ├── models/             # Model-related code
│   ├── data/               # Data processing utilities
│   ├── utils/              # Helper functions
│   ├── api/                # API interfaces
│   └── ui/                 # Streamlit UI code
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── .env.example            # Example environment variables
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG framework
- Qdrant for vector database capabilities
- Docling for document processing
- Google for Gemini LLM and embedding models
- Streamlit for the UI framework 