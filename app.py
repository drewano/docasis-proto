#!/usr/bin/env python3
"""
RAG Intelligent Agent - Application Entry Point (DocAsis)

Streamlit UI for interacting with a RAG pipeline. Allows document upload,
processing, querying, and viewing cited sources.
"""
# Import standard libraries first
import sys
import time
import tempfile
import os
import uuid
import logging
from pathlib import Path

# Import heavy libraries like torch early, before Streamlit if possible
import torch

# Import Streamlit
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path to find 'src' modules
# Assumes app.py is in the root directory alongside the 'src' folder
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
logger.info(f"Added project root to sys.path: {project_root}")

# --- Import Project Modules ---
# It's generally better to import these *outside* UI interaction functions
# if they are needed globally or repeatedly. Handle potential errors during import.
try:
    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore

    from src.utils.config import Config
    from src.data.vector_store import VectorStore
    from src.data.document_processor import DocumentProcessor
    from src.models.embedding_processor import EmbeddingProcessor
    from src.models.rag_model import RAGPipeline
    logger.info("Successfully imported project modules.")
except ImportError as e:
    st.error(f"Failed to import necessary project modules: {e}")
    st.error("Please ensure all dependencies are installed (`pip install -r requirements.txt`) and the project structure is correct.")
    logger.error(f"ImportError: {e}", exc_info=True)
    st.stop() # Stop execution if core components are missing
except Exception as e:
    st.error(f"An unexpected error occurred during module import: {e}")
    logger.error(f"Unexpected ImportError: {e}", exc_info=True)
    st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DocAsis RAG Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Session State Initialization ---
# Use functions for initialization to keep it cleaner
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        'messages': [],
        'uploaded_files_info': [],
        'processing_status': "Ready", # Ready, Processing, Error, Complete
        'processing_complete': False, # Flag pour indiquer si le traitement a été terminé
        'response_text': "",
        'relevance_info': [],
        'config': None,
        'vector_store': None,
        'embedding_processor': None,
        'document_processor': None,
        'rag_pipeline': None
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    logger.debug("Session state initialized.")

# Call initialization function
initialize_session_state()

# --- Helper Function to Initialize Core Components ---
# Lazily initializes components and stores them in session state
def get_component(component_name: str):
    """Gets or initializes a core component (Config, VectorStore, etc.)."""
    if st.session_state.get(component_name) is None:
        logger.info(f"Initializing component: {component_name}...")
        try:
            if component_name == 'config':
                st.session_state.config = Config()
            elif component_name == 'vector_store':
                config = get_component('config') # Ensure config is loaded first
                if config:
                    st.session_state.vector_store = VectorStore(
                        collection_name=config.get('QDRANT_COLLECTION', 'rag_documents')
                    )
                    # Verify connection during initialization
                    st.session_state.vector_store.client.get_collections()
                    logger.info("Qdrant connection verified during VectorStore initialization.")
            elif component_name == 'embedding_processor':
                config = get_component('config')
                if config:
                    st.session_state.embedding_processor = EmbeddingProcessor(
                        model_name=config.get("EMBEDDING_MODEL", "models/text-embedding-004"),
                        batch_size=config.get("EMBEDDING_BATCH_SIZE", 100)
                    )
            elif component_name == 'document_processor':
                config = get_component('config')
                if config:
                    st.session_state.document_processor = DocumentProcessor(config=config.get_all())
            elif component_name == 'rag_pipeline':
                config = get_component('config')
                vector_store = get_component('vector_store')
                if config and vector_store:
                    st.session_state.rag_pipeline = RAGPipeline(
                        config=config,
                        vector_store=vector_store
                    )
            logger.info(f"Component '{component_name}' initialized successfully.")
        except ConnectionError as e:
            st.error(f"Qdrant Connection Error: {str(e)}. Please check QDRANT_URL and API key.")
            logger.error(f"Qdrant ConnectionError during {component_name} init: {e}", exc_info=True)
            st.session_state[component_name] = None # Ensure it remains None on error
            st.stop() # Stop if core connection fails
        except Exception as e:
            st.error(f"Error initializing {component_name}: {str(e)}")
            logger.error(f"Error initializing {component_name}: {e}", exc_info=True)
            st.session_state[component_name] = None # Ensure it remains None on error
            # Decide whether to stop based on component importance
            if component_name in ['config', 'vector_store']:
                 st.stop()

    return st.session_state.get(component_name)

# --- Main Application Title ---
st.title("📄 DocAsis - Intelligent RAG Agent")
st.caption("Upload documents, ask questions, and get answers with cited sources.")

# --- Three-Panel Layout ---
col1, col2, col3 = st.columns([1, 2, 1]) # Document Management | Query/Response | Relevance

# --- Column 1: Document Management ---
with col1:
    st.header("📚 Documents")
    with st.container(border=True):
        # File Uploader
        uploaded_files = st.file_uploader(
            "Upload Files (PDF, DOCX, TXT, MD, HTML, EPUB)",
            accept_multiple_files=True,
            # Get supported types dynamically from DocumentProcessor if available
            type=['.pdf', '.docx', '.txt'] if not get_component('document_processor') else 
                 getattr(get_component('document_processor'), 'SUPPORTED_EXTENSIONS', ['.pdf', '.docx', '.txt']),
            key="file_uploader" # Add key for stability
        )

        # Vérifier si le traitement est terminé, pour réinitialiser l'interface
        if st.session_state.get("processing_complete", False):
            # Réinitialiser le flag
            st.session_state.processing_complete = False
            # Continuer normalement sans essayer de modifier file_uploader
        
        # Processing Button and Logic
        process_button_disabled = st.session_state.processing_status == "Processing..."
        if st.button("Process Uploaded Files", disabled=process_button_disabled or not uploaded_files):
            if not uploaded_files:
                st.warning("Please select files to upload first.")
            else:
                st.session_state.processing_status = "Processing..."
                st.info("Initializing components for processing...")
                st.rerun() # Rerun to show status update and potentially disable button

        # Display processing status consistently
        st.info(f"Status: {st.session_state.processing_status}", icon="ℹ️")

        # Execute processing logic only when status is "Processing..."
        # This prevents re-processing on every interaction after the button is clicked
        if st.session_state.processing_status == "Processing...":
            with st.spinner("Processing documents... Please wait."):
                # Get necessary components (initializes them if needed)
                vector_store = get_component('vector_store')
                embedding_processor = get_component('embedding_processor')
                doc_processor = get_component('document_processor')

                if not all([vector_store, embedding_processor, doc_processor]):
                    st.error("Core processing components failed to initialize. Cannot proceed.")
                    st.session_state.processing_status = "Error"
                    st.rerun()
                else:
                    processed_count = 0
                    error_count = 0
                    total_chunks_processed = 0
                    processed_file_names = [] # Keep track of successfully processed files

                    # Get list of files currently selected in the uploader
                    # We need to re-access the uploader state via its key
                    current_uploads = st.session_state.get("file_uploader", [])
                    if not current_uploads:
                         st.warning("No files found in uploader state. Please re-select files if needed.")
                         st.session_state.processing_status = "Ready" # Reset status
                         st.rerun()

                    for uploaded_file in current_uploads:
                        file_display_name = uploaded_file.name
                        logger.info(f"Starting processing for: {file_display_name}")
                        tmp_file_path = None # Define outside try block for finally
                        try:
                            # 1. Save file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            logger.debug(f"Saved uploaded file to temporary path: {tmp_file_path}")

                            # 2. Process document into chunks
                            file_analysis_status_label = f"Analyzing {file_display_name}..."
                            st.text(file_analysis_status_label)  # Simple text instead of st.status
                            
                            # Check if doc_processor is properly initialized
                            if not doc_processor:
                                st.error(f"Document processor not initialized properly for {file_display_name}.")
                                logger.error("Document processor is None when trying to process document")
                                chunks = None
                            else:
                                chunks = doc_processor.process_document(tmp_file_path)

                            if not chunks:
                                st.warning(f"Could not extract text/chunks from {file_display_name}. Skipping.")
                                logger.warning(f"No chunks extracted from {file_display_name}")
                                error_count += 1
                                continue # Skip to the next file

                            st.text(f"Extracted {len(chunks)} chunks from {file_display_name}")
                            logger.info(f"Extracted {len(chunks)} chunks from {file_display_name}")

                            # 3. Convert chunks to LangChain Documents
                            langchain_docs = [
                                Document(
                                    page_content=chunk['content'],
                                    # Merge metadata, ensuring source is just the filename
                                    metadata={
                                        **chunk['metadata'], # Include all original metadata
                                        'source': Path(chunk['metadata'].get('source', file_display_name)).name # Overwrite source with filename
                                    }
                                ) for chunk in chunks if chunk.get('content') # Ensure content exists
                            ]

                            if not langchain_docs:
                                 st.warning(f"No valid content found in chunks for {file_display_name}. Skipping indexation.")
                                 logger.warning(f"No valid LangChain documents created for {file_display_name}")
                                 error_count += 1
                                 continue

                            # 4. Embed and Store in Qdrant using LangChain integration
                            logger.info(f"Attempting to index {len(langchain_docs)} documents for {file_display_name} into Qdrant...")
                            
                            # Use status for indexing (not nested inside another status)
                            with st.status(f"Indexing {file_display_name} ({len(langchain_docs)} chunks)...") as index_status:
                                try:
                                    # Check if vector_store and embedding_processor are initialized
                                    if not vector_store:
                                        st.error("Vector store not initialized properly.")
                                        logger.error("Vector store is None when trying to index documents")
                                        raise ValueError("Vector store not initialized")

                                    # Ensure the collection exists
                                    vector_store.ensure_collection_exists()

                                    if not embedding_processor:
                                        st.error("Embedding processor not initialized properly.")
                                        logger.error("Embedding processor is None when trying to create embeddings")
                                        raise ValueError("Embedding processor not initialized")

                                    # Use a manual approach to generate embeddings and add them directly to Qdrant
                                    # This avoids serialization issues with QdrantVectorStore.from_documents() which can cause
                                    # "cannot pickle '_thread.RLock' object" errors
                                    
                                    # 1. Generate embeddings with our embedding model
                                    embedding_model = embedding_processor.get_embedding_model()
                                    texts = [doc.page_content for doc in langchain_docs]
                                    metadatas = [doc.metadata for doc in langchain_docs]
                                    
                                    # Generate embeddings in batches to manage memory efficiently
                                    embeddings = []
                                    batch_size = 10  # Adjust based on memory requirements
                                    
                                    index_status.update(label=f"Generating embeddings for {len(texts)} chunks...", state="running")
                                    
                                    for i in range(0, len(texts), batch_size):
                                        batch_texts = texts[i:i+batch_size]
                                        try:
                                            # Use embedding model to get vectors
                                            batch_embeddings = embedding_model.embed_documents(batch_texts)
                                            embeddings.extend(batch_embeddings)
                                            logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                                        except Exception as emb_error:
                                            logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {emb_error}")
                                            raise ValueError(f"Embedding error: {emb_error}")
                                    
                                    # 2. Prepare points for Qdrant
                                    points = []
                                    ids = []
                                    
                                    index_status.update(label=f"Preparing points for {len(embeddings)} embeddings...", state="running")
                                    
                                    for i, (embedding, metadata, text) in enumerate(zip(embeddings, metadatas, texts)):
                                        point_id = str(uuid.uuid4())  # Generate a unique ID
                                        ids.append(point_id)
                                        # Include the document content in the payload as "page_content"
                                        payload = {
                                            **metadata,
                                            "page_content": text  # Add the actual document text
                                        }
                                        points.append({
                                            "id": point_id,
                                            "vector": embedding,
                                            "payload": payload
                                        })
                                    
                                    # 3. Send points to Qdrant
                                    index_status.update(label=f"Sending {len(points)} points to Qdrant...", state="running")
                                    
                                    try:
                                        # Use Qdrant client's upsert method directly
                                        vector_store.client.upsert(
                                            collection_name=vector_store.collection_name,
                                            points=points
                                        )
                                        logger.info(f"Successfully added {len(points)} points to Qdrant for {file_display_name}")
                                    except Exception as upsert_error:
                                        logger.error(f"Error inserting into Qdrant: {upsert_error}")
                                        raise ValueError(f"Qdrant insertion error: {upsert_error}")

                                    logger.info(f"Successfully indexed documents from {file_display_name}.")
                                    index_status.update(label=f"Indexing complete for {file_display_name}", state="complete")

                                    # 5. Update session state for processed files
                                    doc_unique_id = str(uuid.uuid4()) # Simple unique ID for UI list
                                    st.session_state.uploaded_files_info.append({
                                        "id": doc_unique_id,
                                        "name": file_display_name,
                                        "status": "processed",
                                        "chunks": len(langchain_docs)
                                    })
                                    processed_count += 1
                                    total_chunks_processed += len(langchain_docs)
                                    processed_file_names.append(file_display_name)

                                except Exception as index_error:
                                    st.error(f"Failed to index {file_display_name}: {str(index_error)}")
                                    logger.error(f"Error indexing {file_display_name}: {index_error}", exc_info=True)
                                    error_count += 1
                                    index_status.update(label=f"Indexing failed for {file_display_name}", state="error")
                                    # Continue to the next file

                        except Exception as processing_error:
                            st.error(f"General error processing {file_display_name}: {str(processing_error)}")
                            logger.error(f"Error processing file {file_display_name}: {processing_error}", exc_info=True)
                            error_count += 1
                        finally:
                            # Clean up temporary file
                            if tmp_file_path and os.path.exists(tmp_file_path):
                                try:
                                    os.remove(tmp_file_path)
                                    logger.debug(f"Removed temporary file: {tmp_file_path}")
                                except Exception as cleanup_error:
                                    logger.warning(f"Failed to remove temporary file {tmp_file_path}: {cleanup_error}")

                    # Processing loop finished, update overall status
                    st.session_state.processing_status = "Complete" # Or "Error" if only errors? Let's use Complete

                    # Display summary message
                    if processed_count > 0:
                        st.success(f"Successfully processed {processed_count} file(s): {', '.join(processed_file_names)}. Total chunks indexed: {total_chunks_processed}")
                    if error_count > 0:
                        st.warning(f"Failed to process {error_count} file(s). Check logs for details.")
                    if processed_count == 0 and error_count == 0:
                         st.info("No new files were processed.") # Should not happen if button was clicked with files

                    # Au lieu de réinitialiser directement file_uploader, ajoutons un flag
                    # pour indiquer que le traitement est terminé
                    st.session_state["processing_complete"] = True
                    st.rerun() # Rerun to reflect the final status and updated UI

        # --- Display Indexed Documents ---
        st.subheader("Indexed Documents")
        # Filter out duplicates based on name if files are re-processed (simple approach)
        displayed_docs = {}
        for doc_info in reversed(st.session_state.uploaded_files_info): # Show newest first
            if doc_info['name'] not in displayed_docs:
                 displayed_docs[doc_info['name']] = doc_info

        if not displayed_docs:
             st.caption("No documents processed yet.")
        else:
             for doc_info in displayed_docs.values():
                 status_icon = "✅" if doc_info.get('status') == "processed" else "⚠️"
                 chunks_info = f" ({doc_info.get('chunks', 'N/A')} chunks)" if 'chunks' in doc_info else ""
                 st.write(f"{status_icon} {doc_info.get('name', 'Unknown')}{chunks_info}")

        # --- Database Vector Count ---
        vector_store_instance = get_component('vector_store')
        if vector_store_instance:
            try:
                # Check if collection exists before counting
                collection_name = vector_store_instance.collection_name
                if vector_store_instance.client.collection_exists(collection_name=collection_name):
                     count_result = vector_store_instance.client.count(collection_name=collection_name, exact=True) # Use exact count
                     st.metric("Total vectors in database", count_result.count)
                else:
                     st.caption(f"Collection '{collection_name}' does not exist.")
                     st.metric("Total vectors in database", 0)
            except Exception as e:
                st.error(f"Could not get vector count: {str(e)}")
                logger.warning(f"Failed to get vector count: {e}", exc_info=True)

        # --- Clear Documents Button ---
        if st.button("⚠️ Clear All Documents", key="clear_docs"):
            st.warning("This will delete all indexed data. Are you sure?")
            if st.button("Yes, Clear Everything"):
                vector_store = get_component('vector_store')
                if vector_store:
                    try:
                        with st.spinner("Clearing vector store..."):
                            vector_store.clear_collection() # Deletes and recreates
                        st.success("Vector store collection cleared and reset successfully.")
                        logger.info("Vector store collection cleared.")
                        # Reset relevant session state
                        st.session_state.uploaded_files_info = []
                        st.session_state.messages = []
                        st.session_state.response_text = ""
                        st.session_state.relevance_info = []
                        st.session_state.processing_status = "Ready"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing vector store: {str(e)}")
                        logger.error(f"Error clearing vector store: {e}", exc_info=True)
                else:
                    st.error("Vector store component not available.")

# --- Column 2: Query Interface & Response ---
with col2:
    st.header("💬 Query & Response")
    with st.container(border=True, height=600): # Add height and maybe make scrollable
        # Display chat history
        st.write("**Chat History:**")
        if not st.session_state.messages:
             st.caption("Ask a question about the uploaded documents.")
        else:
             for msg in st.session_state.messages:
                 with st.chat_message(msg["role"]):
                     st.markdown(msg["content"])
                     if msg["role"] == "assistant" and "processing_time" in msg:
                         st.caption(f"Time: {msg['processing_time']:.2f}s")


    # Query Input - Place outside the history container
    prompt = st.chat_input("Enter your query here...")

    if prompt:
        logger.info(f"Received query: {prompt[:100]}...")
        # Display user message immediately
        st.chat_message("user").markdown(prompt)
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get RAG Pipeline (initializes if needed)
        rag_pipeline = get_component('rag_pipeline')

        if not rag_pipeline:
            st.error("RAG Pipeline is not available. Cannot process query.")
        else:
            # Check collection status before querying
            collection_check = rag_pipeline.check_collection()
            if collection_check['status'] == 'error':
                st.error(f"Qdrant Collection Error: {collection_check['message']}")
            elif collection_check['point_count'] == 0:
                st.warning("The document collection is empty. Please upload and process documents before querying.")
            else:
                # Proceed with query
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    try:
                        # Execute the RAG query
                        result = rag_pipeline.query(prompt)
                        response = result.get("response", "Sorry, I couldn't generate a response based on the documents.")
                        sources = result.get("sources", []) # These are LangChain Document objects

                        # Process sources for relevance display
                        current_relevance_info = []
                        if sources:
                             logger.info(f"Retrieved {len(sources)} sources for query.")
                             for i, source_doc in enumerate(sources):
                                 # Access metadata safely using .get()
                                 metadata = source_doc.metadata if hasattr(source_doc, 'metadata') else {}
                                 source_file = metadata.get('source', 'Unknown Source')
                                 # Attempt to get score from Qdrant results (might not be in metadata directly)
                                 # Score often needs specific retriever config or post-processing
                                 score = metadata.get('score', None) # Placeholder, score might not be here
                                 
                                 # Make sure we have actual content in the page_content
                                 content_preview = ""
                                 if hasattr(source_doc, 'page_content') and source_doc.page_content:
                                     content_preview = source_doc.page_content[:150] + "..." if len(source_doc.page_content) > 150 else source_doc.page_content
                                 elif hasattr(source_doc, 'metadata') and source_doc.metadata and 'page_content' in source_doc.metadata:
                                     content = source_doc.metadata['page_content']
                                     content_preview = content[:150] + "..." if len(content) > 150 else content
                                 else:
                                     content_preview = "No content available"

                                 relevance_item = {
                                     'source': source_file,
                                     # 'score': score, # Add score only if available and reliable
                                     'chunk_index': metadata.get('chunk_index', i), # Use chunk_index from processing if available
                                     'content_preview': content_preview
                                 }
                                 if 'page' in metadata:
                                     relevance_item['page'] = metadata['page']
                                 # Add score if we found it
                                 if score is not None:
                                     relevance_item['score'] = f"{score:.2f}"

                                 current_relevance_info.append(relevance_item)
                        else:
                             logger.warning("No sources retrieved for the query.")


                        st.session_state.relevance_info = current_relevance_info # Update relevance
                        st.session_state.response_text = response # Update response text

                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        logger.error(f"Error during RAG query execution: {e}", exc_info=True)
                        st.session_state.response_text = "An error occurred while processing your query."
                        st.session_state.relevance_info = []

                    end_time = time.time()
                    processing_time = end_time - start_time
                    logger.info(f"Query processing time: {processing_time:.2f} seconds")

                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": st.session_state.response_text,
                        "processing_time": processing_time
                    })

                # Rerun to display the new assistant message and relevance info
                st.rerun()

# --- Column 3: Relevance Dashboard ---
with col3:
    st.header("🎯 Relevance")
    with st.container(border=True, height=600): # Match height for alignment
        st.subheader("Retrieved Context")
        if not st.session_state.relevance_info:
            st.caption("Contextual information will appear after a query.")
        else:
            # Display relevance info from the latest query
            for i, item in enumerate(st.session_state.relevance_info):
                with st.expander(f"Source {i+1}: {item.get('source', 'N/A')}", expanded= i < 2): # Expand first 2
                    # Display score if available
                    if 'score' in item:
                        st.markdown(f"**Score:** {item['score']}")
                    # Display page if available
                    if 'page' in item:
                        st.markdown(f"**Page:** {item['page']}")
                    # Display chunk index if available
                    if 'chunk_index' in item:
                         st.markdown(f"**Chunk Index:** {item['chunk_index']}")
                    # Show content preview
                    st.caption("Content Preview:")
                    if item.get('content_preview') and item.get('content_preview') != "N/A" and item.get('content_preview') != "...":
                        st.markdown(f"> _{item.get('content_preview')}_")
                    else:
                        st.warning("No content available for this source.")

# --- Main execution check (optional for Streamlit apps) ---
if __name__ == "__main__":
    logger.info("DocAsis Streamlit application started.")
    # No explicit main function call needed, Streamlit handles the execution loop.