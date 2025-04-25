"""
Streamlit UI Application

This module implements the Streamlit UI for the RAG Intelligent Agent with the
three-panel layout specified in the requirements.
"""

import logging
import os
import sys
from pathlib import Path
import tempfile
import uuid
import time

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from src.utils.config import get_config, Config
from src.models.rag_model import RAGPipeline
from src.data.vector_store import VectorStore
from src.data.document_processor import DocumentProcessor
from src.api.model_interface import LLMInterface
from src.models.embedding_processor import EmbeddingProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client import models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Intelligent Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state if needed
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "documents" not in st.session_state:
    st.session_state.documents = []
    
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "embedding_processor" not in st.session_state:
    st.session_state.embedding_processor = None

# Move component initialization into session state check to run only once
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    try:
        st.session_state.config = Config()
        st.session_state.vector_store = VectorStore(
            collection_name=st.session_state.config.get('QDRANT_COLLECTION', 'rag_documents')
        )
        st.session_state.embedding_processor = EmbeddingProcessor(
            model_name=st.session_state.config.get("EMBEDDING_MODEL", "models/text-embedding-004"),
            batch_size=st.session_state.config.get("EMBEDDING_BATCH_SIZE", 100)
        )
        st.session_state.rag_pipeline = RAGPipeline(
            config=st.session_state.config,
            vector_store=st.session_state.vector_store
        )
        st.session_state.initialized = True
        logger.info("Application components initialized successfully.")
    except Exception as e:
        logger.exception(f"Fatal error during initial application setup: {e}")
        st.error(f"Application failed to initialize: {e}. Please check configuration and logs.")
        st.stop()

def main():
    """Main Streamlit application."""
    # Display header
    st.title("RAG Intelligent Agent")
    st.markdown("Enhance LLM responses with relevant document context")
    
    # Check initialization status
    if not st.session_state.get("initialized", False):
        st.error("Application is not initialized. Please check logs.")
        st.stop()

    # Retrieve components from session state reliably
    config = st.session_state.config
    vector_store = st.session_state.vector_store
    embedder = st.session_state.embedding_processor
    rag_pipeline = st.session_state.rag_pipeline
    doc_processor = DocumentProcessor(config=config)
    
    # Three-panel layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Left panel - Document upload and management
    with col1:
        st.header("Document Management")
        st.markdown("Upload and manage documents for retrieval")
        
        # Get supported formats from processor if available, else default
        supported_formats = ["." + fmt for fmt in doc_processor.get_supported_formats()] if hasattr(doc_processor, 'get_supported_formats') else [".pdf", ".docx", ".txt"]
        
        # File uploader
        uploaded_files = st.file_uploader("Upload Documents", 
                                        type=[fmt.lstrip('.') for fmt in supported_formats], 
                                        accept_multiple_files=True,
                                        help=f"Supported types: {', '.join(supported_formats)}")
        
        if uploaded_files:
            # Replace placeholder
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                 processed_count = 0
                 error_count = 0
                 for uploaded_file in uploaded_files:
                     try:
                         # Save uploaded file to a temporary location
                         with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                             tmp_file.write(uploaded_file.getvalue())
                             tmp_file_path = tmp_file.name
                         
                         logger.info(f"Processing temporary file: {tmp_file_path} for {uploaded_file.name}")
                         
                         # 1. Process document into chunks
                         chunks = doc_processor.process_document(tmp_file_path)
                         
                         if not chunks:
                             logger.warning(f"No chunks generated for {uploaded_file.name}")
                             st.warning(f"Could not extract text/chunks from {uploaded_file.name}. Skipping.")
                             error_count += 1
                             continue # Skip to next file
                         
                         # 2. Embed chunks
                         docs_to_embed = [{**chunk['metadata'], 'text': chunk['content']} for chunk in chunks]
                         embedded_docs = embedder.embed_documents(docs_to_embed)
                         
                         # 3. Prepare points for Qdrant
                         points_to_upload = []
                         ids_to_upload = []
                         failed_embeddings = 0
                         for i, doc in enumerate(embedded_docs):
                             if doc.get('embedding'):
                                 # Use content and metadata from the original chunk dictionary
                                 original_chunk = chunks[i]
                                 # Generate a unique ID for each chunk vector
                                 point_id = str(uuid.uuid4())
                                 # Ensure metadata is serializable (e.g., convert Path objects to str)
                                 serializable_metadata = {k: str(v) if isinstance(v, Path) else v for k, v in original_chunk.get('metadata', {}).items()}
                                 
                                 # Override source information with the actual uploaded filename
                                 # This prevents issues with temporary paths showing as source
                                 serializable_metadata['source'] = uploaded_file.name
                                 serializable_metadata['source_document'] = uploaded_file.name
                                 serializable_metadata['original_name'] = uploaded_file.name
                                 
                                 # Preserve the original logic for backwards compatibility
                                 source_path = serializable_metadata.get('source', '')
                                 if source_path:
                                     # Store both full path and just filename
                                     source_filename = Path(source_path).name
                                     serializable_metadata['source_path'] = source_path
                                 
                                 # Optionally store chunk text itself if needed for display without re-retrieval
                                 # serializable_metadata['chunk_text'] = original_chunk.get('content', '')

                                 ids_to_upload.append(point_id)
                                 points_to_upload.append(
                                     models.PointStruct(
                                         id=point_id,
                                         vector=doc['embedding'],
                                         payload=serializable_metadata # Store processed metadata
                                     )
                                 )
                             else:
                                 logger.warning(f"Skipping chunk {i} from {uploaded_file.name} due to missing embedding.")
                                 failed_embeddings += 1
                                 
                         # 4. Upload vectors to Qdrant
                         if points_to_upload:
                              vector_store.upload_vectors(points=points_to_upload, ids=ids_to_upload)
                              logger.info(f"Uploaded {len(points_to_upload)} vectors for {uploaded_file.name}")
                              # Add doc to session state only if successful
                              # Add document ID if useful for targeted deletion later
                              doc_unique_id = str(uuid.uuid4())
                              st.session_state.documents.append({
                                  "id": doc_unique_id, # Add a unique ID for the document itself
                                  "name": uploaded_file.name,
                                  "status": "processed",
                                  "vector_ids": ids_to_upload # Store vector IDs associated with this doc
                              })
                         else:
                              logger.warning(f"No valid vectors generated for {uploaded_file.name}. Nothing uploaded.")
                              if failed_embeddings == len(embedded_docs):
                                   st.error(f"Failed to generate embeddings for all chunks in {uploaded_file.name}. Check logs.")
                                   error_count += 1
                                   continue # Skip to next file if all embeddings failed
                                   
                         # 5. Update session state (simple list of names for now)
                         processed_count += 1
                         
                     except (FileNotFoundError, ValueError, RuntimeError, ConnectionError, Exception) as e:
                          logger.exception(f"Error processing file {uploaded_file.name}: {e}")
                          st.error(f"Error processing {uploaded_file.name}: {e}")
                          error_count += 1
                     finally:
                         # Clean up temporary file
                         if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                             os.remove(tmp_file_path)
                             logger.info(f"Removed temporary file: {tmp_file_path}")
                             del tmp_file_path # Ensure variable is cleared
            
            # Display summary message after processing all files
            if processed_count > 0:
                 st.success(f"Successfully processed {processed_count} file(s).")
            if error_count > 0:
                 st.warning(f"Failed to process {error_count} file(s). Check errors above and logs.")
                 
            # Rerun to clear the uploader state and update the document list
            st.rerun() 

        # Document list display (update structure)
        st.subheader("Indexed Documents")
        if not st.session_state.documents:
            st.write("No documents indexed yet.")
        else:
            # Use columns for better layout if needed, add delete button per doc
            for i, doc_info in enumerate(st.session_state.documents):
                doc_col, del_col = st.columns([4, 1])
                with doc_col:
                    st.markdown(f"- {doc_info.get('name', 'Unknown Document')} (`{len(doc_info.get('vector_ids',[]))} vectors`)")
                with del_col:
                    if st.button(f"❌", key=f"del_{doc_info.get('id', i)}", help=f"Delete {doc_info.get('name')}"):
                        with st.spinner(f"Deleting {doc_info.get('name')}..."):
                            try:
                                ids_to_delete = doc_info.get('vector_ids', [])
                                if ids_to_delete:
                                    vector_store.delete_vectors(ids=ids_to_delete)
                                    st.session_state.documents.pop(i) # Remove from list
                                    st.success(f"Deleted {doc_info.get('name')}.")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.warning("No associated vectors found to delete.")
                            except Exception as e:
                                logger.exception(f"Error deleting document {doc_info.get('name')}: {e}")
                                st.error(f"Failed to delete {doc_info.get('name')}: {e}")

        # Clear all documents button logic (adjust confirmation)
        if st.button("🗑️ Clear All Documents"):
            if not st.session_state.documents:
                 st.info("No documents to clear.")
            else:
                 # Add a confirmation step
                 confirm_clear = st.checkbox("Confirm clearing all documents and vectors?", key="confirm_clear_all")
                 if confirm_clear:
                     with st.spinner("Clearing all documents and vectors..."):
                         try:
                             vector_store.clear_collection() # Assuming this clears all vectors
                             st.session_state.documents = [] # Clear the list in session state
                             st.session_state.messages = [] # Optionally clear chat history too
                             st.success("All documents and vectors cleared successfully.")
                             time.sleep(1) # Brief pause before rerun
                             st.rerun()
                         except Exception as e:
                             logger.exception("Error clearing all documents:")
                             st.error(f"Error clearing documents: {e}")
                 else:
                     st.warning("Checkbox not confirmed. Documents not cleared.")
    
    # Center panel - Query interface and response display (Revised)
    with col2:
        st.header("Query Interface")
        st.markdown("Ask questions about your documents")
        
        # Create a container for the chat interface with fixed height
        chat_container = st.container()
        
        # Add user input at the top of the container
        user_query = st.chat_input("Ask your question:")
        
        # Create a scrollable area for messages
        with chat_container:
            # Display chat messages from history in a scrollable area
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Optionally display time or sources here if stored
                    if "time" in message:
                        st.caption(f"Response time: {message['time']:.2f}s")
                    # Source display deferred to col3 (Task 9)

        # Process user input (placed after displaying the history)
        if user_query:
            # Display user message in chat message container
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Display assistant response placeholder
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    start_time = time.time()
                    try:
                        # Ensure pipeline is initialized
                        if rag_pipeline:
                            logger.info(f"Sending query to RAG pipeline: {user_query}")
                            # Assuming rag_pipeline.query returns a dict like {"response": str, "sources": list}
                            result = rag_pipeline.query(user_query)
                            response = result.get("response", "Sorry, I couldn't generate a response.")
                            sources = result.get("sources", []) # Get sources for Task 9
                            end_time = time.time()
                            duration = end_time - start_time
                            logger.info(f"RAG pipeline responded in {duration:.2f} seconds.")

                            # Display the actual response
                            message_placeholder.markdown(response)

                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "time": duration,
                                "sources": sources # Store sources for Task 9
                            })

                        else:
                            logger.error("RAG Pipeline not initialized.")
                            message_placeholder.error("Error: RAG Pipeline is not available.")
                            st.session_state.messages.append({"role": "assistant", "content": "Error: RAG Pipeline not available."})

                    except Exception as e:
                        logger.exception(f"Error during RAG query: {e}")
                        end_time = time.time()
                        duration = end_time - start_time
                        error_message = f"An error occurred: {e}"
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "time": duration
                        })
            
            # Auto-scroll to bottom by forcing a rerun
            st.rerun()
    
    # Right panel - Relevance Dashboard
    with col3:
        st.header("Relevance Dashboard")
        st.markdown("Document citations and relevance metrics")

        last_assistant_message = None
        if st.session_state.messages:
            # Find the last message from the assistant
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    last_assistant_message = msg
                    break

        if last_assistant_message and "sources" in last_assistant_message and last_assistant_message["sources"]:
            st.subheader("Retrieved Context")
            sources = last_assistant_message["sources"]
            
            # Assuming sources is a list of LangChain Document objects or similar dicts
            for i, source in enumerate(sources):
                try:
                    # Adapt based on actual structure of source object/dict
                    metadata = source.metadata if hasattr(source, 'metadata') else source.get("metadata", {})
                    # Look for score in metadata (where we store it from our RAG pipeline)
                    score = metadata.get("score", None)
                    content = source.page_content if hasattr(source, 'page_content') else source.get("page_content", "No content available")

                    # Get source document name with better fallback options
                    source_doc_name = metadata.get('source_document', '')
                    if not source_doc_name:
                        source_doc_name = metadata.get('file_name', '')
                    if not source_doc_name:
                        source_doc_name = metadata.get('source', 'Unknown Source')
                    
                    # Get page number with better fallback
                    page_num = metadata.get('page', metadata.get('chunk_index', 'N/A'))

                    # Format the source display more clearly
                    source_display = f"**Source {i+1}: {source_doc_name}**"
                    if page_num != 'N/A':
                        source_display += f" (Page: {page_num})"

                    st.markdown(source_display)
                    
                    # Display score if available
                    if score is not None:
                        # Format score as percentage for better user understanding
                        try:
                            normalized_score = min(1.0, float(score))
                            # For cosine similarity, higher is better (closer to 1)
                            score_percent = int(normalized_score * 100)
                            
                            # Display both as metric and visual indicator
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Relevance", f"{score_percent}%")
                            with col2:
                                # Add visual indicator
                                st.progress(normalized_score)
                        except (ValueError, TypeError):
                            # Fallback if score cannot be converted
                            st.metric("Relevance Score", f"{score}")

                    # Expander for content
                    with st.expander(f"Show context snippet #{i+1}"):
                        st.text(content) # Use st.text for preformatted/raw text display
                        
                    st.divider()
                    
                except AttributeError as ae:
                     logger.warning(f"Could not parse source #{i+1} due to unexpected structure: {ae}. Source data: {source}")
                     st.warning(f"Could not display source #{i+1} properly.")
                except Exception as e:
                     logger.exception(f"Error displaying source #{i+1}: {e}")
                     st.error(f"Error displaying source #{i+1}.")

        elif last_assistant_message and ("sources" not in last_assistant_message or not last_assistant_message["sources"]):
             st.write("No specific source documents were retrieved for the last response.")
        else:
            st.write("Submit a query to see relevance information.")

    # Footer
    st.markdown("---")
    st.markdown("RAG Intelligent Agent - Prototype Version")

if __name__ == "__main__":
    main() 