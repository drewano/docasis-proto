"""
RAG Model Implementation

This module provides the implementation of the Retrieval Augmented Generation (RAG) model
that orchestrates the retrieval and generation process using LangChain.
"""

import logging
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from qdrant_client import QdrantClient

# Project imports
from src.utils.config import Config
from src.data.vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG Pipeline for retrieving and generating responses using document context.
    
    This class orchestrates the retrieval and generation process by:
    1. Processing queries
    2. Retrieving relevant document chunks
    3. Using retrieved context with an LLM to generate responses
    4. Tracking which document chunks were used in responses
    """
    
    def __init__(self, config: Config, vector_store: VectorStore):
        """
        Initialize the RAG Pipeline.
        
        Args:
            config: Configuration object.
            vector_store: Initialized VectorStore object.
        """
        self.config = config
        self.vector_store = vector_store
        self.qdrant_client = vector_store.client
        self.collection_name = vector_store.collection_name
        
        logger.info("Initializing RAG Pipeline components...")
        
        try:
            # Initialize LLM (Ensure GOOGLE_API_KEY is configured)
            llm_model = "gemini-2.0-flash"
            logger.info(f"Using LLM model: {llm_model}")
            self.llm = ChatGoogleGenerativeAI(model=llm_model)
            
            # Initialize Embeddings Model for LangChain Qdrant integration
            self.embeddings = GoogleGenerativeAIEmbeddings(model=config.get("EMBEDDING_MODEL", "models/embedding-001"))

            # Initialize LangChain Qdrant Vector Store wrapper
            self.qdrant_langchain = QdrantVectorStore(
                client=self.qdrant_client, 
                collection_name=self.collection_name, 
                embedding=self.embeddings
            )

            # Initialize Retriever
            self.retriever = self.qdrant_langchain.as_retriever(
                search_kwargs={'k': config.get('TOP_K_RESULTS', 5)}
            )

            # Define Prompt Template
            template = """Answer the question based only on the following context:
{context}

Question: {question}"""
            self.prompt = ChatPromptTemplate.from_template(template)

            # Define RAG Chain using LCEL
            self.rag_chain = (
                {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
                | self.prompt 
                | self.llm 
                | StrOutputParser()
            )

            logger.info("RAG Pipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline components: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize RAG Pipeline") from e
        
    @staticmethod
    def _format_docs(docs: List[Any]) -> str:
        """Helper function to format retrieved documents into a string context."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            try:
                # First try to access the page_content attribute directly
                if hasattr(doc, 'page_content') and doc.page_content:
                    formatted_docs.append(doc.page_content)
                    logger.debug(f"Added document {i} to context (from page_content): {doc.page_content[:50]}...")
                # If that fails, check if it's in the metadata (from Qdrant payload)
                elif hasattr(doc, 'metadata') and doc.metadata and 'page_content' in doc.metadata:
                    formatted_docs.append(doc.metadata['page_content'])
                    logger.debug(f"Added document {i} to context (from metadata): {doc.metadata['page_content'][:50]}...")
                else:
                    logger.warning(f"Document {i} has no content in page_content or metadata")
            except (AttributeError, TypeError) as e:
                logger.warning(f"Problem with document {i}: {type(doc)}, error: {e}")
        
        if not formatted_docs:
            logger.warning("No documents could be formatted for context")
            return "No relevant content found."
            
        return "\n\n".join(formatted_docs)

    def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline and retrieve source documents.
        
        Args:
            query_text: The text of the query to process
            **kwargs: Additional parameters (currently unused but kept for future flexibility)
            
        Returns:
            Dict containing:
                - response: Generated response text
                - sources: List of retrieved source Document objects
        """
        logger.info(f"Processing query: {query_text[:50]}...")
        
        if not self.retriever:
             logger.error("Retriever is not initialized.")
             return {"response": "Error: RAG Pipeline not properly initialized.", "sources": []}

        try:
            # Retrieve documents
            retrieved_docs = self.retriever.invoke(query_text)
            num_docs = len(retrieved_docs) if retrieved_docs else 0
            logger.info(f"Retrieved {num_docs} documents for query.")
            
            # Log the first few characters of each document for debugging
            for i, doc in enumerate(retrieved_docs[:3] if retrieved_docs else []):
                # Try to get content from either page_content or metadata
                content = ""
                if hasattr(doc, 'page_content') and doc.page_content:
                    content = doc.page_content[:100]
                elif hasattr(doc, 'metadata') and doc.metadata and 'page_content' in doc.metadata:
                    content = doc.metadata['page_content'][:100]
                else:
                    content = "No content found"
                logger.debug(f"Retrieved doc {i}: {content}...")
            
            if num_docs == 0:
                logger.warning("No documents retrieved from the vector store. Check if documents are properly stored.")
                return {
                    "response": "Je n'ai pas trouvé d'informations pertinentes dans les documents disponibles pour répondre à cette question.",
                    "sources": []
                }

            # Generate response
            try:
                formatted_context = self._format_docs(retrieved_docs)
                
                # Utiliser correctement la chaîne LangChain
                prompt_value = self.prompt.invoke({
                    "context": formatted_context, 
                    "question": query_text
                })
                llm_response = self.llm.invoke(prompt_value)
                response_text = StrOutputParser().invoke(llm_response)

                logger.info(f"Generated response for query: {query_text[:50]}...")
                
                return {
                    "response": response_text,
                    "sources": retrieved_docs
                }
            except Exception as e:
                logger.error(f"Error generating response: {e}", exc_info=True)
                return {
                    "response": f"Erreur lors de la génération de la réponse: {e}",
                    "sources": retrieved_docs
                }
        except Exception as e:
            logger.error(f"Error during RAG query execution: {e}", exc_info=True)
            return {
                "response": f"Error processing query: {e}",
                "sources": []
            }

    def check_collection(self) -> Dict[str, Any]:
        """
        Vérifie l'état de la collection Qdrant et fournit des informations de diagnostic.
        
        Returns:
            Dict contenant des informations sur la collection
        """
        try:
            # Vérifier si la collection existe
            collection_exists = self.qdrant_client.collection_exists(collection_name=self.collection_name)
            if not collection_exists:
                return {
                    "status": "error",
                    "message": f"La collection '{self.collection_name}' n'existe pas dans Qdrant."
                }
                
            # Obtenir les informations sur la collection
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            
            # Compter les points
            count_result = self.qdrant_client.count(collection_name=self.collection_name)
            
            # Effectuer une recherche simple
            sample_vector = self.embeddings.embed_query("test query")
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=sample_vector,
                limit=1
            )
            
            return {
                "status": "success",
                "collection_info": collection_info.dict(),
                "point_count": count_result.count,
                "has_search_results": len(search_result) > 0
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de la collection: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Erreur lors de la vérification de la collection: {e}"
            }

# Example usage (requires environment variables and potentially a running Qdrant instance)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv() # Load .env file for API keys, Qdrant URL etc.
    
    # Basic logging setup for testing
    logging.basicConfig(level=logging.INFO)
    
    # Check required environment variables
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("QDRANT_URL"):
        print("Error: GOOGLE_API_KEY and QDRANT_URL must be set in the environment or .env file.")
    else:
        try:
            # Initialize configuration and vector store
            app_config = Config() # Initialize Config singleton
            vector_db = VectorStore(collection_name="test_rag_collection") # Use a specific test collection
            
            # Ensure test collection exists and has some data (manual setup needed for a full test)
            # For a basic test, we assume the collection exists. 
            # A more robust test would upload sample data first.
            print(f"Attempting to initialize RAGPipeline with collection: {vector_db.collection_name}")
            
            # Initialize RAG Pipeline
            rag_pipeline = RAGPipeline(config=app_config, vector_store=vector_db)
            print("RAG Pipeline Initialized.")

            # Example Query
            test_query = "What is the capital of France?" 
            print(f"\nPerforming test query: '{test_query}'")
            
            # It's unlikely the test collection has relevant info, so this tests the flow
            result = rag_pipeline.query(test_query)
            
            print(f"\nResponse:")
            print(result.get("response"))
            print(f"\nSources Retrieved ({len(result.get('sources', []))}):")
            if result.get("sources"):
                for i, doc in enumerate(result["sources"]):
                     print(f"  Source {i+1}:")
                     print(f"    Content: {doc.page_content[:100]}...") # Show snippet
                     print(f"    Metadata: {doc.metadata}")
            else:
                print("  No sources retrieved (or an error occurred).")

        except RuntimeError as e:
             print(f"Runtime Error during initialization or query: {e}")
        except Exception as e:
             print(f"An unexpected error occurred: {e}")
             import traceback
             traceback.print_exc()

# Remove placeholder methods from the original file if they existed
# (The edit replaces the file content, so placeholders are implicitly removed)
# Placeholder initialize method removed
# Placeholder query method removed
# Placeholder get_relevant_documents method removed 