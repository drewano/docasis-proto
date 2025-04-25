import logging
import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Union, Dict, Any, Tuple, Optional
import time
import uuid

# Import the config module
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages interactions with Qdrant Cloud for vector storage and retrieval.
    
    Attributes:
        client: The Qdrant client used for vector operations
        collection_name: The name of the Qdrant collection to use
        vector_size: The dimensionality of the vectors (default: 768)
        distance: The distance metric for similarity search (default: Cosine)
    """
    def __init__(self, collection_name: str = "rag_documents", url: Optional[str] = None, vector_size: int = 768, distance: Distance = Distance.COSINE):
        """
        Initializes the Qdrant client and ensures the collection exists.

        Args:
            collection_name (str): The name of the collection to interact with.
                                   Defaults to "rag_documents".
            url (Optional[str]): The URL of the Qdrant server. If None, uses the config.
            vector_size (int, optional): The dimensionality of the vectors. Defaults to 768.
            distance (Distance, optional): The distance metric to use. Defaults to Distance.COSINE.
        """
        # Set attributes first to ensure they're available for all method calls
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        # Get the URL from config if not provided
        if url is None:
            # Default URL to use if config doesn't provide one
            default_url = "http://localhost:6333"
            
            # Try to get URL from config
            config_url = Config().get_qdrant_config().get("QDRANT_URL")
            
            # Use config URL if available, otherwise use default
            if config_url is not None:
                url = str(config_url)
            else:
                url = default_url
        
        logger.info(f"Initializing VectorStore with collection '{collection_name}' at {url}")
        
        try:
            # Augmenter les timeouts pour plus de robustesse
            if Config().get_qdrant_config().get("QDRANT_API_KEY"):
                 self.client = QdrantClient(
                    url=url, 
                    api_key=Config().get_qdrant_config().get("QDRANT_API_KEY"),
                    timeout=120,  # Augmenté à 120 secondes
                    prefer_grpc=False  # Désactiver gRPC pour plus de compatibilité
                 )
            else:
                 self.client = QdrantClient(
                    url=url,
                    timeout=120,  # Augmenté à 120 secondes
                    prefer_grpc=False  # Désactiver gRPC pour plus de compatibilité
                 )
            
            logger.info(f"Successfully connected to Qdrant at {url}")
            
            # Vérifier la connexion avec une opération simple
            try:
                collections = self.client.get_collections()
                logger.info(f"Qdrant connection verified. Available collections: {[c.name for c in collections.collections]}")
            except Exception as conn_error:
                logger.warning(f"Connected to Qdrant but couldn't retrieve collections: {conn_error}")
            
            # Ensure the collection exists
            self.ensure_collection_exists()

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {url}: {e}", exc_info=True)
            # Depending on requirements, might re-raise or handle differently
            raise ConnectionError(f"Could not connect to Qdrant: {e}") from e

    # --- Placeholder methods based on the plan ---

    def create_collection(self):
        """
        Create a collection in Qdrant with the specified configuration.
        """
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}' in Qdrant")
        except Exception as e:
            logger.error(f"Error creating collection '{self.collection_name}': {e}")
            # Check if it's just because the collection already exists
            if "already exists" in str(e).lower():
                logger.info(f"Collection '{self.collection_name}' already exists, continuing...")
                return True
            raise ValueError(f"Failed to create collection: {e}")
        return True

    def ensure_collection_exists(self) -> bool:
        """
        Check if the collection exists, create it if it does not.
        
        Returns:
            bool: True if the collection exists or was created successfully, False otherwise.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name in collection_names:
                logger.debug(f"Collection '{self.collection_name}' already exists")
                return True
            
            logger.info(f"Collection '{self.collection_name}' does not exist, creating...")
            return self.create_collection()
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise ValueError(f"Failed to interact with Qdrant: {e}")

    def upload_vectors(self, vectors, metadatas, ids=None):
        """
        Upload vectors to the Qdrant collection.
        
        Args:
            vectors (List[List[float]]): List of vectors to upload
            metadatas (List[Dict]): List of metadata to attach to each vector
            ids (List[str], optional): List of IDs to use. If None, UUIDs will be generated.
        
        Returns:
            List[str]: List of IDs that were uploaded
        """
        if not self.ensure_collection_exists():
            raise ValueError(f"Collection '{self.collection_name}' does not exist and could not be created")
            
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError(f"Number of ids ({len(ids)}) doesn't match number of vectors ({len(vectors)})")
        
        points = []
        for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
            points.append({
                "id": ids[i],
                "vector": vector,
                "payload": metadata
            })
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully uploaded {len(vectors)} vectors to '{self.collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Error uploading vectors to '{self.collection_name}': {e}")
            raise ValueError(f"Failed to upload vectors: {e}")

    def search(self, query_vector: List[float], limit: int = 5) -> List[models.ScoredPoint]:
        """Performs similarity search against the collection.

        Args:
            query_vector (List[float]): The vector to search for.
            limit (int): The maximum number of results to return (default: 5).

        Returns:
            List[models.ScoredPoint]: A list of search results, including score and point data.
        """
        if not query_vector:
            logger.warning("Search called with an empty query vector.")
            return []

        logger.info(f"Performing search in '{self.collection_name}' with limit {limit}")
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,  # Toujours inclure les métadonnées
                timeout=60  # Augmenter le timeout pour la recherche
            )
            logger.info(f"Search found {len(search_result)} results.")
            return search_result

        except Exception as e:
            logger.error(f"Failed to perform search in '{self.collection_name}': {e}", exc_info=True)
            # Depending on needs, could return empty list or re-raise
            return []

    def delete_vectors(self, point_ids: List[Any]):
        """Deletes vectors by their IDs from the collection.

        Args:
            point_ids (List[Any]): A list of IDs for the points to be deleted.
        """
        if not point_ids:
            logger.info("No point IDs provided for deletion.")
            return

        logger.info(f"Attempting to delete {len(point_ids)} points from '{self.collection_name}'")
        try:
            response = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True # Wait for operation to complete
            )
            logger.info(f"Delete response status: {response.status}")
            # Optional: check response status
            if response.status != models.UpdateStatus.COMPLETED:
                logger.warning(f"Delete operation did not complete successfully: {response.status}")

        except Exception as e:
            logger.error(f"Failed to delete vectors from '{self.collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Could not delete vectors: {e}") from e

    def clear_collection(self):
        """
        Deletes all data in the collection by deleting and recreating it.
        WARNING: This is a destructive operation.
        """
        logger.warning(f"Attempting to clear all data from collection '{self.collection_name}' by recreating it.")
        try:
            # First check if collection exists
            if not self.client.collection_exists(collection_name=self.collection_name):
                logger.info(f"Collection '{self.collection_name}' does not exist, creating new collection.")
                return self.create_collection()
            
            # Delete the collection with increased timeout
            self.client.delete_collection(
                collection_name=self.collection_name, 
                timeout=180  # Increased timeout to 3 minutes for more reliable deletion
            )
            logger.info(f"Successfully deleted collection '{self.collection_name}'.")
            
            # Verify the collection was actually deleted
            attempt = 0
            max_attempts = 3
            while attempt < max_attempts:
                try:
                    if not self.client.collection_exists(collection_name=self.collection_name):
                        logger.info(f"Verified collection '{self.collection_name}' was deleted.")
                        break
                    else:
                        logger.warning(f"Collection '{self.collection_name}' still exists after deletion attempt {attempt + 1}.")
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                except Exception as check_e:
                    logger.warning(f"Error checking collection existence after deletion: {check_e}")
                    time.sleep(2)
                attempt += 1
            
            # Pause before recreating
            time.sleep(2) 
            
            # Recreate the collection with retry logic
            retry_attempts = 0
            max_retry = 3
            last_error = None
            
            while retry_attempts < max_retry:
                try:
                    self.create_collection()
                    logger.info(f"Successfully recreated collection '{self.collection_name}' on attempt {retry_attempts + 1}.")
                    return True
                except Exception as e:
                    last_error = e
                    logger.warning(f"Attempt {retry_attempts + 1} to recreate collection failed: {e}")
                    retry_attempts += 1
                    time.sleep(2 * retry_attempts)  # Exponential backoff
            
            # If we got here, all retries failed
            if last_error:
                logger.error(f"Failed to recreate collection after {max_retry} attempts: {last_error}")
                raise RuntimeError(f"Failed to recreate collection after {max_retry} attempts: {last_error}")
            
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection '{self.collection_name}': {e}", exc_info=True)
            # Attempt to recreate anyway in case deletion failed partially but is recoverable
            try:
                logger.warning(f"Attempting to recreate collection '{self.collection_name}' after clear error.")
                self.create_collection()
                return True
            except Exception as recreate_e:
                logger.error(f"Failed to recreate collection '{self.collection_name}' after clear error: {recreate_e}", exc_info=True)
                raise RuntimeError(f"Could not clear and recreate collection: {e}") from e

# Example usage (for testing or demonstration)
if __name__ == '__main__':
    # Requires QDRANT_URL to be set in environment for this example
    if not os.getenv("QDRANT_URL"):
         print("Please set the QDRANT_URL environment variable to run this example.")
    else:
        try:
            vector_store = VectorStore(collection_name="test_collection")
            print(f"VectorStore initialized for collection: {vector_store.collection_name}")
            # Further example calls can be added here as methods are implemented
            # vector_store.create_collection(vector_size=768) 
            # vector_store.upload_vectors(...)
            # results = vector_store.search(...)
            # vector_store.delete_vectors(...)
        except ConnectionError as e:
            print(f"Initialization failed: {e}")
        except ValueError as e:
            print(f"Configuration error: {e}") 