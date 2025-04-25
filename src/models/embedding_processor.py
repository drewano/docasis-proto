import google.generativeai as genai
import os
import logging
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class EmbeddingProcessor:
    """
    Handles text embedding generation using Google's Generative AI models,
    including batching, caching, and error handling.
    """
    def __init__(self,
                 model_name: str = "models/text-embedding-004",
                 batch_size: int = 100,
                 max_retries: int = 3):
        """
        Initializes the EmbeddingProcessor and configures the Google Generative AI library.

        Args:
            model_name (str): The name of the embedding model to use.
            batch_size (int): The number of chunks to process in each API call batch.
            max_retries (int): Maximum number of retries for transient API errors.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY must be set.")

        try:
            # Configure the genai library globally using the API key
            # Pylance might warn about 'configure' not being exported, but this is the documented way.
            genai.configure(api_key=api_key) # type: ignore # Acknowledging potential linter warning
            logging.info("Google Generative AI library configured successfully.")
            # Although configure is used here, API calls might internally use a default client.
            # Explicit client creation is also possible: self.client = genai.Client(api_key=api_key)
        except Exception as e:
            logging.error(f"Failed to configure Google Generative AI library: {e}")
            raise

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache: Dict[str, List[float]] = {}  # In-memory cache
        logging.info(f"EmbeddingProcessor initialized with model: {self.model_name}, batch size: {self.batch_size}")

    def _embed_batch(self, batch_texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generates embeddings for a single batch of texts using the configured model.

        Args:
            batch_texts (List[str]): A list of text strings to embed.

        Returns:
            Optional[List[List[float]]]: A list of embedding vectors, or None if an error occurs.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                # Call the embedding function directly from the genai module
                # Pylance might warn about 'embed_content' not being exported, but this is functional.
                # Alternatively, use an explicit client: result = self.client.models.embed_content(...)
                # Task type 'RETRIEVAL_DOCUMENT' is suitable for embeddings stored for later retrieval.
                result = genai.embed_content(model=self.model_name, # type: ignore # Acknowledging potential linter warning
                                             content=batch_texts,
                                             task_type="RETRIEVAL_DOCUMENT") # Use RETRIEVAL_DOCUMENT for storing

                embeddings = result.get('embedding')

                # Check if embeddings were successfully retrieved and have the expected format (List[List[float]])
                if embeddings is not None and isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
                    logging.info(f"Successfully embedded batch of {len(batch_texts)} chunks.")
                     # Pylance incorrectly infers List[float] here, but for batch input,
                     # the API returns List[List[float]]. The type hint is correct.
                    return embeddings # type: ignore # Acknowledging linter type mismatch error
                elif embeddings is None:
                     logging.error("API call succeeded but returned no 'embedding' field.")
                else:
                     logging.error(f"Unexpected embedding format received from API: {type(embeddings)}")

            except Exception as e:
                retries += 1
                logging.warning(f"API error embedding batch (attempt {retries}/{self.max_retries}): {e}")
                if retries < self.max_retries:
                    sleep_time = 2 ** retries
                    logging.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Failed to embed batch after {self.max_retries} attempts.")
                    return None

        return None

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a list of documents (text chunks with metadata).
        Uses caching and batching for efficiency.

        Args:
            documents (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                              must have a 'text' key containing the string chunk
                                              and optionally other keys for metadata.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the original metadata
                                  and the generated 'embedding' vector (or None if failed/skipped).
        """
        final_results: List[Dict[str, Any]] = [{**doc, 'embedding': None} for doc in documents]
        chunks_to_process_indices: List[int] = []
        chunk_texts_to_process: List[str] = []

        # 1. Check cache and identify chunks needing processing
        for i, doc in enumerate(documents):
            text = doc.get('text')
            if not text or not isinstance(text, str):
                logging.warning(f"Skipping document at index {i} due to missing or invalid 'text' field.")
                continue

            if text in self.cache:
                logging.debug(f"Cache hit for chunk at index {i}: {text[:50]}...")
                final_results[i]['embedding'] = self.cache[text]
            else:
                chunks_to_process_indices.append(i)
                chunk_texts_to_process.append(text)

        # 2. Process chunks in batches
        if chunk_texts_to_process:
            logging.info(f"Processing {len(chunk_texts_to_process)} chunks not found in cache.")
            processed_embeddings: Dict[int, Optional[List[float]]] = {}

            for i in range(0, len(chunks_to_process_indices), self.batch_size):
                batch_indices = chunks_to_process_indices[i:i + self.batch_size]
                # Get texts corresponding to these original indices
                batch_texts = [documents[idx]['text'] for idx in batch_indices] # Assumes valid text checked previously

                if not batch_texts:
                    logging.warning(f"Skipping empty batch starting at index {i}.")
                    continue

                logging.info(f"Processing batch {i // self.batch_size + 1} covering original indices {batch_indices[0]} to {batch_indices[-1]} ({len(batch_texts)} chunks).")
                batch_embeddings = self._embed_batch(batch_texts) # Returns Optional[List[List[float]]]

                if batch_embeddings and len(batch_embeddings) == len(batch_texts):
                    for original_index, text, embedding in zip(batch_indices, batch_texts, batch_embeddings):
                        self.cache[text] = embedding
                        processed_embeddings[original_index] = embedding
                else:
                    logging.error(f"Failed to retrieve embeddings for batch covering indices {batch_indices[0]}-{batch_indices[-1]}. Marking as None.")
                    for original_index in batch_indices:
                        processed_embeddings[original_index] = None

            # 3. Update final_results with newly processed embeddings
            for original_index, embedding in processed_embeddings.items():
                 if original_index < len(final_results): # Boundary check
                     final_results[original_index]['embedding'] = embedding
                 else:
                     logging.error(f"Attempted to update out-of-bounds index {original_index} in final_results.")

        else:
             logging.info("All requested document chunks were found in the cache.")

        return final_results

    def get_embedding_model(self):
        """
        Returns a LangChain compatible embedding model.
        
        Returns:
            GoogleGenerativeAIEmbeddings: LangChain compatible embedding model
        """
        try:
            # LangChain's GoogleGenerativeAIEmbeddings doesn't support max_input_size
            # Token truncation is handled by document_processor.truncate_text()
            api_key = os.getenv("GOOGLE_API_KEY")
            # Create the SecretStr only if the API key is not None
            secret_key = SecretStr(api_key) if api_key is not None else None
            return GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                task_type="RETRIEVAL_DOCUMENT",
                google_api_key=secret_key,
                client=None  # Use default client
            )
        except Exception as e:
            logging.error(f"Erreur lors de la création du modèle d'embedding: {e}")
            # Utiliser un modèle de secours si disponible
            try:
                api_key = os.getenv("GOOGLE_API_KEY")
                secret_key = SecretStr(api_key) if api_key is not None else None
                return GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",  # Modèle de secours
                    task_type="RETRIEVAL_DOCUMENT",
                    google_api_key=secret_key
                )
            except Exception as fallback_error:
                logging.error(f"Erreur avec le modèle de secours: {fallback_error}")
                raise RuntimeError(f"Impossible de créer un modèle d'embedding: {e}")

# Example Usage (remains the same)
if __name__ == '__main__':
    # Ensure you have a .env file with GOOGLE_API_KEY="YOUR_API_KEY"
    processor = EmbeddingProcessor(batch_size=5)

    docs_to_embed = [
        {'text': 'This is the first document chunk.', 'source': 'doc1.txt', 'page': 1},
        {'text': 'Here comes the second piece of text.', 'source': 'doc2.pdf', 'chunk_id': 'a'},
        {'text': 'A third chunk for testing purposes.', 'source': 'doc1.txt', 'page': 2},
        {'text': 'This is the first document chunk.', 'source': 'doc3.md'}, # Duplicate text
        {'text': 'Final chunk.'}, # No metadata other than text
        {'invalid': 'no text field'}, # Invalid document
        {'text': ['list', 'is', 'not', 'string']}, # Invalid text type
        {'text': 'Another valid chunk to test batching.'},
        {'text': 'Yet another chunk.'},
        {'text': 'More text data here.'},
        {'text': 'Chunk number ten.'},
        {'text': 'The final eleventh chunk.'},
    ]

    print("--- Initial Embedding Run ---")
    start_time = time.time()
    embedded_results = processor.embed_documents(docs_to_embed)
    end_time = time.time()
    print(f"First run took {end_time - start_time:.4f} seconds.")

    print("\n--- Embedding Results ---")
    valid_embeddings = 0
    failed_embeddings = 0
    skipped_docs = 0
    for i, result in enumerate(embedded_results):
        print(f"Document {i}:")
        is_valid = 'text' in result and isinstance(result.get('text'), str)
        print(f"  Text: {result.get('text', 'N/A')[:30] if isinstance(result.get('text'), str) else type(result.get('text'))}...")
        print(f"  Metadata: { {k:v for k,v in result.items() if k not in ['text', 'embedding']} }")
        if result.get('embedding'):
            print(f"  Embedding Vector Dim: {len(result['embedding'])}")
            valid_embeddings += 1
        elif not is_valid:
             print("  Embedding: Skipped (invalid input)")
             skipped_docs += 1
        else:
            print("  Embedding: Failed")
            failed_embeddings +=1
        print("-" * 10)
    print(f"Summary: Valid={valid_embeddings}, Failed={failed_embeddings}, Skipped={skipped_docs}")

    print("\n--- Running again to test caching ---")
    start_time = time.time()
    cached_results = processor.embed_documents(docs_to_embed)
    end_time = time.time()
    print(f"Second run took {end_time - start_time:.4f} seconds (should be faster due to cache).")

    cached_valid = sum(1 for r in cached_results if r.get('embedding'))
    cached_failed = sum(1 for i, r in enumerate(cached_results) if not r.get('embedding') and 'text' in r and isinstance(r.get('text'), str))
    cached_skipped = sum(1 for r in cached_results if 'text' not in r or not isinstance(r.get('text'), str))

    print(f"Second Run Summary: Valid={cached_valid}, Failed={cached_failed}, Skipped={cached_skipped}")
    assert valid_embeddings == cached_valid
    assert failed_embeddings == cached_failed
    assert skipped_docs == cached_skipped

    print(f"Cache size: {len(processor.cache)}")