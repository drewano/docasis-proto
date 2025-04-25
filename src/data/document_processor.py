# src/data/document_processor.py (Extrait modifié)
import logging
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Sequence

logger = logging.getLogger(__name__)

# --- MODIFICATION IMPORT DOCLING ---
# Essayer d'abord les chemins recommandés/plus stables ('docling_core')
try:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling.document_converter import DocumentConverter # Souvent stable ici
    from docling_core.types.doc.document import DoclingDocument as Document
    logger.debug("Imported Docling components using recommended/core paths.")
except ImportError as e1:
    logger.warning(f"Could not import from recommended/core paths ({e1}), trying alternative 'docling.*' paths")
    try:
        # Tentative avec les chemins 'docling.*' comme fallback (moins probable de fonctionner si e1 a échoué)
        from docling.document_converter import DocumentConverter
        from docling.chunking import HybridChunker, HierarchicalChunker # L'ancien chemin probable
        from docling_core.types.doc.document import DoclingDocument as Document # Reste souvent ici
        logger.debug("Imported Docling components using alternative 'docling.*' paths.")
    except ImportError as e2:
        logger.error(f"Failed to import necessary Docling components using both paths ({e1}, {e2}). Please ensure 'docling>=2.7.0' and its dependencies (like torch, sentence-transformers) are installed correctly.")
        # Relever l'erreur pour arrêter l'initialisation si Docling n'est pas trouvée
        raise ImportError("Could not find required Docling components. Check installation and paths.") from e2
# --- FIN MODIFICATION IMPORT DOCLING ---

class DocumentProcessor:
    """
    Handles the processing of documents for a RAG system using Docling.
    Responsible for loading, parsing, and chunking documents into segments.
    """
    SUPPORTED_EXTENSIONS: List[str] = ['.pdf', '.docx', '.txt', '.md', '.html', '.epub', '.pptx', '.xlsx']

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DocumentProcessor.
        Args:
            config: Configuration dictionary with document processing parameters
        """
        self.config = config
        doc_config = config.get("document_processing", {})
        self.default_chunk_size: int = doc_config.get("chunk_size", 1000)
        self.default_chunk_overlap: int = doc_config.get("chunk_overlap", 200)
        self.chunk_strategy: str = doc_config.get("chunk_strategy", "hybrid").lower()

        self.converter = DocumentConverter() # Doit être initialisé après les imports réussis

        # Initialize chunkers with proper parameters
        # Le reste de l'initialisation du chunker semble correct,
        # utilisant self.chunk_strategy pour choisir entre Hybrid et Hierarchical
        if self.chunk_strategy == "hybrid":
            try:
                # Ne pas utiliser le modèle Google avec HybridChunker car il n'est pas compatible
                logger.info("Initializing HybridChunker with default settings (no tokenizer)")
                self.chunker = HybridChunker()
                chunker_class_name = "HybridChunker (basic)"
            except Exception as e:
                logger.warning(f"HybridChunker init failed ({e}), falling back to basic init.")
                # Fallback to basic initialization if tokenizer param fails or isn't compatible
                self.chunker = HybridChunker()
                chunker_class_name = "HybridChunker (basic)"
        elif self.chunk_strategy == "hierarchical":
            self.chunker = HierarchicalChunker()
            chunker_class_name = "HierarchicalChunker"
        else:
            logger.warning(
                f"Unknown chunk strategy '{self.chunk_strategy}'. Defaulting to hybrid."
            )
            self.chunk_strategy = "hybrid"
            self.chunker = HybridChunker() # Basic init par défaut
            chunker_class_name = "HybridChunker (defaulted, basic)"

        logger.info(
            f"DocumentProcessor initialized. Strategy: '{self.chunk_strategy}' "
            f"(using {chunker_class_name}). "
            f"Default size/overlap ({self.default_chunk_size}/{self.default_chunk_overlap})."
        )

    # ... (le reste de la classe DocumentProcessor reste identique,
    # y compris process_document, _format_chunks, supports_file_type, etc.)
    # Assurez-vous que l'indentation du fallback dans process_document est correcte.
    # Le code fourni dans la question précédente avait une indentation correcte pour le fallback.

    def truncate_text(self, text, max_chars=1800):
        """ Tronque le texte ... (inchangé) """
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def process_document(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """ Process a single document ... (vérifier indentation du fallback) """
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path.name}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports_file_type(file_path):
            file_ext = file_path.suffix.lower()
            logger.error(f"Unsupported file type: {file_ext}")
            raise ValueError(f"Unsupported file type: {file_ext}")

        try:
            start_time = time.time()

            # 1. Convert document
            try:
                conversion_result = self.converter.convert(str(file_path))
                if not conversion_result or not hasattr(conversion_result, 'document'):
                     logger.error(f"Docling conversion failed or returned unexpected result for {file_path.name}")
                     raise RuntimeError(f"Docling conversion failed for {file_path.name}")
                doc: Document = conversion_result.document
                logger.debug(f"Document converted: {file_path.name}")
            except Exception as conversion_error:
                logger.error(f"Error during Docling conversion for {file_path.name}: {conversion_error}", exc_info=True)
                raise RuntimeError(f"Docling conversion error for {file_path.name}") from conversion_error

            # 2. Get parameters
            chunk_size_override = kwargs.get("chunk_size")
            chunk_overlap_override = kwargs.get("chunk_overlap")
            include_metadata = kwargs.get("include_metadata", True)

            effective_chunk_size = chunk_size_override if chunk_size_override is not None else self.default_chunk_size
            effective_chunk_overlap = chunk_overlap_override if chunk_overlap_override is not None else self.default_chunk_overlap

            # 3. Chunk document with fallback
            try:
                chunk_iter = self.chunker.chunk(doc)
                chunks = list(chunk_iter) # Collecter les chunks
                logger.debug(f"Document chunked: {file_path.name} into {len(chunks)} chunks using '{self.chunk_strategy}'.")
            except Exception as chunk_error:
                logger.error(f"Error during document chunking with '{self.chunk_strategy}': {chunk_error}", exc_info=True)
                logger.warning("Attempting fallback chunking method (Markdown paragraphs)...")
                try:
                    md_text = doc.export_to_markdown()
                    simple_chunks_text = [p for p in md_text.split("\n\n") if p.strip()]

                    # Need to wrap text in a simple object with a 'text' attribute
                    # for _format_chunks compatibility
                    class SimpleChunk:
                         # Indentation correcte ici
                         def __init__(self, text):
                             self.text = text
                         # Add a dummy meta for compatibility if needed by _format_chunks
                         @property
                         def meta(self):
                             return None # Ou un objet Meta vide si nécessaire

                    chunks = [SimpleChunk(text) for text in simple_chunks_text]
                    logger.info(f"Fallback chunking produced {len(chunks)} simple chunks")
                except Exception as fallback_error:
                    logger.error(f"Fallback chunking also failed: {fallback_error}", exc_info=True)
                    raise RuntimeError(f"All chunking methods failed for {file_path.name}") from fallback_error

            # 4. Format chunks
            processed_chunks = self._format_chunks(
                chunks, file_path, include_metadata,
                effective_chunk_size, effective_chunk_overlap
            )

            # 5. Truncate text
            for chunk in processed_chunks:
                 # Ensure 'content' exists before truncating
                 if 'content' in chunk:
                     chunk['content'] = self.truncate_text(chunk['content'])
                 else:
                     logger.warning(f"Chunk missing 'content' before truncation: {chunk.get('metadata', {})}")


            processing_time = time.time() - start_time
            logger.info(
                f"Document processed successfully: {file_path.name} - "
                f"{len(processed_chunks)} chunks created in {processing_time:.2f} seconds"
            )

            return processed_chunks

        except Exception as e:
            logger.exception(f"Error processing document {file_path.name}: {e}", exc_info=True)
            raise

    # _format_chunks, supports_file_type, get_supported_formats, process_text, batch_process_documents
    # restent comme dans le code original fourni.

    def _format_chunks(self, chunks: Sequence[Any], file_path: Path, include_metadata: bool, requested_chunk_size: int, requested_chunk_overlap: int) -> List[Dict[str, Any]]:
        """ Format Docling chunks ... (inchangé) """
        # ... (code comme fourni précédemment)
        formatted_chunks = []
        file_ext = file_path.suffix.lower()
        source_identifier = str(file_path) # Use full path as source ID

        for i, chunk in enumerate(chunks):
            chunk_content = ""
            if hasattr(chunk, 'text') and chunk.text:
                chunk_content = chunk.text
            # ... (autres logiques d'extraction de contenu)
            elif hasattr(chunk, 'serialize') and callable(getattr(chunk, 'serialize')):
                 try:
                     serialized_data = chunk.serialize()
                     if isinstance(serialized_data, dict):
                         chunk_content = serialized_data.get('text', str(serialized_data))
                     else:
                         chunk_content = str(serialized_data)
                 except Exception as serialize_err:
                     logger.warning(f"Serialize failed for chunk {i}: {serialize_err}")
            else:
                chunk_content = str(chunk)
                logger.warning(f"Used fallback string conversion for chunk {i}")

            if not chunk_content.strip():
                logger.warning(f"Skipping empty chunk {i} from {file_path.name}")
                continue

            formatted_chunk = {
                "content": chunk_content,
                "metadata": {
                    "source": source_identifier,
                    "file_name": file_path.name,
                    "file_type": file_ext,
                    "chunk_index": i,
                    "requested_chunk_size": requested_chunk_size,
                    "requested_chunk_overlap": requested_chunk_overlap,
                    "chunker_strategy": self.chunk_strategy,
                    # Ajouter la taille réelle du chunk si disponible et utile
                    "actual_chunk_length_chars": len(chunk_content),
                }
            }

            # ... (extraction de métadonnées détaillées si include_metadata est True)
            if include_metadata:
                 try:
                    # Utilisation de getattr pour plus de sécurité
                    meta_obj = getattr(chunk, 'meta', None)
                    if meta_obj:
                        # Origin
                        origin = getattr(meta_obj, 'origin', None)
                        if origin and hasattr(origin, 'filename'):
                            formatted_chunk["metadata"]["origin_filename"] = origin.filename

                        # Headings
                        headings = getattr(meta_obj, 'headings', None)
                        if headings:
                            formatted_chunk["metadata"]["headings"] = " / ".join(headings)

                        # Page number - tentative plus robuste
                        page_info = None
                        doc_items = getattr(meta_obj, 'doc_items', [])
                        if doc_items:
                            first_item = doc_items[0]
                            prov = getattr(first_item, 'prov', [])
                            if prov:
                                first_prov = prov[0]
                                page_info = getattr(first_prov, 'page_no', None)

                        if page_info is not None:
                            formatted_chunk["metadata"]["page"] = page_info

                 except Exception as meta_err:
                     logger.debug(f"Error extracting detailed metadata from chunk {i}: {meta_err}")


            formatted_chunks.append(formatted_chunk)

        return formatted_chunks


    def supports_file_type(self, file_path: Union[str, Path]) -> bool:
        """Check if a file type is supported ... (inchangé)"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions ... (inchangé)"""
        return self.SUPPORTED_EXTENSIONS

    def process_text(self, text: str, source_name: str = "text_input", **kwargs) -> List[Dict[str, Any]]:
        """ Process a raw text string ... (inchangé) """
        # ... (code comme fourni précédemment)
        if not text or not isinstance(text, str):
             logger.error("Invalid text input for processing.")
             return []
        logger.info(f"Processing text input (length: {len(text)} chars, identifier: {source_name})")
        tmp_file_path = None # Initialiser pour le bloc finally
        try:
             with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file: # Spécifier utf-8
                 tmp_file_path = tmp_file.name
                 tmp_file.write(text)

             chunks = self.process_document(
                 file_path=tmp_file_path,
                 **kwargs
             )

             for chunk in chunks:
                 if "metadata" in chunk:
                     chunk["metadata"]["source"] = source_name
                     chunk["metadata"]["file_name"] = f"{source_name}.txt" # Ou garder vide ?
                     chunk["metadata"]["is_text_input"] = True

             return chunks

        except Exception as e:
             logger.error(f"Error processing text input '{source_name}': {e}", exc_info=True)
             return []

        finally:
             if tmp_file_path:
                 try:
                     Path(tmp_file_path).unlink()
                 except Exception as cleanup_error:
                     logger.warning(f"Failed to remove temporary file '{tmp_file_path}': {cleanup_error}")


    def batch_process_documents(
        self,
        file_paths: List[Union[str, Path]],
        **kwargs
        ) -> Dict[str, Union[List[Dict[str, Any]], List[Dict[str,str]], int]]: # Type hint ajusté
        """ Process multiple documents ... (inchangé) """
        # ... (code comme fourni précédemment)
        processed_results = []
        errors = []
        total_chunks = 0

        for file_path in file_paths:
             try:
                 # Convertir en Path pour la robustesse
                 current_path = Path(file_path)
                 file_chunks = self.process_document(current_path, **kwargs)
                 # Il faut étendre la liste avec les chunks individuels
                 processed_results.extend(file_chunks) # Utiliser extend au lieu de append
                 total_chunks += len(file_chunks)
                 logger.info(f"Successfully processed {current_path.name} into {len(file_chunks)} chunks")

             except Exception as e:
                 error_message = f"Failed to process {file_path}: {e}"
                 logger.error(error_message, exc_info=True)
                 errors.append({
                     "file": str(file_path),
                     "error": str(e) # Garder l'erreur concise pour le retour
                 })

        # Le retour doit correspondre au type hint
        return {
             "processed": processed_results, # C'est une liste de chunks (Dict[str, Any])
             "errors": errors, # C'est une liste de Dict[str, str]
             "total_chunks": total_chunks # C'est un int
        }