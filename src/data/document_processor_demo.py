"""
Demo script for the DocumentProcessor

This script demonstrates how to use the DocumentProcessor class with sample documents.
Run this script directly to test document processing functionality.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Union

# Add the project root to the Python path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.document_processor import DocumentProcessor
from src.utils.config import get_document_processing_config, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_file(file_path: Union[str, Path], config: Dict[str, Any]) -> None:
    """
    Process a single file using the DocumentProcessor.
    
    Args:
        file_path: Path to the document file
        config: Configuration dictionary
    """
    processor = DocumentProcessor(config)
    
    # Convert to Path if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    if not processor.supports_file_type(file_path):
        logger.error(f"Unsupported file type: {file_path.suffix}")
        return
    
    logger.info(f"Processing document: {file_path}")
    
    try:
        # Time the processing
        start_time = time.time()
        
        # Process the document
        chunks = processor.process_document(file_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Document processed in {processing_time:.2f} seconds")
        logger.info(f"Created {len(chunks)} chunks")
        
        # Print some sample chunks for verification
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"Content: {chunk['content'][:100]}...")
            
            if 'metadata' in chunk:
                print("Metadata:")
                for key, value in chunk['metadata'].items():
                    print(f"  - {key}: {value}")
        
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")


def process_text(text: str, config: Dict[str, Any]) -> None:
    """
    Process a text string using the DocumentProcessor.
    
    Args:
        text: Text to process
        config: Configuration dictionary
    """
    processor = DocumentProcessor(config)
    
    try:
        # Time the processing
        start_time = time.time()
        
        # Process the text
        chunks = processor.process_text(text, source_name="demo_text")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Text processed in {processing_time:.2f} seconds")
        logger.info(f"Created {len(chunks)} chunks")
        
        # Print some sample chunks for verification
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"Content: {chunk['content'][:100]}...")
            
            if 'metadata' in chunk:
                print("Metadata:")
                for key, value in chunk['metadata'].items():
                    print(f"  - {key}: {value}")
        
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")
            
    except Exception as e:
        logger.error(f"Error processing text: {e}")


def load_config() -> Dict[str, Any]:
    """
    Load configuration for the document processor.
    
    Returns:
        Configuration dictionary with document processing settings
    """
    try:
        # Try to use the project's config system
        document_config = get_document_processing_config()
        
        # Return a config dict with the expected structure
        return {
            "document_processing": {
                "chunk_size": document_config.get("MAX_CHUNK_SIZE", 1000),
                "chunk_overlap": document_config.get("CHUNK_OVERLAP", 200),
                "chunk_strategy": "hybrid"
            }
        }
    except Exception as e:
        # If something goes wrong, create a default config
        logger.warning(f"Failed to load config from config system: {e}")
        logger.info("Using default configuration")
        
        return {
            "document_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "chunk_strategy": "hybrid"
            }
        }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Demo for DocumentProcessor")
    parser.add_argument("--file", "-f", help="Path to a document file to process")
    parser.add_argument("--text", "-t", help="Text to process instead of a file")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size to use for processing")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap to use for processing")
    parser.add_argument("--strategy", choices=["hybrid", "hierarchical"], default="hybrid", help="Chunking strategy to use")
    
    args = parser.parse_args()
    
    # Check if we have something to process
    if not args.file and not args.text:
        parser.print_help()
        print("\nError: Please provide either a file path or text to process")
        return
    
    try:
        # Load configuration
        config = load_config()
        
        # Override config with command-line arguments
        config.setdefault("document_processing", {})
        config["document_processing"]["chunk_size"] = args.chunk_size
        config["document_processing"]["chunk_overlap"] = args.chunk_overlap
        config["document_processing"]["chunk_strategy"] = args.strategy
        
        # Process file or text
        if args.file:
            process_file(args.file, config)
        elif args.text:
            process_text(args.text, config)
            
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        

if __name__ == "__main__":
    main() 