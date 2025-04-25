"""
Test module for the DocumentProcessor class.

This module contains unit tests for the document processing functionality.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.data.document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for the DocumentProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config = {
            "document_processing": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "chunk_strategy": "hybrid"
            }
        }
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.temp_dir.name)
        
        # Create a sample text file for testing
        self.sample_text = "This is a sample document for testing.\n" * 10
        self.sample_txt_path = self.test_dir_path / "sample.txt"
        with open(self.sample_txt_path, "w") as f:
            f.write(self.sample_text)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.temp_dir.cleanup()

    @patch("src.data.document_processor.DocumentConverter")
    @patch("src.data.document_processor.HybridChunker")
    def test_initialization(self, mock_chunker, mock_converter):
        """Test that DocumentProcessor initializes correctly."""
        processor = DocumentProcessor(self.test_config)
        
        # Check that the configuration was set correctly
        self.assertEqual(processor.config["document_processing"]["chunk_size"], 500)
        self.assertEqual(processor.config["document_processing"]["chunk_overlap"], 100)
        self.assertEqual(processor.config["document_processing"]["chunk_strategy"], "hybrid")
        
        # Check that Docling components were initialized
        mock_converter.assert_called_once()
        mock_chunker.assert_called_once_with(chunk_size=500, chunk_overlap=100)

    @patch("src.data.document_processor.DocumentConverter")
    @patch("src.data.document_processor.HybridChunker")
    def test_supports_file_type(self, mock_chunker, mock_converter):
        """Test file type support checking."""
        processor = DocumentProcessor(self.test_config)
        
        # Test supported file types
        self.assertTrue(processor.supports_file_type("document.pdf"))
        self.assertTrue(processor.supports_file_type("document.docx"))
        self.assertTrue(processor.supports_file_type("document.txt"))
        self.assertTrue(processor.supports_file_type("document.md"))
        self.assertTrue(processor.supports_file_type("document.html"))
        self.assertTrue(processor.supports_file_type("document.epub"))
        
        # Test unsupported file types
        self.assertFalse(processor.supports_file_type("document.xlsx"))
        self.assertFalse(processor.supports_file_type("document.pptx"))
        self.assertFalse(processor.supports_file_type("document.zip"))

    @patch("src.data.document_processor.DocumentConverter")
    @patch("src.data.document_processor.HybridChunker")
    def test_get_supported_formats(self, mock_chunker, mock_converter):
        """Test getting supported formats."""
        processor = DocumentProcessor(self.test_config)
        supported_formats = processor.get_supported_formats()
        
        self.assertIn(".pdf", supported_formats)
        self.assertIn(".docx", supported_formats)
        self.assertIn(".txt", supported_formats)
        self.assertIn(".md", supported_formats)
        self.assertIn(".html", supported_formats)
        self.assertIn(".epub", supported_formats)
        self.assertEqual(len(supported_formats), 6)

    @patch("src.data.document_processor.DocumentConverter")
    @patch("src.data.document_processor.HybridChunker")
    def test_process_document_file_not_found(self, mock_chunker, mock_converter):
        """Test processing a non-existent file."""
        processor = DocumentProcessor(self.test_config)
        
        # Test with a non-existent file
        with self.assertRaises(FileNotFoundError):
            processor.process_document("non_existent_file.pdf")

    @patch("src.data.document_processor.DocumentConverter")
    @patch("src.data.document_processor.HybridChunker")
    def test_process_document_unsupported_file_type(self, mock_chunker, mock_converter):
        """Test processing an unsupported file type."""
        processor = DocumentProcessor(self.test_config)
        
        # Create a temporary file with unsupported extension
        unsupported_file = self.test_dir_path / "unsupported.xlsx"
        with open(unsupported_file, "w") as f:
            f.write("Some content")
        
        # Test with an unsupported file type
        with self.assertRaises(ValueError):
            processor.process_document(unsupported_file)

    @patch("src.data.document_processor.DocumentConverter")
    @patch("src.data.document_processor.HybridChunker")
    def test_process_document(self, mock_chunker, mock_converter):
        """Test processing a document."""
        # Set up mocks
        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter.return_value.convert.return_value = mock_result
        
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Chunk 1 content"
        mock_chunk1.meta.headings = ["Heading 1"]
        mock_chunk1.meta.doc_items = [MagicMock()]
        mock_chunk1.meta.doc_items[0].prov = [MagicMock()]
        mock_chunk1.meta.doc_items[0].prov[0].page_no = 1
        mock_chunk1.meta.doc_items[0].prov[0].bbox = MagicMock(l=10, t=20, r=30, b=40)
        mock_chunk1.meta.origin = MagicMock()
        mock_chunk1.meta.origin.filename = "original.pdf"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Chunk 2 content"
        mock_chunk2.meta.headings = ["Heading 2"]
        mock_chunk2.meta.doc_items = [MagicMock()]
        mock_chunk2.meta.doc_items[0].prov = [MagicMock()]
        mock_chunk2.meta.doc_items[0].prov[0].page_no = 2
        mock_chunk2.meta.doc_items[0].prov[0].bbox = MagicMock(l=15, t=25, r=35, b=45)
        mock_chunk2.meta.origin = MagicMock()
        mock_chunk2.meta.origin.filename = "original.pdf"
        
        mock_chunker.return_value.chunk.return_value = [mock_chunk1, mock_chunk2]
        
        # Initialize processor and process document
        processor = DocumentProcessor(self.test_config)
        result = processor.process_document(self.sample_txt_path)
        
        # Check the result
        self.assertEqual(len(result), 2)
        
        # Check first chunk
        self.assertEqual(result[0]["content"], "Chunk 1 content")
        self.assertEqual(result[0]["metadata"]["source"], str(self.sample_txt_path))
        self.assertEqual(result[0]["metadata"]["file_type"], ".txt")
        self.assertEqual(result[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(result[0]["metadata"]["headings"], ["Heading 1"])
        self.assertEqual(result[0]["metadata"]["page"], 1)
        self.assertEqual(result[0]["metadata"]["bbox"]["left"], 10)
        self.assertEqual(result[0]["metadata"]["bbox"]["top"], 20)
        self.assertEqual(result[0]["metadata"]["bbox"]["right"], 30)
        self.assertEqual(result[0]["metadata"]["bbox"]["bottom"], 40)
        self.assertEqual(result[0]["metadata"]["original_filename"], "original.pdf")
        
        # Check second chunk
        self.assertEqual(result[1]["content"], "Chunk 2 content")
        self.assertEqual(result[1]["metadata"]["chunk_index"], 1)

    @patch("src.data.document_processor.Document")
    @patch("src.data.document_processor.HybridChunker")
    def test_process_text(self, mock_chunker, mock_document):
        """Test processing raw text."""
        # Set up mocks
        mock_doc = MagicMock()
        mock_document.from_text.return_value = mock_doc
        
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Text chunk 1"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Text chunk 2"
        
        mock_chunker.return_value.chunk.return_value = [mock_chunk1, mock_chunk2]
        
        # Initialize processor and process text
        processor = DocumentProcessor(self.test_config)
        processor.chunker = mock_chunker.return_value
        
        result = processor.process_text("Sample text content", source_name="test_source")
        
        # Check the result
        self.assertEqual(len(result), 2)
        
        # Check first chunk
        self.assertEqual(result[0]["content"], "Text chunk 1")
        self.assertEqual(result[0]["metadata"]["source"], "test_source")
        self.assertEqual(result[0]["metadata"]["file_type"], "text")
        self.assertEqual(result[0]["metadata"]["chunk_index"], 0)
        
        # Check second chunk
        self.assertEqual(result[1]["content"], "Text chunk 2")
        self.assertEqual(result[1]["metadata"]["chunk_index"], 1)

    @patch("src.data.document_processor.DocumentConverter")
    @patch("src.data.document_processor.HybridChunker")
    def test_batch_process_documents(self, mock_chunker, mock_converter):
        """Test batch processing of documents."""
        # Create multiple test files
        file1 = self.test_dir_path / "test1.txt"
        file2 = self.test_dir_path / "test2.txt"
        
        with open(file1, "w") as f:
            f.write("Test content 1")
        
        with open(file2, "w") as f:
            f.write("Test content 2")
        
        # Set up mocks
        def mock_process_document(file_path, **kwargs):
            if "test1.txt" in str(file_path):
                return [{"content": "Chunk from test1.txt", "metadata": {"source": str(file_path)}}]
            else:
                return [{"content": "Chunk from test2.txt", "metadata": {"source": str(file_path)}}]
        
        processor = DocumentProcessor(self.test_config)
        processor.process_document = mock_process_document
        
        # Test batch processing
        results = processor.batch_process_documents([file1, file2])
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[str(file1)]), 1)
        self.assertEqual(len(results[str(file2)]), 1)
        # Access dictionary-like objects properly
        self.assertEqual(results[str(file1)][0].get("content"), "Chunk from test1.txt")
        self.assertEqual(results[str(file2)][0].get("content"), "Chunk from test2.txt")


if __name__ == "__main__":
    unittest.main() 