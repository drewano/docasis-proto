"""
Error handling utilities for the RAG Intelligent Agent

This module provides custom exceptions, error handling decorators, 
and other error-related utilities for the application.
"""

import functools
import traceback
import sys
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast, Tuple

from src.utils.logging_utils import app_logger

# Type variables for generics
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


class RAGAgentError(Exception):
    """Base exception class for all RAG Intelligent Agent errors."""
    
    def __init__(self, message: str = "An error occurred in the RAG Intelligent Agent"):
        self.message = message
        super().__init__(self.message)


class ConfigurationError(RAGAgentError):
    """Exception raised for errors in the configuration."""
    
    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(f"Configuration Error: {message}")


class EmbeddingError(RAGAgentError):
    """Exception raised for errors in the embedding process."""
    
    def __init__(self, message: str = "Failed to generate embeddings"):
        super().__init__(f"Embedding Error: {message}")


class DocumentProcessingError(RAGAgentError):
    """Exception raised for errors in document processing."""
    
    def __init__(self, message: str = "Failed to process document"):
        super().__init__(f"Document Processing Error: {message}")


class VectorStoreError(RAGAgentError):
    """Exception raised for errors in vector store operations."""
    
    def __init__(self, message: str = "Vector store operation failed"):
        super().__init__(f"Vector Store Error: {message}")


class LLMError(RAGAgentError):
    """Exception raised for errors in LLM operations."""
    
    def __init__(self, message: str = "LLM operation failed"):
        super().__init__(f"LLM Error: {message}")


class APIError(RAGAgentError):
    """Exception raised for errors in API operations."""
    
    def __init__(self, message: str = "API request failed"):
        super().__init__(f"API Error: {message}")


def error_handler(
    exception_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    default_return_value: Optional[Any] = None, 
    should_reraise: bool = False,
    error_message: Optional[str] = None,
    log_level: str = "error"
) -> Callable[[F], F]:
    """
    Decorator factory for handling exceptions in a function.
    
    Args:
        exception_types: Exception type or list of exception types to catch
        default_return_value: Value to return if an exception is caught
        should_reraise: Whether to re-raise the caught exception
        error_message: Custom error message to log
        log_level: Log level to use (error, warning, critical, etc.)
    
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                # Determine which log method to use
                log_method = getattr(app_logger, log_level.lower(), app_logger.error)
                
                # Format the error message
                message = error_message or f"Error in {func.__name__}: {str(e)}"
                
                # Include exception details and traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                
                # Log the error with traceback
                log_method(f"{message}\n{tb_str}")
                
                # Re-raise or return default value
                if should_reraise:
                    raise
                return default_return_value
                
        return cast(F, wrapper)
    return decorator


def handle_errors(func: F) -> F:
    """
    Simple decorator to catch and log all exceptions without changing the function's behavior.
    All exceptions are logged and re-raised.
    
    Args:
        func: The function to decorate
    
    Returns:
        Decorated function that logs exceptions
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            app_logger.error(
                f"Exception in {func.__name__}: {e}\n"
                f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
            )
            raise
    return cast(F, wrapper) 