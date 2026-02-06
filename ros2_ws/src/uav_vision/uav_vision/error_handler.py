#!/usr/bin/env python3
"""
Error Handler with Retry Mechanisms
Provides robust error handling and recovery strategies
"""

import time
import logging
from typing import Callable, Optional, Any, Type, Tuple
from functools import wraps
from enum import Enum
import traceback


class RetryStrategy(Enum):
    """Retry strategies"""
    NONE = "none"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


class ErrorHandler:
    """
    Error handler with retry mechanisms and recovery strategies.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retries
            retry_strategy: Retry strategy to use
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            retryable_exceptions: Tuple of exception types that should be retried
        """
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions or (Exception,)
        self.logger = logging.getLogger(__name__)
    
    def retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed. Last error: {e}"
                    )
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay based on retry strategy.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.retry_strategy == RetryStrategy.NONE:
            return 0.0
        elif self.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
            return min(delay, self.max_delay)
        elif self.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
            return min(delay, self.max_delay)
        elif self.retry_strategy == RetryStrategy.FIXED_DELAY:
            return self.base_delay
        else:
            return self.base_delay
    
    def handle_error(
        self,
        error: Exception,
        context: str = "",
        recover_func: Optional[Callable] = None
    ) -> bool:
        """
        Handle an error with optional recovery.
        
        Args:
            error: The exception that occurred
            context: Context information
            recover_func: Optional recovery function
            
        Returns:
            True if error was handled/recovered, False otherwise
        """
        error_msg = f"Error in {context}: {error}" if context else str(error)
        self.logger.error(error_msg, exc_info=True)
        
        if recover_func:
            try:
                self.logger.info(f"Attempting recovery for {context}")
                recover_func()
                self.logger.info(f"Recovery successful for {context}")
                return True
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery failed for {context}: {recovery_error}",
                    exc_info=True
                )
                return False
        
        return False


def retry_on_error(
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
):
    """
    Decorator for automatic retry on errors.
    
    Args:
        max_retries: Maximum number of retries
        retry_strategy: Retry strategy
        base_delay: Base delay in seconds
        retryable_exceptions: Exception types to retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        handler = ErrorHandler(
            max_retries=max_retries,
            retry_strategy=retry_strategy,
            base_delay=base_delay,
            retryable_exceptions=retryable_exceptions
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return handler.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default_return: Any = None,
    error_handler: Optional[ErrorHandler] = None,
    context: str = ""
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        error_handler: Optional error handler
        context: Context information for logging
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        if error_handler:
            error_handler.handle_error(e, context=context)
        else:
            logging.getLogger(__name__).error(
                f"Error in {context}: {e}",
                exc_info=True
            )
        return default_return

