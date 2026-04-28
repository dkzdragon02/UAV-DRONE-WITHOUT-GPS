import pytest
import time
from unittest.mock import Mock, patch

from uav_vision.error_handler import (
    ErrorHandler,
    RetryStrategy,
    retry_on_error,
    safe_execute
)


class TestErrorHandler:
    def test_init_default(self):
        handler = ErrorHandler()
        
        assert handler.max_retries == 3
        assert handler.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert handler.base_delay == 1.0
    
    def test_init_custom(self):
        handler = ErrorHandler(
            max_retries=5,
            retry_strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=2.0
        )
        
        assert handler.max_retries == 5
        assert handler.retry_strategy == RetryStrategy.LINEAR_BACKOFF
        assert handler.base_delay == 2.0
    
    def test_retry_success_first_attempt(self):
        handler = ErrorHandler(max_retries=3)
        
        func = Mock(return_value=42)
        result = handler.retry(func)
        
        assert result == 42
        assert func.call_count == 1
    
    def test_retry_success_after_failures(self):
        handler = ErrorHandler(max_retries=3)
        
        func = Mock(side_effect=[ValueError("Error"), ValueError("Error"), 42])
        result = handler.retry(func)
        
        assert result == 42
        assert func.call_count == 3
    
    def test_retry_all_failures(self):
        handler = ErrorHandler(max_retries=2)
        
        func = Mock(side_effect=ValueError("Error"))
        
        with pytest.raises(ValueError):
            handler.retry(func)
        
        assert func.call_count == 3  # max_retries + 1
    
    def test_retry_strategy_none(self):
        handler = ErrorHandler(
            max_retries=1,
            retry_strategy=RetryStrategy.NONE
        )
        
        func = Mock(side_effect=[ValueError("Error"), 42])
        start = time.time()
        result = handler.retry(func)
        elapsed = time.time() - start
        
        assert result == 42
        assert elapsed < 0.1  # Should be very fast
    
    def test_retry_strategy_exponential_backoff(self):
        handler = ErrorHandler(
            max_retries=2,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=0.1
        )
        
        delays = []
        original_sleep = time.sleep
        
        def mock_sleep(duration):
            delays.append(duration)
            original_sleep(0.01)  # Short sleep for testing
        
        func = Mock(side_effect=[ValueError("Error"), ValueError("Error"), 42])
        
        with patch('time.sleep', mock_sleep):
            handler.retry(func)
        
        assert len(delays) == 2
        assert delays[0] == pytest.approx(0.1, rel=0.1)
        assert delays[1] == pytest.approx(0.2, rel=0.1)
    
    def test_retry_strategy_linear_backoff(self):
        handler = ErrorHandler(
            max_retries=2,
            retry_strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=0.1
        )
        
        delays = []
        original_sleep = time.sleep
        
        def mock_sleep(duration):
            delays.append(duration)
            original_sleep(0.01)
        
        func = Mock(side_effect=[ValueError("Error"), ValueError("Error"), 42])
        
        with patch('time.sleep', mock_sleep):
            handler.retry(func)
        
        assert len(delays) == 2
        assert delays[0] == pytest.approx(0.1, rel=0.1)
        assert delays[1] == pytest.approx(0.2, rel=0.1)
    
    def test_retry_strategy_fixed_delay(self):
        handler = ErrorHandler(
            max_retries=2,
            retry_strategy=RetryStrategy.FIXED_DELAY,
            base_delay=0.1
        )
        
        delays = []
        original_sleep = time.sleep
        
        def mock_sleep(duration):
            delays.append(duration)
            original_sleep(0.01)
        
        func = Mock(side_effect=[ValueError("Error"), ValueError("Error"), 42])
        
        with patch('time.sleep', mock_sleep):
            handler.retry(func)
        
        assert len(delays) == 2
        assert all(d == pytest.approx(0.1, rel=0.1) for d in delays)
    
    def test_retry_max_delay(self):
        handler = ErrorHandler(
            max_retries=5,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=10.0,
            max_delay=1.0
        )
        
        delays = []
        original_sleep = time.sleep
        
        def mock_sleep(duration):
            delays.append(duration)
            original_sleep(0.01)
        
        func = Mock(side_effect=[ValueError("Error")] * 5 + [42])
        
        with patch('time.sleep', mock_sleep):
            handler.retry(func)
        
        assert all(d <= 1.0 for d in delays)
    
    def test_retryable_exceptions(self):
        handler = ErrorHandler(
            max_retries=1,
            retryable_exceptions=(ValueError,)
        )
        
        func1 = Mock(side_effect=[ValueError("Error"), 42])
        result = handler.retry(func1)
        assert result == 42
        assert func1.call_count == 2
        
        func2 = Mock(side_effect=KeyError("Error"))
        with pytest.raises(KeyError):
            handler.retry(func2)
        assert func2.call_count == 1
    
    def test_handle_error_with_recovery(self):
        handler = ErrorHandler()
        recovery_func = Mock()
        
        error = ValueError("Test error")
        result = handler.handle_error(error, context="test", recover_func=recovery_func)
        
        assert result is True
        assert recovery_func.call_count == 1
    
    def test_handle_error_without_recovery(self):
        handler = ErrorHandler()
        
        error = ValueError("Test error")
        result = handler.handle_error(error, context="test")
        
        assert result is False


class TestRetryDecorator:
    def test_decorator_success(self):
        @retry_on_error(max_retries=2)
        def test_func():
            return 42
        
        result = test_func()
        assert result == 42
    
    def test_decorator_retry(self):
        call_count = [0]
        
        @retry_on_error(max_retries=2, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Error")
            return 42
        
        result = test_func()
        assert result == 42
        assert call_count[0] == 3
    
    def test_decorator_all_failures(self):
        @retry_on_error(max_retries=2)
        def test_func():
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            test_func()


class TestSafeExecute:
    def test_safe_execute_success(self):
        func = Mock(return_value=42)
        result = safe_execute(func, default_return=None)
        
        assert result == 42
        func.assert_called_once()
    
    def test_safe_execute_error(self):
        func = Mock(side_effect=ValueError("Error"))
        result = safe_execute(func, default_return=None)
        
        assert result is None
        func.assert_called_once()
    
    def test_safe_execute_with_error_handler(self):
        handler = ErrorHandler()
        handler.handle_error = Mock(return_value=True)
        
        func = Mock(side_effect=ValueError("Error"))
        result = safe_execute(
            func,
            default_return=None,
            error_handler=handler,
            context="test"
        )
        
        assert result is None
        assert handler.handle_error.called

