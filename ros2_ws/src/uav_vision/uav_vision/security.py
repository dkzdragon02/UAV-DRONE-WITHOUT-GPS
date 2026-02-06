#!/usr/bin/env python3
"""
Security Utilities
Provides authentication, validation, and security features
"""

import hashlib
import hmac
import secrets
import re
from typing import Optional, Dict, Any, Callable
from functools import wraps
import time


class InputValidator:
    """
    Input validation utilities.
    """
    
    @staticmethod
    def validate_position(position: Any) -> bool:
        """
        Validate position data.
        
        Args:
            position: Position data (list, tuple, or array)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(position, (list, tuple)):
                return False
            
            if len(position) < 2 or len(position) > 3:
                return False
            
            # Check all values are numeric
            for val in position:
                if not isinstance(val, (int, float)):
                    return False
                # Check reasonable bounds
                if abs(val) > 10000:  # 10km max
                    return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_quaternion(quaternion: Any) -> bool:
        """
        Validate quaternion.
        
        Args:
            quaternion: Quaternion [x, y, z, w]
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(quaternion, (list, tuple)):
                return False
            
            if len(quaternion) != 4:
                return False
            
            # Check all values are numeric
            for val in quaternion:
                if not isinstance(val, (int, float)):
                    return False
            
            # Check normalization (approximately)
            q = [float(v) for v in quaternion]
            norm = sum(v * v for v in q) ** 0.5
            if abs(norm - 1.0) > 0.1:  # Allow some tolerance
                return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_string(
        string: str,
        max_length: int = 1000,
        pattern: Optional[str] = None
    ) -> bool:
        """
        Validate string input.
        
        Args:
            string: String to validate
            max_length: Maximum length
            pattern: Optional regex pattern
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(string, str):
            return False
        
        if len(string) > max_length:
            return False
        
        if pattern:
            if not re.match(pattern, string):
                return False
        
        return True
    
    @staticmethod
    def validate_numeric(
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> bool:
        """
        Validate numeric value.
        
        Args:
            value: Value to validate
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, (int, float)):
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True


class Authenticator:
    """
    Simple authentication system.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize authenticator.
        
        Args:
            secret_key: Secret key for HMAC (None to generate)
        """
        if secret_key is None:
            self.secret_key = secrets.token_hex(32)
        else:
            self.secret_key = secret_key
        
        self.tokens: Dict[str, Dict] = {}
        self.token_timeout = 3600  # 1 hour
    
    def generate_token(self, user_id: str) -> str:
        """
        Generate authentication token.
        
        Args:
            user_id: User identifier
            
        Returns:
            Authentication token
        """
        timestamp = str(int(time.time()))
        message = f"{user_id}:{timestamp}"
        
        token = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Store token
        self.tokens[token] = {
            'user_id': user_id,
            'timestamp': int(timestamp),
            'expires': int(timestamp) + self.token_timeout
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate authentication token.
        
        Args:
            token: Token to validate
            
        Returns:
            User ID if valid, None otherwise
        """
        if token not in self.tokens:
            return None
        
        token_data = self.tokens[token]
        
        # Check expiration
        if time.time() > token_data['expires']:
            del self.tokens[token]
            return None
        
        return token_data['user_id']
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked, False if not found
        """
        if token in self.tokens:
            del self.tokens[token]
            return True
        return False
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens"""
        current_time = time.time()
        expired = [
            token for token, data in self.tokens.items()
            if current_time > data['expires']
        ]
        for token in expired:
            del self.tokens[token]


def require_authentication(authenticator: Authenticator):
    """
    Decorator to require authentication.
    
    Args:
        authenticator: Authenticator instance
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract token from kwargs or args
            token = kwargs.get('token') or (args[0] if args else None)
            
            if not token:
                raise ValueError("Authentication token required")
            
            user_id = authenticator.validate_token(token)
            if not user_id:
                raise ValueError("Invalid or expired token")
            
            # Add user_id to kwargs
            kwargs['user_id'] = user_id
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_input(validator_func: Callable):
    """
    Decorator to validate function inputs.
    
    Args:
        validator_func: Validation function
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            for arg in args:
                if not validator_func(arg):
                    raise ValueError(f"Invalid input: {arg}")
            
            for key, value in kwargs.items():
                if not validator_func(value):
                    raise ValueError(f"Invalid input for {key}: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiter for API calls.
    """
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Dict[str, list] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if call is allowed.
        
        Args:
            identifier: Caller identifier
            
        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()
        
        if identifier not in self.calls:
            self.calls[identifier] = []
        
        # Remove old calls
        self.calls[identifier] = [
            t for t in self.calls[identifier]
            if current_time - t < self.time_window
        ]
        
        # Check limit
        if len(self.calls[identifier]) >= self.max_calls:
            return False
        
        # Record call
        self.calls[identifier].append(current_time)
        return True

