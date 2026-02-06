#!/usr/bin/env python3
"""
Performance Optimizer
Provides optimization utilities and recommendations
"""

import time
import numpy as np
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass
import threading
from collections import deque


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    category: str
    issue: str
    recommendation: str
    priority: str  # 'high', 'medium', 'low'
    estimated_improvement: str


class PerformanceOptimizer:
    """
    Performance optimizer that analyzes code and provides recommendations.
    """
    
    def __init__(self):
        """Initialize optimizer"""
        self.recommendations: List[OptimizationRecommendation] = []
    
    def analyze_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> List[OptimizationRecommendation]:
        """
        Analyze a function for optimization opportunities.
        
        Args:
            func: Function to analyze
            *args, **kwargs: Function arguments
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check execution time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if elapsed > 0.1:  # > 100ms
            recommendations.append(OptimizationRecommendation(
                category='timing',
                issue=f'Function {func.__name__} takes {elapsed:.3f}s',
                recommendation='Consider optimizing or caching results',
                priority='high' if elapsed > 1.0 else 'medium',
                estimated_improvement='20-50%'
            ))
        
        return recommendations
    
    def check_data_structure_usage(self, data: any) -> List[OptimizationRecommendation]:
        """
        Check data structure usage for optimization.
        
        Args:
            data: Data structure to check
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check if using list for frequent lookups
        if isinstance(data, list) and len(data) > 100:
            recommendations.append(OptimizationRecommendation(
                category='data_structure',
                issue='Large list used for lookups',
                recommendation='Consider using set or dict for O(1) lookups',
                priority='medium',
                estimated_improvement='10-100x for lookups'
            ))
        
        # Check numpy array usage
        if isinstance(data, np.ndarray):
            if data.dtype == np.float64 and data.size > 1000:
                recommendations.append(OptimizationRecommendation(
                    category='memory',
                    issue='Large float64 array',
                    recommendation='Consider using float32 if precision allows',
                    priority='low',
                    estimated_improvement='50% memory reduction'
                ))
        
        return recommendations


class AsyncProcessor:
    """
    Async processor for non-blocking operations.
    Useful for I/O-bound or CPU-intensive tasks.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize async processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.thread_pool: List[threading.Thread] = []
        self.task_queue = deque()
        self.results: Dict[str, any] = {}
        self.lock = threading.Lock()
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs):
        """
        Submit a task for async processing.
        
        Args:
            task_id: Unique task identifier
            func: Function to execute
            *args, **kwargs: Function arguments
        """
        def task_wrapper():
            try:
                result = func(*args, **kwargs)
                with self.lock:
                    self.results[task_id] = {'result': result, 'status': 'completed'}
            except Exception as e:
                with self.lock:
                    self.results[task_id] = {'error': str(e), 'status': 'failed'}
        
        thread = threading.Thread(target=task_wrapper, daemon=True)
        thread.start()
        self.thread_pool.append(thread)
        
        # Clean up finished threads
        self.thread_pool = [t for t in self.thread_pool if t.is_alive()]
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[any]:
        """
        Get result from async task.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Task result or None
        """
        start = time.time()
        while True:
            with self.lock:
                if task_id in self.results:
                    result = self.results[task_id]
                    if result['status'] == 'completed':
                        return result['result']
                    elif result['status'] == 'failed':
                        raise Exception(result.get('error', 'Task failed'))
            
            if timeout and (time.time() - start) > timeout:
                return None
            
            time.sleep(0.01)  # Small sleep to avoid busy waiting


class RateLimiter:
    """
    Rate limiter to control execution frequency.
    Useful for preventing excessive resource usage.
    """
    
    def __init__(self, max_rate: float):
        """
        Initialize rate limiter.
        
        Args:
            max_rate: Maximum rate (calls per second)
        """
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self.last_call_time = 0.0
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """
        Acquire permission to execute.
        
        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            
            if elapsed < self.min_interval:
                return False
            
            self.last_call_time = current_time
            return True
    
    def wait(self):
        """Wait until next allowed execution"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            
            self.last_call_time = time.time()


class Cache:
    """
    Simple caching utility for function results.
    """
    
    def __init__(self, max_size: int = 100, ttl: Optional[float] = None):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time to live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}  # {key: (value, timestamp)}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            value, timestamp = self.cache[key]
            
            # Check TTL
            if self.ttl and (time.time() - timestamp) > self.ttl:
                del self.cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: any):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)

