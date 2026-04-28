import time
import numpy as np
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass
import threading
from collections import deque

@dataclass
class OptimizationRecommendation:
    category: str
    issue: str
    recommendation: str
    priority: str  # 'high', 'medium', 'low'
    estimated_improvement: str

class PerformanceOptimizer:
    def __init__(self):
        self.recommendations: List[OptimizationRecommendation] = []
    def analyze_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> List[OptimizationRecommendation]:
        recommendations = []
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
        recommendations = []
        if isinstance(data, list) and len(data) > 100:
            recommendations.append(OptimizationRecommendation(
                category='data_structure',
                issue='Large list used for lookups',
                recommendation='Consider using set or dict for O(1) lookups',
                priority='medium',
                estimated_improvement='10-100x for lookups'
            ))

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
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool: List[threading.Thread] = []
        self.task_queue = deque()
        self.results: Dict[str, any] = {}
        self.lock = threading.Lock()
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs):
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
        self.thread_pool = [t for t in self.thread_pool if t.is_alive()]
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[any]:
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
    def __init__(self, max_rate: float):
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self.last_call_time = 0.0
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            if elapsed < self.min_interval:
                return False
            self.last_call_time = current_time
            return True
    
    def wait(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()

class Cache:
    def __init__(self, max_size: int = 100, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}  # {key: (value, timestamp)}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[any]:
        with self.lock:
            if key not in self.cache:
                return None
            value, timestamp = self.cache[key]
            if self.ttl and (time.time() - timestamp) > self.ttl:
                del self.cache[key]
                return None
            return value
    
    def set(self, key: str, value: any):
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)

