import cProfile
import pstats
import io
import time
from functools import wraps
from typing import Callable, Optional, Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

@dataclass
class ProfileStats:
    function_name: str
    call_count: int
    total_time: float
    cumulative_time: float
    per_call_time: float
    file_name: str
    line_number: int

class Profiler:   
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.enabled = False
        self.stats: List[ProfileStats] = []
    
    def start(self):
        self.profiler.enable()
        self.enabled = True
    
    def stop(self):
        self.profiler.disable()
        self.enabled = False
    
    def reset(self):
        self.profiler = cProfile.Profile()
        self.stats = []
    
    def get_stats(self, sort_by: str = 'cumulative', limit: int = 20) -> List[ProfileStats]:
        try:
            stream = io.StringIO()
            stats = pstats.Stats(self.profiler, stream=stream)
            stats.sort_stats(sort_by)
            stats.print_stats(limit)
        except TypeError:
            return []
        
        self.stats = self._parse_stats(stats)
        return self.stats[:limit]
    
    def _parse_stats(self, stats: pstats.Stats) -> List[ProfileStats]:
        result = []
        
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_num, func_name = func
            
            stat = ProfileStats(
                function_name=func_name,
                call_count=cc,
                total_time=tt,
                cumulative_time=ct,
                per_call_time=tt / cc if cc > 0 else 0.0,
                file_name=filename,
                line_number=line_num
            )
            result.append(stat)
        
        return result
    
    def get_hotspots(self, threshold: float = 0.1) -> List[ProfileStats]:
        stats = self.get_stats()
        if not stats:
            return []
        
        total_time = max(s.cumulative_time for s in stats)
        threshold_time = total_time * threshold
        
        return [s for s in stats if s.cumulative_time >= threshold_time]
    
    def save_stats(self, filename: str):
        if not self.enabled:
            return
        
        stats = pstats.Stats(self.profiler)
        stats.dump_stats(filename)
    
    def load_stats(self, filename: str):
        self.profiler = cProfile.Profile()
        self.profiler.load_stats(filename)


class FunctionProfiler:
    _registry: Dict[str, Dict] = defaultdict(lambda: {
        'call_count': 0,
        'total_time': 0.0,
        'min_time': float('inf'),
        'max_time': 0.0,
        'times': []
    })
    
    def __init__(self, func: Callable):
        self.func = func
        self.func_name = f"{func.__module__}.{func.__name__}"
        wraps(func)(self)
    
    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = self.func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            
            stats = self._registry[self.func_name]
            stats['call_count'] += 1
            stats['total_time'] += elapsed
            stats['min_time'] = min(stats['min_time'], elapsed)
            stats['max_time'] = max(stats['max_time'], elapsed)
            stats['times'].append(elapsed)
    
    @classmethod
    def get_stats(cls, func_name: Optional[str] = None) -> Dict:
        if func_name:
            return cls._registry.get(func_name, {})
        return dict(cls._registry)
    
    @classmethod
    def get_summary(cls) -> Dict[str, Dict]:
        summary = {}
        for func_name, stats in cls._registry.items():
            times = stats['times']
            summary[func_name] = {
                'call_count': stats['call_count'],
                'total_time': stats['total_time'],
                'avg_time': statistics.mean(times) if times else 0.0,
                'min_time': stats['min_time'] if stats['min_time'] != float('inf') else 0.0,
                'max_time': stats['max_time'],
                'median_time': statistics.median(times) if times else 0.0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0.0
            }
        return summary
    
    @classmethod
    def reset(cls):
        cls._registry.clear()

def profile_function(func: Callable) -> Callable:
    return FunctionProfiler(func)

class PerformanceBenchmark:
    def __init__(self, warmup_iterations: int = 3):
        self.warmup_iterations = warmup_iterations
        self.results: Dict[str, List[float]] = {}
    
    def benchmark(
        self,
        func: Callable,
        name: str,
        iterations: int = 100,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        self.results[name] = times
        
        return {
            'name': name,
            'iterations': iterations,
            'total_time': sum(times),
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'throughput': iterations / sum(times) if sum(times) > 0 else 0.0
        }
    
    def compare(self, *benchmark_names: str) -> Dict:
        comparison = {}
        for name in benchmark_names:
            if name in self.results:
                times = self.results[name]
                comparison[name] = {
                    'avg': statistics.mean(times),
                    'min': min(times),
                    'max': max(times),
                    'median': statistics.median(times)
                }
        return comparison
    
    def get_results(self) -> Dict[str, List[float]]:
        return self.results.copy()
    
    def reset(self):
        self.results.clear()

