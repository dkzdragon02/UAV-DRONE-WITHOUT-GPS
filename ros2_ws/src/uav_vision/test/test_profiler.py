"""
Unit tests for profiler
"""

import pytest
import time
from uav_vision.profiler import (
    Profiler,
    FunctionProfiler,
    profile_function,
    PerformanceBenchmark
)


class TestProfiler:
    """Test suite for Profiler"""
    
    def test_profiler_start_stop(self):
        """Test starting and stopping profiler"""
        profiler = Profiler()
        
        assert profiler.enabled is False
        profiler.start()
        assert profiler.enabled is True
        profiler.stop()
        assert profiler.enabled is False
    
    def test_profiler_reset(self):
        """Test resetting profiler"""
        profiler = Profiler()
        profiler.start()
        profiler.stop()
        profiler.reset()
        
        assert profiler.enabled is False
        assert len(profiler.stats) == 0
    
    def test_profiler_basic(self):
        """Test basic profiling"""
        profiler = Profiler()
        profiler.start()
        
        # Do some work
        def test_func():
            time.sleep(0.01)
            return sum(range(100))
        
        result = test_func()
        profiler.stop()
        
        assert result == 4950
        stats = profiler.get_stats(limit=10)
        assert len(stats) > 0


class TestFunctionProfiler:
    """Test suite for FunctionProfiler"""
    
    def test_function_profiler_decorator(self):
        """Test function profiler as decorator"""
        FunctionProfiler.reset()
        
        @profile_function
        def test_func(x):
            time.sleep(0.01)
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        stats = FunctionProfiler.get_stats('test_profiler.test_func')
        assert stats['call_count'] == 1
        assert stats['total_time'] > 0
    
    def test_function_profiler_multiple_calls(self):
        """Test profiling multiple function calls"""
        FunctionProfiler.reset()
        
        @profile_function
        def test_func(x):
            return x * 2
        
        test_func(1)
        test_func(2)
        test_func(3)
        
        stats = FunctionProfiler.get_stats('test_profiler.test_func')
        assert stats['call_count'] == 3
    
    def test_function_profiler_summary(self):
        """Test getting summary"""
        FunctionProfiler.reset()
        
        @profile_function
        def func1():
            return 1
        
        @profile_function
        def func2():
            return 2
        
        func1()
        func2()
        
        summary = FunctionProfiler.get_summary()
        assert len(summary) >= 2


class TestPerformanceBenchmark:
    """Test suite for PerformanceBenchmark"""
    
    def test_benchmark_basic(self):
        """Test basic benchmarking"""
        benchmark = PerformanceBenchmark()
        
        def test_func():
            return sum(range(100))
        
        result = benchmark.benchmark(test_func, 'sum_test', iterations=10)
        
        assert result['name'] == 'sum_test'
        assert result['iterations'] == 10
        assert result['avg_time'] > 0
        assert result['throughput'] > 0
    
    def test_benchmark_compare(self):
        """Test comparing benchmarks"""
        benchmark = PerformanceBenchmark()
        
        def fast_func():
            return 1
        
        def slow_func():
            time.sleep(0.001)
            return 2
        
        benchmark.benchmark(fast_func, 'fast', iterations=10)
        benchmark.benchmark(slow_func, 'slow', iterations=10)
        
        comparison = benchmark.compare('fast', 'slow')
        assert 'fast' in comparison
        assert 'slow' in comparison
        assert comparison['fast']['avg'] < comparison['slow']['avg']
    
    def test_benchmark_reset(self):
        """Test resetting benchmark"""
        benchmark = PerformanceBenchmark()
        
        def test_func():
            return 1
        
        benchmark.benchmark(test_func, 'test', iterations=5)
        assert len(benchmark.get_results()) == 1
        
        benchmark.reset()
        assert len(benchmark.get_results()) == 0

