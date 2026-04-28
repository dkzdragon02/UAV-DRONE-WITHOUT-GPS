import pytest
import time
from uav_vision.profiler import (
    Profiler,
    FunctionProfiler,
    profile_function,
    PerformanceBenchmark
)


class TestProfiler:
    def test_profiler_start_stop(self):
        profiler = Profiler()

        assert profiler.enabled is False
        profiler.start()
        assert profiler.enabled is True
        profiler.stop()
        assert profiler.enabled is False
    
    def test_profiler_reset(self):
        profiler = Profiler()
        profiler.start()
        profiler.stop()
        profiler.reset()
        
        assert profiler.enabled is False
        assert len(profiler.stats) == 0
    
    def test_profiler_basic(self):
        profiler = Profiler()
        profiler.start()
        
        def test_func():
            time.sleep(0.01)
            return sum(range(100))
        
        result = test_func()
        profiler.stop()
        
        assert result == 4950
        stats = profiler.get_stats(limit=10)
        assert len(stats) > 0


class TestFunctionProfiler:
    def test_function_profiler_decorator(self):
        FunctionProfiler.reset()
        
        @profile_function
        def test_func(x):
            time.sleep(0.01)
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        all_stats = FunctionProfiler.get_stats()
        matching = [v for k, v in all_stats.items() if 'test_func' in k]
        assert len(matching) == 1
        assert matching[0]['call_count'] == 1
        assert matching[0]['total_time'] > 0
    
    def test_function_profiler_multiple_calls(self):
        FunctionProfiler.reset()
        
        @profile_function
        def test_func(x):
            return x * 2
        
        test_func(1)
        test_func(2)
        test_func(3)
        
        all_stats = FunctionProfiler.get_stats()
        matching = [v for k, v in all_stats.items() if 'test_func' in k]
        assert len(matching) == 1
        assert matching[0]['call_count'] == 3
    
    def test_function_profiler_summary(self):
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
    def test_benchmark_basic(self):
        benchmark = PerformanceBenchmark()
        
        def test_func():
            return sum(range(100))
        
        result = benchmark.benchmark(test_func, 'sum_test', iterations=10)
        
        assert result['name'] == 'sum_test'
        assert result['iterations'] == 10
        assert result['avg_time'] > 0
        assert result['throughput'] > 0
    
    def test_benchmark_compare(self):
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
        benchmark = PerformanceBenchmark()
        
        def test_func():
            return 1
        
        benchmark.benchmark(test_func, 'test', iterations=5)
        assert len(benchmark.get_results()) == 1
        
        benchmark.reset()
        assert len(benchmark.get_results()) == 0

