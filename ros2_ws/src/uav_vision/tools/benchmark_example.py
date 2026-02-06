#!/usr/bin/env python3
"""
Example benchmark script
Demonstrates how to use performance profiling tools
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from uav_vision.profiler import PerformanceBenchmark, profile_function
from uav_vision.memory_profiler import MemoryProfiler, get_memory_usage
from uav_vision.performance_optimizer import PerformanceOptimizer
import numpy as np
import time


@profile_function
def process_image_slow(image):
    """Slow image processing"""
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = image[i, j] * 2
    return result


@profile_function
def process_image_fast(image):
    """Fast image processing using vectorization"""
    return image * 2


def benchmark_image_processing():
    """Benchmark image processing functions"""
    print("=== Image Processing Benchmark ===\n")
    
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    benchmark = PerformanceBenchmark()
    
    # Benchmark slow version
    print("Benchmarking slow version...")
    slow_result = benchmark.benchmark(
        process_image_slow,
        'slow_processing',
        iterations=5,
        image=test_image
    )
    print(f"  Average time: {slow_result['avg_time']*1000:.2f} ms")
    print(f"  Throughput: {slow_result['throughput']:.2f} ops/s\n")
    
    # Benchmark fast version
    print("Benchmarking fast version...")
    fast_result = benchmark.benchmark(
        process_image_fast,
        'fast_processing',
        iterations=100,
        image=test_image
    )
    print(f"  Average time: {fast_result['avg_time']*1000:.2f} ms")
    print(f"  Throughput: {fast_result['throughput']:.2f} ops/s\n")
    
    # Compare
    comparison = benchmark.compare('slow_processing', 'fast_processing')
    speedup = comparison['slow_processing']['avg'] / comparison['fast_processing']['avg']
    print(f"Speedup: {speedup:.2f}x faster\n")
    
    # Function profiling summary
    from uav_vision.profiler import FunctionProfiler
    summary = FunctionProfiler.get_summary()
    print("Function Profiling Summary:")
    for func_name, stats in summary.items():
        print(f"  {func_name}:")
        print(f"    Calls: {stats['call_count']}")
        print(f"    Avg time: {stats['avg_time']*1000:.2f} ms")


def memory_profiling_example():
    """Example of memory profiling"""
    print("\n=== Memory Profiling Example ===\n")
    
    profiler = MemoryProfiler()
    profiler.start()
    profiler.set_baseline()
    
    # Allocate some memory
    print("Allocating memory...")
    large_array = np.random.rand(1000, 1000)
    time.sleep(0.1)
    
    # Take snapshot
    snapshot = profiler.take_snapshot()
    print(f"Current memory: {snapshot.current_memory / (1024*1024):.2f} MB")
    print(f"Peak memory: {snapshot.peak_memory / (1024*1024):.2f} MB")
    
    # Compare with baseline
    comparison = profiler.compare_with_baseline()
    if comparison.get('differences'):
        print("\nTop memory allocations:")
        for diff in comparison['differences'][:5]:
            print(f"  {diff['size'] / 1024:.2f} KB")
    
    # Get stats
    stats = profiler.get_memory_stats()
    print(f"\nMemory stats: {stats}")
    
    profiler.stop()


def optimization_recommendations():
    """Example of optimization recommendations"""
    print("\n=== Optimization Recommendations ===\n")
    
    optimizer = PerformanceOptimizer()
    
    def slow_function():
        """Slow function for analysis"""
        result = []
        for i in range(10000):
            result.append(i * 2)
        return result
    
    recommendations = optimizer.analyze_function(slow_function)
    
    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(f"  [{rec.priority.upper()}] {rec.issue}")
            print(f"    → {rec.recommendation}")
            print(f"    Estimated improvement: {rec.estimated_improvement}\n")


if __name__ == '__main__':
    print("Performance Benchmarking Examples\n")
    print("=" * 50 + "\n")
    
    # Run benchmarks
    benchmark_image_processing()
    
    # Memory profiling
    memory_profiling_example()
    
    # Optimization recommendations
    optimization_recommendations()
    
    print("\n" + "=" * 50)
    print("Benchmarking complete!")

