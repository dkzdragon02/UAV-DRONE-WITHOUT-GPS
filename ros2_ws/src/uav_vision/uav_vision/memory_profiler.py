#!/usr/bin/env python3
"""
Memory Profiler
Provides memory profiling and leak detection
"""

import sys
import tracemalloc
import gc
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class MemorySnapshot:
    """Memory snapshot"""
    timestamp: float
    current_memory: int
    peak_memory: int
    top_stats: List[tuple]


class MemoryProfiler:
    """
    Memory profiler for tracking memory usage and detecting leaks.
    """
    
    def __init__(self):
        """Initialize memory profiler"""
        self.tracemalloc_enabled = False
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[tracemalloc.Snapshot] = None
    
    def start(self):
        """Start memory tracking"""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_enabled = True
    
    def stop(self):
        """Stop memory tracking"""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
    
    def take_snapshot(self, top_n: int = 10) -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            top_n: Number of top allocations to track
            
        Returns:
            Memory snapshot
        """
        if not self.tracemalloc_enabled:
            self.start()
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:top_n]
        
        current = sum(stat.size for stat in top_stats)
        peak = tracemalloc.get_traced_memory()[1]
        
        mem_snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_memory=current,
            peak_memory=peak,
            top_stats=[(stat.traceback.format(), stat.size, stat.count) for stat in top_stats]
        )
        
        self.snapshots.append(mem_snapshot)
        return mem_snapshot
    
    def set_baseline(self):
        """Set baseline snapshot for comparison"""
        if not self.tracemalloc_enabled:
            self.start()
        self.baseline_snapshot = tracemalloc.take_snapshot()
    
    def compare_with_baseline(self, top_n: int = 10) -> Dict:
        """
        Compare current memory with baseline.
        
        Args:
            top_n: Number of top differences to show
            
        Returns:
            Comparison dictionary
        """
        if not self.baseline_snapshot:
            return {}
        
        current_snapshot = tracemalloc.take_snapshot()
        top_stats = current_snapshot.compare_to(
            self.baseline_snapshot,
            'lineno'
        )[:top_n]
        
        return {
            'differences': [
                {
                    'file': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                    'size_diff': stat.size_diff,
                    'size': stat.size,
                    'count_diff': stat.count_diff,
                    'count': stat.count
                }
                for stat in top_stats
            ]
        }
    
    def detect_leaks(self, threshold_mb: float = 10.0) -> bool:
        """
        Detect potential memory leaks.
        
        Args:
            threshold_mb: Memory increase threshold in MB
            
        Returns:
            True if leak detected, False otherwise
        """
        if len(self.snapshots) < 2:
            return False
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        memory_increase = (last.current_memory - first.current_memory) / (1024 * 1024)
        
        return memory_increase > threshold_mb
    
    def get_memory_stats(self) -> Dict:
        """
        Get current memory statistics.
        
        Returns:
            Memory statistics dictionary
        """
        if not self.tracemalloc_enabled:
            return {}
        
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            'current_mb': current / (1024 * 1024),
            'peak_mb': peak / (1024 * 1024),
            'snapshots_count': len(self.snapshots)
        }
    
    def get_gc_stats(self) -> Dict:
        """
        Get garbage collector statistics.
        
        Returns:
            GC statistics dictionary
        """
        return {
            'collections': gc.get_count(),
            'thresholds': gc.get_threshold(),
            'stats': gc.get_stats()
        }
    
    def force_gc(self):
        """Force garbage collection"""
        collected = gc.collect()
        return collected
    
    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict:
        """
        Profile memory usage of a function.
        
        Args:
            func: Function to profile
            *args, **kwargs: Function arguments
            
        Returns:
            Memory profile dictionary
        """
        self.start()
        self.set_baseline()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Take snapshot
        snapshot = self.take_snapshot()
        comparison = self.compare_with_baseline()
        
        return {
            'result': result,
            'snapshot': snapshot,
            'comparison': comparison,
            'stats': self.get_memory_stats()
        }


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage (simple version without tracemalloc).
    
    Returns:
        Memory usage dictionary
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {}


class MemoryMonitor:
    """
    Continuous memory monitoring.
    Tracks memory usage over time and detects anomalies.
    """
    
    def __init__(self, check_interval: float = 1.0):
        """
        Initialize memory monitor.
        
        Args:
            check_interval: Check interval in seconds
        """
        self.check_interval = check_interval
        self.memory_history: List[Dict] = []
        self.max_history = 1000
        self.monitoring = False
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring = True
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
    
    def check_memory(self) -> Dict:
        """
        Check current memory usage.
        
        Returns:
            Memory usage dictionary
        """
        usage = get_memory_usage()
        usage['timestamp'] = time.time()
        
        self.memory_history.append(usage)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
        
        return usage
    
    def get_memory_trend(self) -> Dict:
        """
        Get memory usage trend.
        
        Returns:
            Trend statistics
        """
        if len(self.memory_history) < 2:
            return {}
        
        rss_values = [m.get('rss_mb', 0) for m in self.memory_history]
        
        return {
            'current_mb': rss_values[-1] if rss_values else 0,
            'min_mb': min(rss_values) if rss_values else 0,
            'max_mb': max(rss_values) if rss_values else 0,
            'avg_mb': sum(rss_values) / len(rss_values) if rss_values else 0,
            'trend': 'increasing' if rss_values[-1] > rss_values[0] else 'decreasing'
        }
    
    def detect_anomaly(self, threshold_percent: float = 20.0) -> bool:
        """
        Detect memory anomaly (sudden increase).
        
        Args:
            threshold_percent: Percentage increase threshold
            
        Returns:
            True if anomaly detected
        """
        if len(self.memory_history) < 2:
            return False
        
        recent = self.memory_history[-10:] if len(self.memory_history) >= 10 else self.memory_history
        older = self.memory_history[:len(self.memory_history) - len(recent)]
        
        if not older:
            return False
        
        recent_avg = sum(m.get('rss_mb', 0) for m in recent) / len(recent)
        older_avg = sum(m.get('rss_mb', 0) for m in older) / len(older)
        
        if older_avg == 0:
            return False
        
        increase_percent = ((recent_avg - older_avg) / older_avg) * 100
        
        return increase_percent > threshold_percent

