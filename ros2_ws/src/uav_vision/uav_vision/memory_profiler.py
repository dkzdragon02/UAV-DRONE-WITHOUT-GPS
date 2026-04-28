import sys
import tracemalloc
import gc
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class MemorySnapshot:
    timestamp: float
    current_memory: int
    peak_memory: int
    top_stats: List[tuple]

class MemoryProfiler:
    def __init__(self):
        self.tracemalloc_enabled = False
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[tracemalloc.Snapshot] = None
    
    def start(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_enabled = True
    
    def stop(self):
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
    
    def take_snapshot(self, top_n: int = 10) -> MemorySnapshot:
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
        if not self.tracemalloc_enabled:
            self.start()
        self.baseline_snapshot = tracemalloc.take_snapshot()
    
    def compare_with_baseline(self, top_n: int = 10) -> Dict:
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
        if len(self.snapshots) < 2:
            return False
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        memory_increase = (last.current_memory - first.current_memory) / (1024 * 1024)
        
        return memory_increase > threshold_mb
    
    def get_memory_stats(self) -> Dict:
        if not self.tracemalloc_enabled:
            return {}
        
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            'current_mb': current / (1024 * 1024),
            'peak_mb': peak / (1024 * 1024),
            'snapshots_count': len(self.snapshots)
        }
    
    def get_gc_stats(self) -> Dict:
        return {
            'collections': gc.get_count(),
            'thresholds': gc.get_threshold(),
            'stats': gc.get_stats()
        }
    
    def force_gc(self):
        collected = gc.collect()
        return collected
    
    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict:

        self.start()
        self.set_baseline()
        
        result = func(*args, **kwargs)
        snapshot = self.take_snapshot()
        comparison = self.compare_with_baseline()
        
        return {
            'result': result,
            'snapshot': snapshot,
            'comparison': comparison,
            'stats': self.get_memory_stats()
        }

def get_memory_usage() -> Dict[str, float]:
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
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.memory_history: List[Dict] = []
        self.max_history = 1000
        self.monitoring = False
    
    def start_monitoring(self):
        self.monitoring = True
    
    def stop_monitoring(self):
        self.monitoring = False
    
    def check_memory(self) -> Dict:
        usage = get_memory_usage()
        usage['timestamp'] = time.time()
        
        self.memory_history.append(usage)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
        
        return usage
    
    def get_memory_trend(self) -> Dict:
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

