import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: str = ""

class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            self._record_metric(Metric(
                name=name,
                value=self.counters[key],
                metric_type=MetricType.COUNTER,
                labels=labels or {}
            ))
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            self._record_metric(Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                labels=labels or {}
            ))
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def _record_metric(self, metric: Metric):
        key = self._make_key(metric.name, metric.labels)
        self.metrics[key].append(metric)
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        key = self._make_key(name, labels)
        return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        key = self._make_key(name, labels)
        return self.gauges.get(key)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, float]]:
        key = self._make_key(name, labels)
        values = self.histograms.get(key)
        
        if not values:
            return None
        
        import statistics
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {
                    k: self.get_histogram_stats(k.split('{')[0])
                    for k in self.histograms.keys()
                }
            }
    
    def export_prometheus_format(self) -> str:
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # Histograms
        for key in self.histograms.keys():
            name = key.split('{')[0]
            stats = self.get_histogram_stats(name)
            if stats:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['mean'] * stats['count']}")
                lines.append(f"{name}_mean {stats['mean']}")
        
        return "\n".join(lines)

_global_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    return _global_collector

