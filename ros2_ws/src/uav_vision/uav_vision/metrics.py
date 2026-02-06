#!/usr/bin/env python3
"""
Metrics Collection System
Collects and publishes system metrics
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric definition"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


class MetricsCollector:
    """
    Metrics collector for system metrics.
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
            labels: Optional labels
        """
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
        """
        Set a gauge metric.
        
        Args:
            name: Gauge name
            value: Gauge value
            labels: Optional labels
        """
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
        """
        Record a histogram value.
        
        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create key from name and labels"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def _record_metric(self, metric: Metric):
        """Record a metric"""
        key = self._make_key(metric.name, metric.labels)
        self.metrics[key].append(metric)
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value"""
        key = self._make_key(name, labels)
        return self.gauges.get(key)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get histogram statistics.
        
        Args:
            name: Histogram name
            labels: Optional labels
            
        Returns:
            Statistics dictionary or None
        """
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
        """
        Get all metrics.
        
        Returns:
            Dictionary with all metrics
        """
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
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted string
        """
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


# Global metrics collector instance
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _global_collector

