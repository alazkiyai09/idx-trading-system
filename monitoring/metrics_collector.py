"""
Metrics Collector Module

Collects rolling system metrics for monitoring and alerting.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    label: str = ""


class MetricsCollector:
    """Collects and stores rolling system metrics.

    Tracks error rates, API latency, signal success rates,
    and other operational metrics.

    Example:
        metrics = MetricsCollector()
        metrics.record("api_latency_ms", 150.0)
        metrics.record("scan_success", 1.0)
        print(metrics.get_average("api_latency_ms", hours=1))
    """

    def __init__(self, max_history_hours: int = 168) -> None:
        """Initialize metrics collector.

        Args:
            max_history_hours: Maximum history to retain (default: 7 days).
        """
        self._metrics: Dict[str, Deque[MetricPoint]] = {}
        self._max_age = timedelta(hours=max_history_hours)

    def record(self, metric_name: str, value: float, label: str = "") -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            label: Optional label for categorization.
        """
        if metric_name not in self._metrics:
            self._metrics[metric_name] = deque()

        self._metrics[metric_name].append(MetricPoint(
            timestamp=datetime.now(),
            value=value,
            label=label,
        ))

        self._cleanup(metric_name)

    def _cleanup(self, metric_name: str) -> None:
        """Remove expired data points."""
        cutoff = datetime.now() - self._max_age
        points = self._metrics[metric_name]
        while points and points[0].timestamp < cutoff:
            points.popleft()

    def get_average(
        self,
        metric_name: str,
        hours: int = 24,
    ) -> Optional[float]:
        """Get average value for a metric over a time period.

        Args:
            metric_name: Metric name.
            hours: Number of hours to look back.

        Returns:
            Average value, or None if no data.
        """
        points = self._get_recent(metric_name, hours)
        if not points:
            return None
        return sum(p.value for p in points) / len(points)

    def get_count(
        self,
        metric_name: str,
        hours: int = 24,
    ) -> int:
        """Get count of data points for a metric.

        Args:
            metric_name: Metric name.
            hours: Number of hours to look back.

        Returns:
            Number of data points.
        """
        return len(self._get_recent(metric_name, hours))

    def get_rate(
        self,
        metric_name: str,
        hours: int = 24,
    ) -> Optional[float]:
        """Get success rate (average of 0/1 values).

        Args:
            metric_name: Metric name with 0/1 values.
            hours: Number of hours to look back.

        Returns:
            Success rate (0-1), or None if no data.
        """
        return self.get_average(metric_name, hours)

    def _get_recent(
        self, metric_name: str, hours: int
    ) -> list:
        """Get recent data points."""
        if metric_name not in self._metrics:
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            p for p in self._metrics[metric_name]
            if p.timestamp >= cutoff
        ]

    def get_summary(self, hours: int = 24) -> Dict[str, Dict]:
        """Get summary of all metrics.

        Args:
            hours: Number of hours to look back.

        Returns:
            Dictionary of metric summaries.
        """
        summary = {}
        for name in self._metrics:
            points = self._get_recent(name, hours)
            if points:
                values = [p.value for p in points]
                summary[name] = {
                    "count": len(points),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                }
        return summary
