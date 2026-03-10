"""
Health Monitor Module

Monitors system health and reports issues.
Checks data freshness, database integrity, API connectivity, and disk space.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health of a single component.

    Attributes:
        name: Component name.
        status: Health status.
        message: Status message.
        last_checked: When this was last checked.
        details: Additional details.
    """
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    last_checked: Optional[datetime] = None
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health.

    Attributes:
        status: Overall status (worst component status).
        components: Individual component health.
        timestamp: When the check was performed.
    """
    status: HealthStatus = HealthStatus.UNKNOWN
    components: List[ComponentHealth] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """Monitors system health and reports issues.

    Example:
        monitor = HealthMonitor()
        health = monitor.check_all()
        if health.status == HealthStatus.CRITICAL:
            notify_admin(health)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        db_path: Optional[str] = None,
    ) -> None:
        """Initialize health monitor.

        Args:
            data_dir: Data directory to monitor.
            db_path: Database path to check.
        """
        self._data_dir = data_dir or Path("data")
        self._db_path = db_path

    def check_data_freshness(self) -> ComponentHealth:
        """Check if market data is up-to-date.

        Returns:
            ComponentHealth for data freshness.
        """
        health = ComponentHealth(
            name="data_freshness",
            last_checked=datetime.now(),
        )

        try:
            market_dir = self._data_dir / "market"
            if not market_dir.exists():
                health.status = HealthStatus.WARNING
                health.message = "Market data directory does not exist"
                return health

            # Check for recent data files
            data_files = list(market_dir.glob("*.db")) + list(market_dir.glob("*.csv"))
            if not data_files:
                health.status = HealthStatus.WARNING
                health.message = "No market data files found"
                return health

            health.status = HealthStatus.HEALTHY
            health.message = f"Found {len(data_files)} data file(s)"

        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.message = f"Failed to check data freshness: {e}"

        return health

    def check_database_integrity(self) -> ComponentHealth:
        """Check database is accessible and consistent.

        Returns:
            ComponentHealth for database.
        """
        health = ComponentHealth(
            name="database",
            last_checked=datetime.now(),
        )

        try:
            from core.data.database import DatabaseManager
            db = DatabaseManager()
            health.status = HealthStatus.HEALTHY
            health.message = "Database accessible"
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.message = f"Database error: {e}"

        return health

    def check_disk_space(self, min_free_gb: float = 1.0) -> ComponentHealth:
        """Check available disk space.

        Args:
            min_free_gb: Minimum free space in GB.

        Returns:
            ComponentHealth for disk space.
        """
        health = ComponentHealth(
            name="disk_space",
            last_checked=datetime.now(),
        )

        try:
            stat = os.statvfs(str(self._data_dir))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)

            health.details["free_gb"] = f"{free_gb:.2f}"
            health.details["total_gb"] = f"{total_gb:.2f}"
            health.details["usage_pct"] = f"{(1 - free_gb / total_gb) * 100:.1f}%"

            if free_gb < min_free_gb:
                health.status = HealthStatus.CRITICAL
                health.message = f"Low disk space: {free_gb:.2f} GB free"
            elif free_gb < min_free_gb * 2:
                health.status = HealthStatus.WARNING
                health.message = f"Disk space getting low: {free_gb:.2f} GB free"
            else:
                health.status = HealthStatus.HEALTHY
                health.message = f"Disk space OK: {free_gb:.2f} GB free"

        except Exception as e:
            health.status = HealthStatus.WARNING
            health.message = f"Could not check disk space: {e}"

        return health

    def check_api_connectivity(self) -> ComponentHealth:
        """Check LLM API connectivity.

        Returns:
            ComponentHealth for API connectivity.
        """
        health = ComponentHealth(
            name="llm_api",
            last_checked=datetime.now(),
        )

        try:
            from config.settings import settings
            if settings.anthropic_api_key:
                health.status = HealthStatus.HEALTHY
                health.message = "API key configured"
            else:
                health.status = HealthStatus.WARNING
                health.message = "No API key configured"
        except Exception as e:
            health.status = HealthStatus.WARNING
            health.message = f"Could not check API: {e}"

        return health

    def check_all(self) -> SystemHealth:
        """Run all health checks.

        Returns:
            SystemHealth with all component statuses.
        """
        components = [
            self.check_data_freshness(),
            self.check_database_integrity(),
            self.check_disk_space(),
            self.check_api_connectivity(),
        ]

        # Overall status is the worst component status
        status_priority = {
            HealthStatus.CRITICAL: 0,
            HealthStatus.WARNING: 1,
            HealthStatus.UNKNOWN: 2,
            HealthStatus.HEALTHY: 3,
        }

        worst = min(components, key=lambda c: status_priority.get(c.status, 2))

        return SystemHealth(
            status=worst.status,
            components=components,
            timestamp=datetime.now(),
        )
