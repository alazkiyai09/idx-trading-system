"""
Retry Handler Module

Provides retry logic with exponential backoff and circuit breaker
pattern for external LLM API calls.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

from llm.base_client import LLMAPIError, LLMRateLimitError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for exponential backoff.
        retry_on: Exception types to retry on.
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retry_on: tuple = (LLMAPIError, LLMRateLimitError, ConnectionError, TimeoutError)


@dataclass
class CircuitBreakerState:
    """State of the circuit breaker.

    Attributes:
        failure_count: Number of consecutive failures.
        last_failure_time: Time of the last failure.
        is_open: Whether the circuit is open (blocking calls).
        failure_threshold: Number of failures before opening.
        recovery_timeout: Seconds to wait before trying again.
    """
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    failure_threshold: int = 5
    recovery_timeout: float = 300.0  # 5 minutes

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures. "
                f"Will retry after {self.recovery_timeout}s."
            )

    def record_success(self) -> None:
        """Record a success and close the circuit."""
        if self.is_open:
            logger.info("Circuit breaker CLOSED after successful call.")
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

    def can_proceed(self) -> bool:
        """Check if a call can proceed.

        Returns:
            True if the circuit is closed or recovery timeout has elapsed.
        """
        if not self.is_open:
            return True

        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        if elapsed >= self.recovery_timeout:
            logger.info("Circuit breaker attempting recovery (half-open state).")
            return True

        return False


class RetryHandler:
    """Handles retry logic with exponential backoff and circuit breaker.

    Example:
        handler = RetryHandler()

        @handler.with_retry
        def call_api():
            return client.generate("prompt")

        result = call_api()
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        enable_circuit_breaker: bool = True,
    ) -> None:
        """Initialize retry handler.

        Args:
            config: Retry configuration.
            enable_circuit_breaker: Whether to use circuit breaker.
        """
        self.config = config or RetryConfig()
        self._circuit_breaker = CircuitBreakerState() if enable_circuit_breaker else None

    def with_retry(self, func: F) -> F:
        """Decorator that adds retry logic to a function.

        Args:
            func: Function to wrap with retry logic.

        Returns:
            Wrapped function with retry behavior.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.execute(func, *args, **kwargs)
        return wrapper  # type: ignore

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            LLMAPIError: If all retries are exhausted.
        """
        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            raise LLMAPIError(
                "Circuit breaker is OPEN. LLM service appears unavailable. "
                f"Will retry after {self._circuit_breaker.recovery_timeout}s."
            )

        last_exception: Optional[Exception] = None
        delay = self.config.base_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return result

            except self.config.retry_on as e:
                last_exception = e

                if attempt == self.config.max_retries:
                    logger.error(
                        f"All {self.config.max_retries} retries exhausted for "
                        f"{func.__name__}: {e}"
                    )
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    break

                # Special handling for rate limits
                if isinstance(e, LLMRateLimitError):
                    delay = min(delay * self.config.backoff_factor * 2, self.config.max_delay)
                    logger.warning(
                        f"Rate limit hit. Waiting {delay:.1f}s before retry "
                        f"(attempt {attempt + 1}/{self.config.max_retries})"
                    )
                else:
                    logger.warning(
                        f"Retry {attempt + 1}/{self.config.max_retries} for "
                        f"{func.__name__} after {delay:.1f}s: {e}"
                    )

                time.sleep(delay)
                delay = min(delay * self.config.backoff_factor, self.config.max_delay)

            except Exception as e:
                # Non-retryable exception
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise

        raise LLMAPIError(
            f"Failed after {self.config.max_retries} retries: {last_exception}"
        )

    def reset(self) -> None:
        """Reset the circuit breaker state."""
        if self._circuit_breaker:
            self._circuit_breaker.record_success()
