"""
Cost Tracker Module

Tracks token usage and costs per LLM API call,
with daily budgets and warnings.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from llm.base_client import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """Record of a single LLM API call.

    Attributes:
        timestamp: When the call was made.
        provider: LLM provider used.
        model: Model name.
        input_tokens: Input tokens used.
        output_tokens: Output tokens generated.
        cost_usd: Cost in USD.
        purpose: Description of what the call was for.
    """
    timestamp: datetime
    provider: LLMProvider
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    purpose: str = ""


@dataclass
class DailySummary:
    """Daily usage summary.

    Attributes:
        date: Date of the summary.
        total_calls: Total number of API calls.
        total_input_tokens: Total input tokens.
        total_output_tokens: Total output tokens.
        total_cost_usd: Total cost in USD.
        by_provider: Cost breakdown by provider.
        by_model: Cost breakdown by model.
    """
    date: date
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    by_provider: Dict[str, float] = field(default_factory=dict)
    by_model: Dict[str, float] = field(default_factory=dict)


class CostTracker:
    """Tracks LLM API costs and enforces budgets.

    Example:
        tracker = CostTracker(daily_budget=5.0)
        tracker.record(response, purpose="fundamental analysis")
        print(tracker.get_daily_summary())
    """

    def __init__(
        self,
        daily_budget: float = 10.0,
        warn_threshold: float = 0.8,
    ) -> None:
        """Initialize cost tracker.

        Args:
            daily_budget: Daily budget in USD.
            warn_threshold: Fraction of budget that triggers a warning (0-1).
        """
        self.daily_budget = daily_budget
        self.warn_threshold = warn_threshold
        self._records: List[CallRecord] = []
        self._warned_today = False

    def record(
        self,
        response: LLMResponse,
        provider: LLMProvider = LLMProvider.CLAUDE,
        purpose: str = "",
    ) -> None:
        """Record an API call.

        Args:
            response: LLM response to record.
            provider: LLM provider that was used.
            purpose: What the call was for (e.g., "auditor_agent").
        """
        record = CallRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            purpose=purpose,
        )
        self._records.append(record)

        logger.debug(
            f"LLM call: {provider.value}/{response.model} "
            f"tokens={response.total_tokens} cost=${response.cost_usd:.4f} "
            f"purpose={purpose}"
        )

        # Check budget
        daily_cost = self.get_daily_cost()
        if daily_cost >= self.daily_budget:
            logger.error(
                f"Daily LLM budget EXCEEDED: ${daily_cost:.2f} / ${self.daily_budget:.2f}"
            )
        elif daily_cost >= self.daily_budget * self.warn_threshold and not self._warned_today:
            logger.warning(
                f"Daily LLM budget at {daily_cost/self.daily_budget*100:.0f}%: "
                f"${daily_cost:.2f} / ${self.daily_budget:.2f}"
            )
            self._warned_today = True

    def get_daily_cost(self, target_date: Optional[date] = None) -> float:
        """Get total cost for a specific date.

        Args:
            target_date: Date to check. Defaults to today.

        Returns:
            Total cost in USD.
        """
        target = target_date or date.today()
        return sum(
            r.cost_usd
            for r in self._records
            if r.timestamp.date() == target
        )

    def get_daily_summary(self, target_date: Optional[date] = None) -> DailySummary:
        """Get detailed daily summary.

        Args:
            target_date: Date to summarize. Defaults to today.

        Returns:
            DailySummary with breakdowns.
        """
        target = target_date or date.today()
        today_records = [r for r in self._records if r.timestamp.date() == target]

        summary = DailySummary(date=target)
        for r in today_records:
            summary.total_calls += 1
            summary.total_input_tokens += r.input_tokens
            summary.total_output_tokens += r.output_tokens
            summary.total_cost_usd += r.cost_usd

            provider_name = r.provider.value
            summary.by_provider[provider_name] = (
                summary.by_provider.get(provider_name, 0.0) + r.cost_usd
            )
            summary.by_model[r.model] = (
                summary.by_model.get(r.model, 0.0) + r.cost_usd
            )

        return summary

    def is_within_budget(self) -> bool:
        """Check if today's spending is within budget.

        Returns:
            True if within budget.
        """
        return self.get_daily_cost() < self.daily_budget

    def get_remaining_budget(self) -> float:
        """Get remaining budget for today.

        Returns:
            Remaining budget in USD.
        """
        return max(0.0, self.daily_budget - self.get_daily_cost())

    def reset_daily(self) -> None:
        """Reset the daily warning flag. Called at start of new day."""
        self._warned_today = False

    def get_all_records(self) -> List[CallRecord]:
        """Get all recorded API calls.

        Returns:
            List of CallRecord objects.
        """
        return list(self._records)
