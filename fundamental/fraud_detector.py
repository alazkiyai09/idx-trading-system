"""
Fraud Detector Module

Detects potential fraud and manipulation in financial statements using:
- Benford's Law analysis
- Balance sheet equation verification
- Cash flow reconciliation
- Round number frequency analysis
- Anomaly detection
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FraudIndicator(Enum):
    """Types of fraud indicators."""
    BENFORD_VIOLATION = "benford_violation"
    BALANCE_MISMATCH = "balance_mismatch"
    CASHFLOW_MISMATCH = "cashflow_mismatch"
    ROUND_NUMBERS = "round_numbers"
    UNUSUAL_PATTERNS = "unusual_patterns"
    REVENUE_MANIPULATION = "revenue_manipulation"
    EXPENSE_ANOMALY = "expense_anomaly"


class RiskLevel(Enum):
    """Risk level for fraud indicators."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FraudCheck:
    """Result of a single fraud check.

    Attributes:
        indicator: Type of fraud indicator.
        risk_level: Risk level.
        score: Numerical score (0-1, higher = more suspicious).
        description: Description of finding.
        details: Additional details.
    """

    indicator: FraudIndicator
    risk_level: RiskLevel
    score: float = 0.0
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FraudAnalysisResult:
    """Result of fraud analysis.

    Attributes:
        overall_risk: Overall fraud risk level.
        overall_score: Overall fraud score (0-1).
        checks: Individual check results.
        fraud_probability: Estimated probability of fraud.
        red_flags: List of red flag descriptions.
        summary: Summary of analysis.
    """

    overall_risk: RiskLevel = RiskLevel.LOW
    overall_score: float = 0.0
    checks: List[FraudCheck] = field(default_factory=list)
    fraud_probability: float = 0.0
    red_flags: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_risk": self.overall_risk.value,
            "overall_score": self.overall_score,
            "fraud_probability": self.fraud_probability,
            "red_flags": self.red_flags,
            "checks": [
                {
                    "indicator": c.indicator.value,
                    "risk_level": c.risk_level.value,
                    "score": c.score,
                    "description": c.description,
                }
                for c in self.checks
            ],
            "summary": self.summary,
        }


class FraudDetector:
    """Detects fraud indicators in financial statements.

    Uses multiple techniques to identify potential manipulation:
    - Benford's Law: First digit distribution analysis
    - Balance sheet: Assets = Liabilities + Equity
    - Cash flow: Operating cash flow vs net income
    - Round numbers: Suspicious frequency of round numbers
    - Pattern analysis: Unusual trends or jumps

    Example:
        detector = FraudDetector()
        result = detector.analyze(financial_data)
        print(f"Fraud probability: {result.fraud_probability:.1%}")
    """

    # Expected Benford's Law distribution for first digits
    BENFORD_DISTRIBUTION = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046,
    }

    def __init__(self, benford_threshold: float = 0.1) -> None:
        """Initialize fraud detector.

        Args:
            benford_threshold: Chi-square threshold for Benford's Law.
        """
        self.benford_threshold = benford_threshold

    def analyze(self, financial_data: Dict[str, Any]) -> FraudAnalysisResult:
        """Analyze financial data for fraud indicators.

        Args:
            financial_data: Dictionary with financial statement data.

        Returns:
            FraudAnalysisResult with findings.
        """
        result = FraudAnalysisResult()

        # Run all fraud checks
        self._check_benford_law(financial_data, result)
        self._check_balance_sheet(financial_data, result)
        self._check_cashflow_reconciliation(financial_data, result)
        self._check_round_numbers(financial_data, result)
        self._check_unusual_patterns(financial_data, result)

        # Calculate overall score
        if result.checks:
            result.overall_score = np.mean([c.score for c in result.checks])
            result.fraud_probability = min(result.overall_score, 1.0)

            # Determine risk level
            if result.overall_score >= 0.7:
                result.overall_risk = RiskLevel.CRITICAL
            elif result.overall_score >= 0.5:
                result.overall_risk = RiskLevel.HIGH
            elif result.overall_score >= 0.3:
                result.overall_risk = RiskLevel.MEDIUM
            else:
                result.overall_risk = RiskLevel.LOW

        # Generate summary
        result.summary = self._generate_summary(result)
        result.red_flags = self._collect_red_flags(result)

        return result

    def _check_benford_law(
        self,
        data: Dict[str, Any],
        result: FraudAnalysisResult,
    ) -> None:
        """Check if numbers follow Benford's Law distribution.

        Benford's Law states that in naturally occurring datasets,
        the first digit is more likely to be small (1 appears ~30% of time).

        Args:
            data: Financial data.
            result: Result to update.
        """
        # Extract all numbers from data
        numbers = self._extract_numbers(data)

        if len(numbers) < 30:
            # Not enough data for reliable analysis
            return

        # Get first digits
        first_digits = []
        for num in numbers:
            if num != 0:
                # Get absolute value and find first digit
                abs_num = abs(num)
                while abs_num >= 10:
                    abs_num /= 10
                first_digit = int(abs_num)
                if 1 <= first_digit <= 9:
                    first_digits.append(first_digit)

        if len(first_digits) < 30:
            return

        # Calculate observed distribution
        observed = {}
        for digit in range(1, 10):
            observed[digit] = first_digits.count(digit) / len(first_digits)

        # Compare with expected (Benford's Law)
        chi_square = 0.0
        for digit in range(1, 10):
            expected = self.BENFORD_DISTRIBUTION[digit]
            observed_val = observed.get(digit, 0)
            chi_square += ((observed_val - expected) ** 2) / expected

        # Normalize score (higher = more deviation from Benford)
        score = min(chi_square / 10, 1.0)

        # Determine risk level
        if chi_square > 20:
            risk = RiskLevel.HIGH
            desc = f"Significant deviation from Benford's Law (χ²={chi_square:.2f})"
        elif chi_square > 15:
            risk = RiskLevel.MEDIUM
            desc = f"Moderate deviation from Benford's Law (χ²={chi_square:.2f})"
        else:
            risk = RiskLevel.LOW
            desc = f"Numbers follow expected distribution (χ²={chi_square:.2f})"

        check = FraudCheck(
            indicator=FraudIndicator.BENFORD_VIOLATION,
            risk_level=risk,
            score=score,
            description=desc,
            details={
                "chi_square": chi_square,
                "observed_distribution": observed,
                "sample_size": len(first_digits),
            },
        )
        result.checks.append(check)

    def _check_balance_sheet(
        self,
        data: Dict[str, Any],
        result: FraudAnalysisResult,
    ) -> None:
        """Verify balance sheet equation: Assets = Liabilities + Equity.

        Args:
            data: Financial data.
            result: Result to update.
        """
        balance_sheet = data.get("balance_sheet", {})

        assets = balance_sheet.get("total_assets")
        liabilities = balance_sheet.get("total_liabilities")
        equity = balance_sheet.get("total_equity")

        if assets is None or liabilities is None or equity is None:
            return

        # Calculate expected vs actual
        expected_equity = assets - liabilities
        difference = abs(equity - expected_equity)

        # Allow small rounding differences (0.5% tolerance)
        tolerance = abs(assets) * 0.005

        if difference > tolerance:
            score = min(difference / abs(assets), 1.0)
            risk = RiskLevel.HIGH if score > 0.05 else RiskLevel.MEDIUM
            desc = f"Balance sheet mismatch: {difference:,.0f} IDR difference"
        else:
            score = 0.0
            risk = RiskLevel.LOW
            desc = "Balance sheet equation verified"

        check = FraudCheck(
            indicator=FraudIndicator.BALANCE_MISMATCH,
            risk_level=risk,
            score=score,
            description=desc,
            details={
                "total_assets": assets,
                "total_liabilities": liabilities,
                "total_equity": equity,
                "expected_equity": expected_equity,
                "difference": difference,
            },
        )
        result.checks.append(check)

    def _check_cashflow_reconciliation(
        self,
        data: Dict[str, Any],
        result: FraudAnalysisResult,
    ) -> None:
        """Check if cash flow reconciles with net income.

        Operating cash flow should generally track with net income
        over time. Large persistent deviations may indicate manipulation.

        Args:
            data: Financial data.
            result: Result to update.
        """
        income_stmt = data.get("income_statement", {})
        cashflow = data.get("cash_flow", {})

        net_income = income_stmt.get("net_income")
        operating_cf = cashflow.get("operating_cash_flow")

        if net_income is None or operating_cf is None:
            return

        # Calculate cash flow quality ratio
        if net_income != 0:
            cf_ratio = operating_cf / net_income
        else:
            cf_ratio = 0

        # Very low or negative ratio when profitable is suspicious
        if net_income > 0 and operating_cf <= 0:
            score = 0.8
            risk = RiskLevel.HIGH
            desc = f"Positive net income but negative operating cash flow"
        elif net_income > 0 and cf_ratio < 0.5:
            score = 0.6
            risk = RiskLevel.MEDIUM
            desc = f"Low cash flow quality (OCF/NI = {cf_ratio:.2f})"
        elif cf_ratio > 1.5:
            score = 0.3
            risk = RiskLevel.LOW
            desc = f"Strong cash generation (OCF/NI = {cf_ratio:.2f})"
        else:
            score = 0.0
            risk = RiskLevel.LOW
            desc = f"Normal cash flow relationship (OCF/NI = {cf_ratio:.2f})"

        check = FraudCheck(
            indicator=FraudIndicator.CASHFLOW_MISMATCH,
            risk_level=risk,
            score=score,
            description=desc,
            details={
                "net_income": net_income,
                "operating_cash_flow": operating_cf,
                "cf_quality_ratio": cf_ratio,
            },
        )
        result.checks.append(check)

    def _check_round_numbers(
        self,
        data: Dict[str, Any],
        result: FraudAnalysisResult,
    ) -> None:
        """Check for unusual frequency of round numbers.

        Authentic financial data should have natural number distribution.
        High frequency of round numbers may indicate estimation or fabrication.

        Args:
            data: Financial data.
            result: Result to update.
        """
        numbers = self._extract_numbers(data)

        if len(numbers) < 20:
            return

        # Count "round" numbers (multiples of 1000, 10000, etc.)
        round_count = 0
        for num in numbers:
            if num == 0:
                continue
            # Check if number is "round"
            abs_num = abs(num)
            if abs_num >= 1000:
                # Count as round if divisible by large powers of 10
                if abs_num % 1_000_000 == 0:
                    round_count += 1
                elif abs_num >= 100_000 and abs_num % 100_000 == 0:
                    round_count += 0.5

        # Expected round number frequency is low
        round_ratio = round_count / len(numbers)

        if round_ratio > 0.3:
            score = 0.7
            risk = RiskLevel.HIGH
            desc = f"High frequency of round numbers ({round_ratio:.1%})"
        elif round_ratio > 0.15:
            score = 0.4
            risk = RiskLevel.MEDIUM
            desc = f"Elevated round number frequency ({round_ratio:.1%})"
        else:
            score = 0.0
            risk = RiskLevel.LOW
            desc = f"Normal number distribution ({round_ratio:.1%} round)"

        check = FraudCheck(
            indicator=FraudIndicator.ROUND_NUMBERS,
            risk_level=risk,
            score=score,
            description=desc,
            details={
                "round_ratio": round_ratio,
                "sample_size": len(numbers),
            },
        )
        result.checks.append(check)

    def _check_unusual_patterns(
        self,
        data: Dict[str, Any],
        result: FraudAnalysisResult,
    ) -> None:
        """Check for unusual patterns in financial data.

        Looks for:
        - Large jumps in revenue
        - Unusual expense ratios
        - Inconsistent margins

        Args:
            data: Financial data.
            result: Result to update.
        """
        income_stmt = data.get("income_statement", {})

        revenue = income_stmt.get("revenue")
        gross_profit = income_stmt.get("gross_profit")
        operating_income = income_stmt.get("operating_income")
        net_income = income_stmt.get("net_income")

        score = 0.0
        findings = []

        # Check margin consistency
        if revenue and revenue > 0:
            if gross_profit is not None:
                gross_margin = gross_profit / revenue
                if gross_margin > 0.8:
                    score += 0.2
                    findings.append(f"Unusually high gross margin ({gross_margin:.1%})")
                elif gross_margin < 0:
                    score += 0.3
                    findings.append("Negative gross margin")

            if net_income is not None:
                net_margin = net_income / revenue
                if net_margin > 0.5:
                    score += 0.2
                    findings.append(f"Unusually high net margin ({net_margin:.1%})")

        # Check if expenses are reasonable
        if gross_profit and operating_income and gross_profit > 0:
            expense_ratio = (gross_profit - operating_income) / gross_profit
            if expense_ratio < 0:
                score += 0.3
                findings.append("Operating expenses exceed gross profit")

        score = min(score, 1.0)

        if score >= 0.5:
            risk = RiskLevel.HIGH
        elif score >= 0.3:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        desc = "; ".join(findings) if findings else "No unusual patterns detected"

        check = FraudCheck(
            indicator=FraudIndicator.UNUSUAL_PATTERNS,
            risk_level=risk,
            score=score,
            description=desc,
            details={"findings": findings},
        )
        result.checks.append(check)

    def _extract_numbers(self, data: Dict[str, Any]) -> List[float]:
        """Extract all numeric values from financial data.

        Args:
            data: Financial data dictionary.

        Returns:
            List of numeric values.
        """
        numbers = []

        def extract_recursive(obj):
            if isinstance(obj, (int, float)):
                if obj != 0:
                    numbers.append(float(obj))
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    extract_recursive(item)

        extract_recursive(data)
        return numbers

    def _generate_summary(self, result: FraudAnalysisResult) -> str:
        """Generate analysis summary.

        Args:
            result: Analysis result.

        Returns:
            Summary string.
        """
        high_risk = [c for c in result.checks if c.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        medium_risk = [c for c in result.checks if c.risk_level == RiskLevel.MEDIUM]

        lines = [
            f"Fraud Analysis: {result.overall_risk.value.upper()} Risk",
            f"Overall Score: {result.overall_score:.2f}",
            f"Fraud Probability: {result.fraud_probability:.1%}",
            "",
        ]

        if high_risk:
            lines.append("HIGH RISK FINDINGS:")
            for check in high_risk:
                lines.append(f"  - {check.description}")
            lines.append("")

        if medium_risk:
            lines.append("MEDIUM RISK FINDINGS:")
            for check in medium_risk:
                lines.append(f"  - {check.description}")

        return "\n".join(lines)

    def _collect_red_flags(self, result: FraudAnalysisResult) -> List[str]:
        """Collect red flag descriptions.

        Args:
            result: Analysis result.

        Returns:
            List of red flag descriptions.
        """
        return [
            check.description
            for check in result.checks
            if check.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        ]


def analyze_fraud(financial_data: Dict[str, Any]) -> FraudAnalysisResult:
    """Convenience function to analyze financial data for fraud.

    Args:
        financial_data: Dictionary with financial statement data.

    Returns:
        FraudAnalysisResult.
    """
    detector = FraudDetector()
    return detector.analyze(financial_data)
