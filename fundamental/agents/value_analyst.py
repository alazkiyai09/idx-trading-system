"""
Value Analyst Agent Module

Conservative, value-focused analysis emphasizing:
- Earnings quality
- Cash flow verification
- Dividend sustainability
- Margin stability
"""

import logging
from typing import Dict, List, Optional, Any

from .base import BaseAgent, AgentRole, AgentReport, AgentFinding

logger = logging.getLogger(__name__)


class ValueAnalyst(BaseAgent):
    """Conservative, value-focused analysis agent.

    Provides skeptical perspective emphasizing:
    - Earnings quality and sustainability
    - Cash flow verification
    - Balance sheet strength
    - Valuation discipline

    Example:
        agent = ValueAnalyst()
        report = agent.analyze(financial_data, other_reports)
    """

    def __init__(self) -> None:
        """Initialize value analyst."""
        super().__init__(AgentRole.VALUE_ANALYST)

    def analyze(
        self,
        data: Dict[str, Any],
        other_reports: Optional[List[AgentReport]] = None,
    ) -> AgentReport:
        """Analyze value and quality.

        Args:
            data: Financial data.
            other_reports: Reports from other agents for cross-reference.

        Returns:
            AgentReport with value analysis.
        """
        report = AgentReport(agent_role=self.role)
        findings = []

        income = data.get("income_statement", {})
        balance = data.get("balance_sheet", {})
        cashflow = data.get("cash_flow", {})
        ratios = data.get("ratios", {})

        # Analyze earnings quality
        self._analyze_earnings_quality(income, cashflow, findings)

        # Analyze balance sheet strength
        self._analyze_balance_sheet_strength(balance, findings)

        # Analyze cash generation
        self._analyze_cash_generation(cashflow, income, findings)

        # Cross-reference with other agents
        if other_reports:
            self._cross_reference_reports(other_reports, findings)

        report.findings = findings
        report.overall_score = self._calculate_weighted_score(findings)
        report.recommendation = self._generate_recommendation(report)
        report.confidence = 0.75

        return report

    def _analyze_earnings_quality(
        self,
        income: Dict[str, Any],
        cashflow: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze quality of reported earnings."""
        net_income = income.get("net_income", 0)
        operating_cf = cashflow.get("operating_cash_flow", 0)

        # Cash flow quality ratio
        if net_income > 0:
            cf_ratio = operating_cf / net_income

            if cf_ratio >= 1.0:
                findings.append(AgentFinding(
                    category="earnings_quality",
                    description=f"High quality earnings - cash flow exceeds net income ({cf_ratio:.1f}x)",
                    severity="info",
                    score=85,
                ))
            elif cf_ratio >= 0.8:
                findings.append(AgentFinding(
                    category="earnings_quality",
                    description=f"Acceptable earnings quality - CF/NI = {cf_ratio:.1f}x",
                    severity="info",
                    score=70,
                ))
            elif cf_ratio > 0:
                findings.append(AgentFinding(
                    category="earnings_quality",
                    description=f"Lower earnings quality - CF/NI = {cf_ratio:.1f}x",
                    severity="warning",
                    score=50,
                ))
            else:
                findings.append(AgentFinding(
                    category="earnings_quality",
                    description="Poor earnings quality - negative operating cash flow",
                    severity="concern",
                    score=25,
                ))

    def _analyze_balance_sheet_strength(
        self,
        balance: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze balance sheet strength."""
        total_equity = balance.get("total_equity", 0)
        total_assets = balance.get("total_assets", 0)
        total_liabilities = balance.get("total_liabilities", 0)
        current_assets = balance.get("current_assets", 0)
        current_liabilities = balance.get("current_liabilities", 0)

        # Debt to equity
        if total_equity > 0:
            debt_equity = total_liabilities / total_equity

            if debt_equity < 0.5:
                findings.append(AgentFinding(
                    category="balance_sheet",
                    description=f"Conservative leverage (D/E = {debt_equity:.1f}x)",
                    severity="info",
                    score=85,
                ))
            elif debt_equity < 1.0:
                findings.append(AgentFinding(
                    category="balance_sheet",
                    description=f"Moderate leverage (D/E = {debt_equity:.1f}x)",
                    severity="info",
                    score=70,
                ))
            elif debt_equity < 2.0:
                findings.append(AgentFinding(
                    category="balance_sheet",
                    description=f"Elevated leverage (D/E = {debt_equity:.1f}x)",
                    severity="warning",
                    score=50,
                ))
            else:
                findings.append(AgentFinding(
                    category="balance_sheet",
                    description=f"High leverage concern (D/E = {debt_equity:.1f}x)",
                    severity="concern",
                    score=30,
                ))

        # Current ratio
        if current_liabilities > 0:
            current_ratio = current_assets / current_liabilities

            if current_ratio >= 1.5:
                findings.append(AgentFinding(
                    category="liquidity",
                    description=f"Strong liquidity (current ratio = {current_ratio:.1f})",
                    severity="info",
                    score=80,
                ))
            elif current_ratio >= 1.0:
                findings.append(AgentFinding(
                    category="liquidity",
                    description=f"Adequate liquidity (current ratio = {current_ratio:.1f})",
                    severity="info",
                    score=65,
                ))
            else:
                findings.append(AgentFinding(
                    category="liquidity",
                    description=f"Liquidity risk (current ratio = {current_ratio:.1f})",
                    severity="concern",
                    score=35,
                ))

    def _analyze_cash_generation(
        self,
        cashflow: Dict[str, Any],
        income: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze cash generation capability."""
        operating_cf = cashflow.get("operating_cash_flow", 0)
        free_cf = cashflow.get("free_cash_flow", operating_cf * 0.7)  # Estimate if not available

        if operating_cf > 0 and free_cf > 0:
            findings.append(AgentFinding(
                category="cash_generation",
                description="Positive free cash flow generation",
                severity="info",
                score=80,
            ))
        elif operating_cf > 0:
            findings.append(AgentFinding(
                category="cash_generation",
                description="Positive operating cash flow but capex heavy",
                severity="info",
                score=60,
            ))
        else:
            findings.append(AgentFinding(
                category="cash_generation",
                description="Cash burn - negative operating cash flow",
                severity="concern",
                score=25,
            ))

    def _cross_reference_reports(
        self,
        other_reports: List[AgentReport],
        findings: List[AgentFinding],
    ) -> None:
        """Cross-reference with other agent reports."""
        for other in other_reports:
            if other.agent_role == AgentRole.AUDITOR:
                # Check for auditor concerns
                for finding in other.findings:
                    if finding.severity in ("concern", "critical"):
                        findings.append(AgentFinding(
                            category="cross_reference",
                            description=f"Auditor concern: {finding.description}",
                            severity="warning",
                            score=40,
                        ))
                        break

    def _generate_recommendation(self, report: AgentReport) -> str:
        """Generate value-focused recommendation."""
        if report.overall_score >= 70:
            return "VALUE ASSESSMENT: High quality - strong fundamentals support investment case"
        elif report.overall_score >= 50:
            return "VALUE ASSESSMENT: Adequate - fundamentals reasonable but not exceptional"
        else:
            return "VALUE ASSESSMENT: Concerns - fundamental weaknesses identified"
