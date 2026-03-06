"""
Growth Analyst Agent Module

Forward-looking, optimistic analysis focusing on:
- Revenue growth trends
- Market expansion potential
- Competitive advantages
- Growth catalysts
"""

import logging
from typing import Dict, List, Optional, Any

from .base import BaseAgent, AgentRole, AgentReport, AgentFinding

logger = logging.getLogger(__name__)


class GrowthAnalyst(BaseAgent):
    """Optimistic, growth-focused analysis agent.

    Provides forward-looking perspective emphasizing:
    - Growth opportunities
    - Competitive strengths
    - Market potential
    - Positive catalysts

    Example:
        agent = GrowthAnalyst()
        report = agent.analyze(financial_data)
    """

    def __init__(self) -> None:
        """Initialize growth analyst."""
        super().__init__(AgentRole.GROWTH_ANALYST)

    def analyze(
        self,
        data: Dict[str, Any],
        other_reports: Optional[List[AgentReport]] = None,
    ) -> AgentReport:
        """Analyze growth potential.

        Args:
            data: Financial data.
            other_reports: Reports from other agents.

        Returns:
            AgentReport with growth analysis.
        """
        report = AgentReport(agent_role=self.role)
        findings = []

        income = data.get("income_statement", {})
        balance = data.get("balance_sheet", {})
        ratios = data.get("ratios", {})

        # Analyze revenue growth
        self._analyze_revenue_growth(income, data.get("prior_year", {}), findings)

        # Analyze profitability trajectory
        self._analyze_profitability_trend(income, findings)

        # Analyze reinvestment
        self._analyze_reinvestment(balance, income, findings)

        # Analyze market position (from ratios)
        self._analyze_market_position(income, ratios, findings)

        report.findings = findings
        report.overall_score = self._calculate_weighted_score(findings)
        report.recommendation = self._generate_recommendation(report)
        report.confidence = 0.7

        return report

    def _analyze_revenue_growth(
        self,
        income: Dict[str, Any],
        prior: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze revenue growth trends."""
        revenue = income.get("revenue", 0)
        prior_revenue = prior.get("income_statement", {}).get("revenue", 0)

        if prior_revenue > 0:
            growth_rate = (revenue - prior_revenue) / prior_revenue

            if growth_rate > 0.20:
                findings.append(AgentFinding(
                    category="revenue_growth",
                    description=f"Strong revenue growth of {growth_rate:.0%} - expanding business",
                    severity="info",
                    score=90,
                ))
            elif growth_rate > 0.10:
                findings.append(AgentFinding(
                    category="revenue_growth",
                    description=f"Solid revenue growth of {growth_rate:.0%}",
                    severity="info",
                    score=80,
                ))
            elif growth_rate > 0:
                findings.append(AgentFinding(
                    category="revenue_growth",
                    description=f"Moderate revenue growth of {growth_rate:.0%}",
                    severity="info",
                    score=65,
                ))
            else:
                findings.append(AgentFinding(
                    category="revenue_growth",
                    description=f"Revenue decline of {abs(growth_rate):.0%} - growth concerns",
                    severity="concern",
                    score=30,
                ))

    def _analyze_profitability_trend(
        self,
        income: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze profitability trajectory."""
        revenue = income.get("revenue", 0)
        gross_profit = income.get("gross_profit", 0)
        net_income = income.get("net_income", 0)

        if revenue > 0:
            gross_margin = gross_profit / revenue
            net_margin = net_income / revenue

            if gross_margin > 0.35:
                findings.append(AgentFinding(
                    category="profitability",
                    description=f"Strong gross margin ({gross_margin:.0%}) indicates pricing power",
                    severity="info",
                    score=85,
                ))
            elif gross_margin > 0.20:
                findings.append(AgentFinding(
                    category="profitability",
                    description=f"Healthy gross margin ({gross_margin:.0%})",
                    severity="info",
                    score=70,
                ))

            if net_margin > 0.15:
                findings.append(AgentFinding(
                    category="profitability",
                    description=f"Excellent net margin ({net_margin:.0%}) shows operational efficiency",
                    severity="info",
                    score=85,
                ))

    def _analyze_reinvestment(
        self,
        balance: Dict[str, Any],
        income: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze reinvestment for growth."""
        total_assets = balance.get("total_assets", 0)
        prior_assets = balance.get("prior_total_assets", total_assets * 0.9)  # Estimate if not available

        if prior_assets > 0:
            asset_growth = (total_assets - prior_assets) / prior_assets

            if asset_growth > 0.15:
                findings.append(AgentFinding(
                    category="reinvestment",
                    description=f"Strong asset base growth ({asset_growth:.0%}) indicates reinvestment",
                    severity="info",
                    score=80,
                ))

    def _analyze_market_position(
        self,
        income: Dict[str, Any],
        ratios: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze market position indicators."""
        # Check for scale advantages
        revenue = income.get("revenue", 0)

        if revenue > 1_000_000_000_000:  # > 1 trillion IDR
            findings.append(AgentFinding(
                category="market_position",
                description="Large scale provides competitive advantages",
                severity="info",
                score=80,
            ))

        # Check ROE for capital efficiency
        roe = ratios.get("roe")
        if roe and roe > 0.15:
            findings.append(AgentFinding(
                category="market_position",
                description=f"Strong ROE ({roe:.0%}) indicates competitive advantage",
                severity="info",
                score=80,
            ))

    def _generate_recommendation(self, report: AgentReport) -> str:
        """Generate growth-focused recommendation."""
        if report.overall_score >= 70:
            return "GROWTH OUTLOOK: Positive - company shows strong growth characteristics"
        elif report.overall_score >= 50:
            return "GROWTH OUTLOOK: Neutral - moderate growth potential"
        else:
            return "GROWTH OUTLOOK: Concerns - limited growth indicators"
