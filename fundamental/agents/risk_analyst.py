"""
Risk Analyst Agent Module

Pessimistic, risk-focused analysis identifying:
- Financial risks
- Operational risks
- Market risks
- Worst-case scenarios
"""

import logging
from typing import Dict, List, Optional, Any

from .base import BaseAgent, AgentRole, AgentReport, AgentFinding

logger = logging.getLogger(__name__)


class RiskAnalyst(BaseAgent):
    """Pessimistic, risk-focused analysis agent.

    Identifies and challenges:
    - Financial vulnerabilities
    - Operational risks
    - Market/competitive threats
    - Worst-case scenarios

    Acts as devil's advocate to other analysts.

    Example:
        agent = RiskAnalyst()
        report = agent.analyze(financial_data, other_reports)
    """

    def __init__(self) -> None:
        """Initialize risk analyst."""
        super().__init__(AgentRole.RISK_ANALYST)

    def analyze(
        self,
        data: Dict[str, Any],
        other_reports: Optional[List[AgentReport]] = None,
    ) -> AgentReport:
        """Analyze risks and vulnerabilities.

        Args:
            data: Financial data.
            other_reports: Reports from other agents to challenge.

        Returns:
            AgentReport with risk analysis.
        """
        report = AgentReport(agent_role=self.role)
        findings = []

        income = data.get("income_statement", {})
        balance = data.get("balance_sheet", {})
        cashflow = data.get("cash_flow", {})
        ratios = data.get("ratios", {})

        # Analyze financial risks
        self._analyze_financial_risks(income, balance, findings)

        # Analyze operational risks
        self._analyze_operational_risks(income, balance, findings)

        # Analyze liquidity risks
        self._analyze_liquidity_risks(balance, cashflow, findings)

        # Challenge other analysts
        if other_reports:
            self._challenge_other_analysts(other_reports, findings)

        report.findings = findings
        report.overall_score = 100 - self._calculate_weighted_score(findings)  # Invert for risk
        report.recommendation = self._generate_recommendation(report)
        report.confidence = 0.7

        return report

    def _analyze_financial_risks(
        self,
        income: Dict[str, Any],
        balance: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze financial vulnerabilities."""
        net_income = income.get("net_income", 0)
        interest = income.get("interest_expense", 0)
        ebit = income.get("operating_income", 0)

        total_equity = balance.get("total_equity", 0)
        total_liabilities = balance.get("total_liabilities", 0)

        # Interest coverage risk
        if interest > 0:
            coverage = ebit / interest
            if coverage < 2.0:
                findings.append(AgentFinding(
                    category="financial_risk",
                    description=f"Low interest coverage ({coverage:.1f}x) - debt service risk",
                    severity="concern",
                    score=30,
                ))
            elif coverage < 3.0:
                findings.append(AgentFinding(
                    category="financial_risk",
                    description=f"Moderate interest coverage ({coverage:.1f}x)",
                    severity="warning",
                    score=50,
                ))

        # Leverage risk
        if total_equity > 0:
            debt_equity = total_liabilities / total_equity
            if debt_equity > 2.0:
                findings.append(AgentFinding(
                    category="financial_risk",
                    description=f"High leverage (D/E = {debt_equity:.1f}x) increases bankruptcy risk",
                    severity="concern",
                    score=25,
                ))

        # Profitability risk
        revenue = income.get("revenue", 0)
        if revenue > 0:
            net_margin = net_income / revenue
            if net_margin < 0.05:
                findings.append(AgentFinding(
                    category="financial_risk",
                    description=f"Thin margins ({net_margin:.0%}) vulnerable to cost increases",
                    severity="warning",
                    score=45,
                ))

    def _analyze_operational_risks(
        self,
        income: Dict[str, Any],
        balance: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze operational vulnerabilities."""
        revenue = income.get("revenue", 0)
        cogs = income.get("cost_of_goods_sold", 0)
        inventory = balance.get("inventory", 0)

        # Inventory risk
        if inventory > 0 and cogs > 0:
            inventory_days = (inventory / cogs) * 365
            if inventory_days > 90:
                findings.append(AgentFinding(
                    category="operational_risk",
                    description=f"High inventory days ({inventory_days:.0f}) - obsolescence risk",
                    severity="warning",
                    score=45,
                ))

        # Revenue concentration (placeholder)
        findings.append(AgentFinding(
            category="operational_risk",
            description="Revenue concentration analysis requires additional data",
            severity="info",
            score=60,
        ))

    def _analyze_liquidity_risks(
        self,
        balance: Dict[str, Any],
        cashflow: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze liquidity vulnerabilities."""
        current_assets = balance.get("current_assets", 0)
        current_liabilities = balance.get("current_liabilities", 0)
        cash = balance.get("cash", 0)
        operating_cf = cashflow.get("operating_cash_flow", 0)

        # Current ratio risk
        if current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            if current_ratio < 1.0:
                findings.append(AgentFinding(
                    category="liquidity_risk",
                    description=f"Liquidity crisis risk (current ratio = {current_ratio:.1f})",
                    severity="critical",
                    score=15,
                ))
            elif current_ratio < 1.2:
                findings.append(AgentFinding(
                    category="liquidity_risk",
                    description=f"Tight liquidity (current ratio = {current_ratio:.1f})",
                    severity="concern",
                    score=35,
                ))

        # Cash flow risk
        if operating_cf < 0:
            findings.append(AgentFinding(
                category="liquidity_risk",
                description="Negative operating cash flow - funding risk",
                severity="concern",
                score=25,
            ))

    def _challenge_other_analysts(
        self,
        other_reports: List[AgentReport],
        findings: List[AgentFinding],
    ) -> None:
        """Challenge findings from other analysts."""
        for other in other_reports:
            if other.agent_role == AgentRole.GROWTH_ANALYST:
                # Challenge growth assumptions
                if other.overall_score > 70:
                    findings.append(AgentFinding(
                        category="challenge",
                        description="Growth assumptions may be optimistic - verify sustainability",
                        severity="warning",
                        score=40,
                    ))

            elif other.agent_role == AgentRole.VALUE_ANALYST:
                # Challenge value assumptions
                for finding in other.findings:
                    if "high quality" in finding.description.lower():
                        findings.append(AgentFinding(
                            category="challenge",
                            description="Earnings quality claims should be verified with cash flow trends",
                            severity="info",
                            score=55,
                        ))

    def _generate_recommendation(self, report: AgentReport) -> str:
        """Generate risk-focused recommendation."""
        critical = [f for f in report.findings if f.severity == "critical"]
        concerns = [f for f in report.findings if f.severity == "concern"]

        if critical:
            return f"RISK ALERT: {critical[0].description}. Investment not recommended."
        elif len(concerns) > 2:
            return f"RISK WARNING: Multiple concerns identified ({len(concerns)}). Proceed with caution."
        elif concerns:
            return f"RISK NOTICE: {concerns[0].description}. Monitor closely."
        else:
            return "RISK ASSESSMENT: No significant risks identified beyond normal business risks."
