"""
Forensic Analyst Agent Module

Final fraud and authenticity assessment:
- Reviews all prior agent findings
- Assesses fraud probability
- Identifies manipulation indicators
- Provides authenticity verdict
"""

import logging
from typing import Dict, List, Optional, Any

from .base import BaseAgent, AgentRole, AgentReport, AgentFinding

logger = logging.getLogger(__name__)


class ForensicAnalyst(BaseAgent):
    """Forensic analysis agent for fraud detection.

    Reviews all findings and provides:
    - Fraud probability assessment
    - Authenticity verdict
    - Specific manipulation indicators
    - Investigation recommendations

    Example:
        agent = ForensicAnalyst()
        report = agent.analyze(data, other_reports)
        print(f"Fraud probability: {report.overall_score}")
    """

    def __init__(self) -> None:
        """Initialize forensic analyst."""
        super().__init__(AgentRole.FORENSIC_ANALYST)

    def analyze(
        self,
        data: Dict[str, Any],
        other_reports: Optional[List[AgentReport]] = None,
    ) -> AgentReport:
        """Perform forensic analysis.

        Args:
            data: Financial data including fraud analysis results.
            other_reports: Reports from other agents.

        Returns:
            AgentReport with forensic findings.
        """
        report = AgentReport(agent_role=self.role)
        findings = []

        fraud_analysis = data.get("fraud_analysis", {})

        # Analyze fraud indicators
        self._analyze_fraud_indicators(fraud_analysis, findings)

        # Review other agent reports for red flags
        if other_reports:
            self._review_agent_reports(other_reports, findings)

        # Check for manipulation patterns
        self._check_manipulation_patterns(data, findings)

        # Make authenticity assessment
        self._assess_authenticity(findings, report)

        report.findings = findings
        report.overall_score = self._calculate_fraud_score(findings)
        report.recommendation = self._generate_verdict(report)
        report.confidence = 0.8

        return report

    def _analyze_fraud_indicators(
        self,
        fraud_analysis: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze fraud detection results."""
        if not fraud_analysis:
            findings.append(AgentFinding(
                category="fraud_analysis",
                description="Fraud analysis not available",
                severity="warning",
                score=50,
            ))
            return

        overall_risk = fraud_analysis.get("overall_risk", "unknown")
        fraud_prob = fraud_analysis.get("fraud_probability", 0)

        if overall_risk == "critical":
            findings.append(AgentFinding(
                category="fraud_risk",
                description=f"CRITICAL fraud risk detected (probability: {fraud_prob:.0%})",
                severity="critical",
                score=10,
                evidence=fraud_analysis.get("red_flags", []),
            ))
        elif overall_risk == "high":
            findings.append(AgentFinding(
                category="fraud_risk",
                description=f"HIGH fraud risk detected (probability: {fraud_prob:.0%})",
                severity="concern",
                score=25,
                evidence=fraud_analysis.get("red_flags", []),
            ))
        elif overall_risk == "medium":
            findings.append(AgentFinding(
                category="fraud_risk",
                description=f"MODERATE fraud risk (probability: {fraud_prob:.0%})",
                severity="warning",
                score=50,
            ))
        else:
            findings.append(AgentFinding(
                category="fraud_risk",
                description=f"Low fraud risk (probability: {fraud_prob:.0%})",
                severity="info",
                score=85,
            ))

        # Add specific red flags
        red_flags = fraud_analysis.get("red_flags", [])
        for flag in red_flags[:3]:
            findings.append(AgentFinding(
                category="red_flag",
                description=flag,
                severity="warning",
                score=40,
            ))

    def _review_agent_reports(
        self,
        other_reports: List[AgentReport],
        findings: List[AgentFinding],
    ) -> None:
        """Review other agent reports for concerning patterns."""
        for other in other_reports:
            # Check auditor report
            if other.agent_role == AgentRole.AUDITOR:
                for finding in other.findings:
                    if finding.severity in ("concern", "critical"):
                        findings.append(AgentFinding(
                            category="auditor_concern",
                            description=f"Auditor issue: {finding.description}",
                            severity="warning",
                            score=35,
                        ))

            # Check for divergent views
            if other.agent_role == AgentRole.RISK_ANALYST:
                if other.overall_score < 40:
                    findings.append(AgentFinding(
                        category="risk_concern",
                        description="Risk analyst identified significant concerns",
                        severity="warning",
                        score=40,
                    ))

    def _check_manipulation_patterns(
        self,
        data: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Check for specific manipulation patterns."""
        income = data.get("income_statement", {})
        cashflow = data.get("cash_flow", {})

        net_income = income.get("net_income", 0)
        operating_cf = cashflow.get("operating_cash_flow", 0)

        # Earnings manipulation check
        if net_income > 0 and operating_cf < net_income * 0.5:
            findings.append(AgentFinding(
                category="manipulation",
                description="Earnings significantly exceed cash flow - possible manipulation",
                severity="concern",
                score=30,
            ))

    def _assess_authenticity(
        self,
        findings: List[AgentFinding],
        report: AgentReport,
    ) -> None:
        """Assess overall authenticity."""
        critical_count = len([f for f in findings if f.severity == "critical"])
        concern_count = len([f for f in findings if f.severity == "concern"])

        if critical_count > 0:
            authenticity = "HIGHLY QUESTIONABLE"
        elif concern_count > 2:
            authenticity = "QUESTIONABLE"
        elif concern_count > 0:
            authenticity = "UNCERTAIN"
        else:
            authenticity = "APPEARS AUTHENTIC"

        findings.append(AgentFinding(
            category="authenticity",
            description=f"Authenticity assessment: {authenticity}",
            severity="info" if authenticity == "APPEARS AUTHENTIC" else "warning",
            score=80 if authenticity == "APPEARS AUTHENTIC" else 40,
        ))

    def _calculate_fraud_score(self, findings: List[AgentFinding]) -> float:
        """Calculate fraud score (higher = more suspicious)."""
        if not findings:
            return 0

        # Weight by severity
        severity_weights = {
            "critical": 1.0,
            "concern": 0.7,
            "warning": 0.4,
            "info": 0.0,
        }

        total_score = 0
        for finding in findings:
            weight = severity_weights.get(finding.severity, 0)
            # Invert score (higher original score = lower fraud risk)
            total_score += weight * (100 - finding.score)

        return min(total_score / len(findings), 100)

    def _generate_verdict(self, report: AgentReport) -> str:
        """Generate forensic verdict."""
        if report.overall_score > 70:
            return "VERDICT: HIGH FRAUD PROBABILITY - Do not invest without thorough investigation"
        elif report.overall_score > 40:
            return "VERDICT: ELEVATED CONCERNS - Additional verification recommended"
        elif report.overall_score > 20:
            return "VERDICT: SOME CONCERNS - Standard due diligence advised"
        else:
            return "VERDICT: LOW FRAUD INDICATORS - Financials appear authentic"
