"""
Auditor Agent Module

Analyzes auditor reports and opinions to assess:
- Opinion type (unqualified, qualified, adverse, disclaimer)
- Going concern warnings
- Key audit matters
- Auditor independence
- Historical patterns
"""

import logging
import re
from typing import Dict, List, Optional, Any

from .base import BaseAgent, AgentRole, AgentReport, AgentFinding

logger = logging.getLogger(__name__)


class AuditorAgent(BaseAgent):
    """Analyzes auditor reports and opinions.

    Examines auditor reports for:
    - Opinion type and strength
    - Going concern warnings
    - Key audit matters and emphasis of matter
    - Auditor changes and independence issues

    Example:
        agent = AuditorAgent()
        report = agent.analyze({"auditor_report": "..."})
        print(f"Auditor score: {report.overall_score}")
    """

    # Opinion type patterns
    UNQUALIFIED_PATTERNS = [
        r"pendapat wajar tanpa pengecualian",
        r"unqualified opinion",
        r"opini wajar tanpa pengecualian",
        r"present fairly",
        r"free from material misstatement",
    ]

    QUALIFIED_PATTERNS = [
        r"pendapat wajar dengan pengecualian",
        r"qualified opinion",
        r"opini wajar dengan pengecualian",
        r"except for",
        r"subject to",
    ]

    ADVERSE_PATTERNS = [
        r"pendapat tidak wajar",
        r"adverse opinion",
        r"opini tidak wajar",
        r"materially misstated",
    ]

    DISCLAIMER_PATTERNS = [
        r"tidak menyatakan pendapat",
        r"disclaimer of opinion",
        r"opini tidak menyatakan pendapat",
        r"unable to express an opinion",
        r"scope limitation",
    ]

    # Going concern patterns
    GOING_CONCERN_PATTERNS = [
        r"going concern",
        r"kelangsungan usaha",
        r"ketidakpastian signifikan",
        r"significant uncertainty",
        r"material uncertainty",
        r"ability to continue",
        r"continuity of operations",
    ]

    # Red flag patterns
    RED_FLAG_PATTERNS = [
        r"penggantian auditor",
        r"auditor change",
        r"rotasi auditor",
        r"auditor rotation",
        r"keterbatasan prosedur",
        r"scope limitation",
        r"ketidaksesuaian",
        r"non-compliance",
        r"pelanggaran",
        r"violation",
    ]

    def __init__(self) -> None:
        """Initialize auditor agent."""
        super().__init__(AgentRole.AUDITOR)

    def analyze(
        self,
        data: Dict[str, Any],
        other_reports: Optional[List[AgentReport]] = None,
    ) -> AgentReport:
        """Analyze auditor report.

        Args:
            data: Financial data including auditor report text.
            other_reports: Reports from other agents.

        Returns:
            AgentReport with auditor analysis.
        """
        report = AgentReport(agent_role=self.role)
        findings = []

        auditor_text = data.get("auditor_report", "")
        if isinstance(auditor_text, dict):
            auditor_text = " ".join(str(v) for v in auditor_text.values())

        if not auditor_text:
            findings.append(AgentFinding(
                category="auditor_availability",
                description="Auditor report not available for analysis",
                severity="warning",
                score=60,
            ))
            report.confidence = 0.3
        else:
            report.confidence = 0.8

            # Analyze opinion type
            self._analyze_opinion_type(auditor_text, findings)

            # Check for going concern warnings
            self._check_going_concern(auditor_text, findings)

            # Check for red flags
            self._check_red_flags(auditor_text, findings)

            # Analyze auditor information
            self._analyze_auditor_info(auditor_text, data, findings)

        report.findings = findings
        report.overall_score = self._calculate_weighted_score(findings, {
            "opinion_type": 2.0,
            "going_concern": 3.0,
            "red_flags": 2.0,
            "auditor_info": 1.0,
        })

        report.recommendation = self._generate_recommendation(report)

        return report

    def _analyze_opinion_type(self, text: str, findings: List[AgentFinding]) -> None:
        """Analyze the type of auditor opinion.

        Args:
            text: Auditor report text.
            findings: List to add findings to.
        """
        text_lower = text.lower()

        # Check for each opinion type
        is_unqualified = any(
            re.search(p, text_lower) for p in self.UNQUALIFIED_PATTERNS
        )
        is_qualified = any(
            re.search(p, text_lower) for p in self.QUALIFIED_PATTERNS
        )
        is_adverse = any(
            re.search(p, text_lower) for p in self.ADVERSE_PATTERNS
        )
        is_disclaimer = any(
            re.search(p, text_lower) for p in self.DISCLAIMER_PATTERNS
        )

        if is_adverse:
            findings.append(AgentFinding(
                category="opinion_type",
                description="ADVERSE opinion - financial statements materially misstated",
                severity="critical",
                score=10,
                evidence=["Adverse opinion language detected"],
            ))
        elif is_disclaimer:
            findings.append(AgentFinding(
                category="opinion_type",
                description="DISCLAIMER of opinion - auditor unable to form opinion",
                severity="critical",
                score=15,
                evidence=["Disclaimer language detected"],
            ))
        elif is_qualified:
            findings.append(AgentFinding(
                category="opinion_type",
                description="QUALIFIED opinion - issues identified in audit",
                severity="concern",
                score=40,
                evidence=["Qualified opinion language detected"],
            ))
        elif is_unqualified:
            findings.append(AgentFinding(
                category="opinion_type",
                description="UNQUALIFIED (clean) opinion - favorable auditor assessment",
                severity="info",
                score=90,
                evidence=["Unqualified opinion language detected"],
            ))
        else:
            findings.append(AgentFinding(
                category="opinion_type",
                description="Opinion type could not be determined",
                severity="warning",
                score=50,
            ))

    def _check_going_concern(self, text: str, findings: List[AgentFinding]) -> None:
        """Check for going concern warnings.

        Args:
            text: Auditor report text.
            findings: List to add findings to.
        """
        text_lower = text.lower()

        going_concern_matches = [
            pattern for pattern in self.GOING_CONCERN_PATTERNS
            if re.search(pattern, text_lower)
        ]

        if going_concern_matches:
            findings.append(AgentFinding(
                category="going_concern",
                description="Going concern warning detected - company survival at risk",
                severity="critical",
                score=15,
                evidence=[f"Pattern found: {p}" for p in going_concern_matches[:3]],
            ))
        else:
            findings.append(AgentFinding(
                category="going_concern",
                description="No going concern warnings detected",
                severity="info",
                score=90,
            ))

    def _check_red_flags(self, text: str, findings: List[AgentFinding]) -> None:
        """Check for auditor-related red flags.

        Args:
            text: Auditor report text.
            findings: List to add findings to.
        """
        text_lower = text.lower()

        red_flags = []
        for pattern in self.RED_FLAG_PATTERNS:
            if re.search(pattern, text_lower):
                red_flags.append(pattern)

        if red_flags:
            findings.append(AgentFinding(
                category="red_flags",
                description=f"Auditor red flags detected: {len(red_flags)} issues found",
                severity="concern" if len(red_flags) < 3 else "warning",
                score=30 if len(red_flags) < 3 else 40,
                evidence=[f"Red flag: {p}" for p in red_flags[:5]],
            ))
        else:
            findings.append(AgentFinding(
                category="red_flags",
                description="No auditor red flags detected",
                severity="info",
                score=85,
            ))

    def _analyze_auditor_info(
        self,
        text: str,
        data: Dict[str, Any],
        findings: List[AgentFinding],
    ) -> None:
        """Analyze auditor information.

        Args:
            text: Auditor report text.
            data: Full financial data.
            findings: List to add findings to.
        """
        # Check for Big 4 auditor (indicates higher quality)
        big_4_patterns = [
            r"pwc",
            r"price.?waterhouse",
            r"deloitte",
            r"ernst.?young",
            r"e&y",
            r"kpmg",
        ]

        text_lower = text.lower()
        is_big_4 = any(re.search(p, text_lower) for p in big_4_patterns)

        if is_big_4:
            findings.append(AgentFinding(
                category="auditor_info",
                description="Big 4 auditor - higher audit quality expected",
                severity="info",
                score=85,
                evidence=["Big 4 auditor detected"],
            ))
        else:
            findings.append(AgentFinding(
                category="auditor_info",
                description="Non-Big 4 auditor",
                severity="info",
                score=70,
            ))

    def _generate_recommendation(self, report: AgentReport) -> str:
        """Generate recommendation from findings.

        Args:
            report: Agent report.

        Returns:
            Recommendation string.
        """
        critical = [f for f in report.findings if f.severity == "critical"]
        concerns = [f for f in report.findings if f.severity == "concern"]

        if critical:
            return f"CRITICAL: {critical[0].description}. Exercise extreme caution."
        elif concerns:
            return f"CONCERN: {concerns[0].description}. Further investigation recommended."
        elif report.overall_score >= 70:
            return "FAVORABLE: Clean auditor opinion with no significant issues."
        else:
            return "NEUTRAL: Auditor report analysis inconclusive."
