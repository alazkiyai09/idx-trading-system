"""
Base Agent Module

Provides base class for fundamental analysis agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class AgentRole(Enum):
    """Role of the analysis agent."""
    AUDITOR = "auditor"
    GROWTH_ANALYST = "growth_analyst"
    VALUE_ANALYST = "value_analyst"
    RISK_ANALYST = "risk_analyst"
    FORENSIC_ANALYST = "forensic_analyst"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentFinding:
    """A single finding from an agent.

    Attributes:
        category: Category of finding.
        description: Description of finding.
        severity: Severity level (info, warning, concern, critical).
        score: Numerical score (0-100).
        evidence: Supporting evidence.
    """

    category: str
    description: str
    severity: str = "info"
    score: float = 50.0
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "score": self.score,
            "evidence": self.evidence,
        }


@dataclass
class AgentReport:
    """Report from an analysis agent.

    Attributes:
        agent_role: Role of the agent.
        overall_score: Overall score (0-100).
        findings: List of findings.
        recommendation: Final recommendation.
        confidence: Confidence in analysis.
        cross_references: References to other agent findings.
    """

    agent_role: AgentRole
    overall_score: float = 50.0
    findings: List[AgentFinding] = field(default_factory=list)
    recommendation: str = ""
    confidence: float = 0.5
    cross_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_role": self.agent_role.value,
            "overall_score": self.overall_score,
            "findings": [f.to_dict() for f in self.findings],
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "cross_references": self.cross_references,
        }


class BaseAgent(ABC):
    """Base class for analysis agents.

    Each agent provides a specific perspective on the fundamental analysis:
    - Auditor: Analyzes auditor reports and opinions
    - Growth Analyst: Optimistic, forward-looking perspective
    - Value Analyst: Conservative, value-focused perspective
    - Risk Analyst: Pessimistic, risk-identification perspective
    - Forensic Analyst: Fraud detection and authenticity
    - Synthesizer: Combines all perspectives (uses Claude Opus)

    Example:
        class MyAgent(BaseAgent):
            def analyze(self, data, other_reports=None):
                report = AgentReport(agent_role=AgentRole.MY_ROLE)
                # ... analysis logic
                return report
    """

    def __init__(self, role: AgentRole) -> None:
        """Initialize agent.

        Args:
            role: Agent role.
        """
        self.role = role

    @abstractmethod
    def analyze(
        self,
        data: Dict[str, Any],
        other_reports: Optional[List[AgentReport]] = None,
    ) -> AgentReport:
        """Analyze financial data.

        Args:
            data: Financial data to analyze.
            other_reports: Reports from other agents (for cross-referencing).

        Returns:
            AgentReport with findings.
        """
        pass

    def _calculate_weighted_score(
        self,
        findings: List[AgentFinding],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate weighted score from findings.

        Args:
            findings: List of findings.
            weights: Optional category weights.

        Returns:
            Weighted score (0-100).
        """
        if not findings:
            return 50.0

        if weights is None:
            # Equal weights
            return sum(f.score for f in findings) / len(findings)

        total_weight = 0.0
        weighted_sum = 0.0

        for finding in findings:
            weight = weights.get(finding.category, 1.0)
            weighted_sum += finding.score * weight
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return weighted_sum / total_weight

    def _add_severity_finding(
        self,
        findings: List[AgentFinding],
        category: str,
        description: str,
        value: float,
        thresholds: Dict[str, tuple],
    ) -> None:
        """Add finding with automatic severity classification.

        Args:
            findings: List to add finding to.
            category: Finding category.
            description: Finding description.
            value: Value to classify.
            thresholds: Severity thresholds (severity -> (low, high)).
        """
        severity = "info"
        score = 50.0

        for sev, (low, high) in thresholds.items():
            if low <= value < high:
                severity = sev
                break

        # Convert severity to score
        severity_scores = {
            "info": 80,
            "warning": 60,
            "concern": 40,
            "critical": 20,
        }
        score = severity_scores.get(severity, 50)

        findings.append(AgentFinding(
            category=category,
            description=description,
            severity=severity,
            score=score,
        ))
