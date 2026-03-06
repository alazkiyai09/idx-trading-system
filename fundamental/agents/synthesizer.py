"""
Synthesizer Agent Module

Combines all agent findings into a cohesive analysis:
- Aggregates findings from all agents
- Resolves conflicting viewpoints
- Generates final scores and recommendation
- Creates comprehensive investment thesis
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseAgent, AgentRole, AgentReport, AgentFinding

logger = logging.getLogger(__name__)


class InvestmentRating(Enum):
    """Overall investment rating."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    UNDERWEIGHT = "underweight"
    SELL = "sell"


class ConfidenceLevel(Enum):
    """Confidence level in the analysis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class InvestmentThesis:
    """Investment thesis summary."""
    rating: InvestmentRating
    confidence: ConfidenceLevel
    overall_score: float
    key_strengths: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    primary_thesis: str = ""
    catalysts: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)


@dataclass
class AgentConsensus:
    """Consensus across agents."""
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    consensus_strength: float = 0.0
    divergent_views: List[str] = field(default_factory=list)


class Synthesizer(BaseAgent):
    """Synthesizer agent that combines all findings.

    Acts as the final step in the multi-agent analysis:
    - Aggregates all agent reports
    - Identifies consensus and divergence
    - Resolves conflicts using weighted approach
    - Generates final investment recommendation

    Example:
        synthesizer = Synthesizer()
        final_report = synthesizer.synthesize(all_agent_reports, financial_data)
        print(f"Rating: {final_report.thesis.rating.value}")
    """

    # Weight assigned to each agent type
    AGENT_WEIGHTS = {
        AgentRole.AUDITOR: 1.5,      # Auditor opinion is critical
        AgentRole.VALUE_ANALYST: 1.2,  # Value focus is important
        AgentRole.RISK_ANALYST: 1.3,   # Risk perspective weighted higher
        AgentRole.GROWTH_ANALYST: 1.0, # Growth is baseline
        AgentRole.FORENSIC_ANALYST: 1.5, # Forensic issues are critical
    }

    def __init__(self) -> None:
        """Initialize synthesizer."""
        super().__init__(AgentRole.SYNTHESIZER)

    def analyze(
        self,
        data: Dict[str, Any],
        other_reports: Optional[List[AgentReport]] = None,
    ) -> AgentReport:
        """Synthesize all agent reports.

        Args:
            data: Financial data.
            other_reports: All agent reports to synthesize.

        Returns:
            AgentReport with synthesized findings and investment thesis.
        """
        report = AgentReport(agent_role=self.role)
        findings = []

        if not other_reports:
            findings.append(AgentFinding(
                category="synthesis",
                description="No agent reports available for synthesis",
                severity="warning",
                score=50,
            ))
            report.findings = findings
            report.overall_score = 50
            report.confidence = 0.3
            return report

        # Analyze consensus
        consensus = self._analyze_consensus(other_reports)
        findings.append(AgentFinding(
            category="consensus",
            description=f"Agent consensus: {consensus.bullish_count} bullish, {consensus.bearish_count} bearish",
            severity="info",
            score=int(consensus.consensus_strength * 100),
            evidence=[f"Divergent view: {v}" for v in consensus.divergent_views[:3]],
        ))

        # Identify key themes
        themes = self._identify_key_themes(other_reports)
        for theme in themes[:5]:
            findings.append(AgentFinding(
                category="theme",
                description=theme,
                severity="info",
                score=70,
            ))

        # Identify critical issues
        critical_issues = self._extract_critical_issues(other_reports)
        for issue in critical_issues[:3]:
            findings.append(AgentFinding(
                category="critical_issue",
                description=issue["description"],
                severity=issue["severity"],
                score=issue["score"],
            ))

        # Calculate weighted overall score
        weighted_score = self._calculate_weighted_score(other_reports)
        findings.append(AgentFinding(
            category="overall",
            description=f"Weighted composite score: {weighted_score:.1f}",
            severity="info",
            score=int(weighted_score),
        ))

        report.findings = findings
        report.overall_score = weighted_score
        report.confidence = self._calculate_confidence(consensus, other_reports)

        # Generate investment thesis
        thesis = self._generate_thesis(other_reports, consensus, weighted_score, data)
        report.recommendation = self._format_recommendation(thesis)

        return report

    def synthesize(
        self,
        reports: List[AgentReport],
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentReport:
        """Main synthesis method (alias for analyze).

        Args:
            reports: All agent reports.
            data: Optional financial data.

        Returns:
            Synthesized AgentReport.
        """
        return self.analyze(data or {}, reports)

    def _analyze_consensus(self, reports: List[AgentReport]) -> AgentConsensus:
        """Analyze consensus across agents."""
        consensus = AgentConsensus()

        for report in reports:
            if report.overall_score >= 70:
                consensus.bullish_count += 1
            elif report.overall_score <= 40:
                consensus.bearish_count += 1
            else:
                consensus.neutral_count += 1

        total = len(reports)
        if total > 0:
            # Consensus strength is higher when agents agree
            max_agreement = max(
                consensus.bullish_count,
                consensus.bearish_count,
                consensus.neutral_count
            )
            consensus.consensus_strength = max_agreement / total

        # Identify divergent views
        if consensus.bullish_count > 0 and consensus.bearish_count > 0:
            consensus.divergent_views.append(
                "Significant divergence between bullish and bearish agents"
            )

        return consensus

    def _identify_key_themes(self, reports: List[AgentReport]) -> List[str]:
        """Identify key themes across all reports."""
        themes = []
        theme_counts: Dict[str, int] = {}

        for report in reports:
            for finding in report.findings:
                category = finding.category
                if finding.score >= 70:
                    theme_key = f"Positive: {category}"
                    theme_counts[theme_key] = theme_counts.get(theme_key, 0) + 1
                elif finding.score <= 40:
                    theme_key = f"Concern: {category}"
                    theme_counts[theme_key] = theme_counts.get(theme_key, 0) + 1

        # Sort by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        themes = [theme for theme, count in sorted_themes if count >= 1]

        return themes[:5]

    def _extract_critical_issues(
        self,
        reports: List[AgentReport]
    ) -> List[Dict[str, Any]]:
        """Extract critical issues from all reports."""
        issues = []

        for report in reports:
            for finding in report.findings:
                if finding.severity in ("critical", "concern"):
                    issues.append({
                        "description": f"[{report.agent_role.value}] {finding.description}",
                        "severity": finding.severity,
                        "score": finding.score,
                        "source": report.agent_role.value,
                    })

        # Sort by score (lowest first = most critical)
        issues.sort(key=lambda x: x["score"])
        return issues

    def _calculate_weighted_score(self, reports: List[AgentReport]) -> float:
        """Calculate weighted composite score."""
        if not reports:
            return 50

        total_weight = 0
        weighted_sum = 0

        for report in reports:
            weight = self.AGENT_WEIGHTS.get(report.agent_role, 1.0)
            # Adjust weight by confidence
            adjusted_weight = weight * report.confidence

            weighted_sum += report.overall_score * adjusted_weight
            total_weight += adjusted_weight

        if total_weight == 0:
            return 50

        return weighted_sum / total_weight

    def _calculate_confidence(
        self,
        consensus: AgentConsensus,
        reports: List[AgentReport]
    ) -> float:
        """Calculate confidence in the synthesis."""
        # Base confidence on consensus strength
        base_confidence = consensus.consensus_strength

        # Adjust for number of reports
        report_factor = min(len(reports) / 5, 1.0)  # 5 reports = full confidence

        # Adjust for individual report confidence
        avg_report_confidence = sum(r.confidence for r in reports) / len(reports)

        # Combine factors
        confidence = (base_confidence * 0.4 + report_factor * 0.3 + avg_report_confidence * 0.3)

        # Reduce confidence if there are divergent views
        if consensus.divergent_views:
            confidence *= 0.9

        return min(confidence, 1.0)

    def _generate_thesis(
        self,
        reports: List[AgentReport],
        consensus: AgentConsensus,
        overall_score: float,
        data: Dict[str, Any]
    ) -> InvestmentThesis:
        """Generate investment thesis."""
        # Determine rating
        if overall_score >= 75:
            rating = InvestmentRating.STRONG_BUY
        elif overall_score >= 60:
            rating = InvestmentRating.BUY
        elif overall_score >= 45:
            rating = InvestmentRating.HOLD
        elif overall_score >= 30:
            rating = InvestmentRating.UNDERWEIGHT
        else:
            rating = InvestmentRating.SELL

        # Determine confidence level
        if consensus.consensus_strength >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif consensus.consensus_strength >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        # Extract strengths
        strengths = []
        for report in reports:
            for finding in report.findings:
                if finding.score >= 75 and finding.severity == "info":
                    strengths.append(finding.description)

        # Extract risks
        risks = []
        for report in reports:
            for finding in report.findings:
                if finding.score <= 40 or finding.severity in ("concern", "critical"):
                    risks.append(finding.description)

        # Generate primary thesis
        thesis = InvestmentThesis(
            rating=rating,
            confidence=confidence,
            overall_score=overall_score,
            key_strengths=strengths[:5],
            key_risks=risks[:5],
            primary_thesis=self._generate_primary_thesis(rating, consensus, reports),
            catalysts=self._extract_catalysts(reports),
            concerns=self._extract_concerns(reports),
        )

        return thesis

    def _generate_primary_thesis(
        self,
        rating: InvestmentRating,
        consensus: AgentConsensus,
        reports: List[AgentReport]
    ) -> str:
        """Generate primary thesis statement."""
        if rating == InvestmentRating.STRONG_BUY:
            return (
                "Strong investment opportunity with compelling fundamentals across "
                "multiple analysis dimensions. Low risk profile with significant upside potential."
            )
        elif rating == InvestmentRating.BUY:
            return (
                "Attractive investment opportunity. Fundamentals support positive outlook "
                "with manageable risk profile."
            )
        elif rating == InvestmentRating.HOLD:
            return (
                "Neutral investment case. Fundamentals are sound but lack clear catalysts. "
                "Hold existing positions, wait for better entry point."
            )
        elif rating == InvestmentRating.UNDERWEIGHT:
            return (
                "Elevated risk profile with concerning fundamentals. Consider reducing "
                "exposure or avoiding new positions."
            )
        else:
            return (
                "Unfavorable investment profile. Significant concerns identified across "
                "multiple dimensions. Recommend avoiding or divesting."
            )

    def _extract_catalysts(self, reports: List[AgentReport]) -> List[str]:
        """Extract potential catalysts."""
        catalysts = []

        for report in reports:
            if report.agent_role == AgentRole.GROWTH_ANALYST:
                for finding in report.findings:
                    if "growth" in finding.category and finding.score >= 70:
                        catalysts.append(finding.description)

        return catalysts[:3]

    def _extract_concerns(self, reports: List[AgentReport]) -> List[str]:
        """Extract key concerns."""
        concerns = []

        for report in reports:
            if report.agent_role in (AgentRole.RISK_ANALYST, AgentRole.FORENSIC_ANALYST):
                for finding in report.findings:
                    if finding.severity in ("concern", "critical", "warning"):
                        concerns.append(f"{report.agent_role.value}: {finding.description}")

        return concerns[:5]

    def _format_recommendation(self, thesis: InvestmentThesis) -> str:
        """Format recommendation string."""
        rating_display = thesis.rating.value.replace("_", " ").upper()
        confidence_display = thesis.confidence.value.upper()

        rec = f"INVESTMENT RATING: {rating_display}\n"
        rec += f"Confidence: {confidence_display}\n"
        rec += f"Composite Score: {thesis.overall_score:.1f}/100\n\n"
        rec += f"Thesis: {thesis.primary_thesis}\n\n"

        if thesis.key_strengths:
            rec += "Key Strengths:\n"
            for strength in thesis.key_strengths[:3]:
                rec += f"  • {strength}\n"

        if thesis.key_risks:
            rec += "\nKey Risks:\n"
            for risk in thesis.key_risks[:3]:
                rec += f"  • {risk}\n"

        return rec
