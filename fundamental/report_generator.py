"""
Report Generator Module

Generates comprehensive fundamental analysis reports:
- Markdown reports for human review
- Structured data for system integration
- Summary reports for quick reference
- Detailed reports for deep analysis
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .agents.base import AgentReport, AgentRole
from .agents.synthesizer import InvestmentThesis, InvestmentRating

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output format."""
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class ReportType(Enum):
    """Type of report to generate."""
    FULL = "full"
    SUMMARY = "summary"
    EXECUTIVE = "executive"


@dataclass
class FundamentalReport:
    """Complete fundamental analysis report."""
    ticker: str
    company_name: str
    analysis_date: datetime
    thesis: Optional[InvestmentThesis] = None
    agent_reports: List[AgentReport] = field(default_factory=list)
    financial_summary: Dict[str, Any] = field(default_factory=dict)
    ratio_summary: Dict[str, Any] = field(default_factory=dict)
    fraud_analysis: Dict[str, Any] = field(default_factory=dict)
    key_findings: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "analysis_date": self.analysis_date.isoformat(),
            "thesis": {
                "rating": self.thesis.rating.value if self.thesis else None,
                "confidence": self.thesis.confidence.value if self.thesis else None,
                "overall_score": self.thesis.overall_score if self.thesis else None,
                "primary_thesis": self.thesis.primary_thesis if self.thesis else None,
            } if self.thesis else None,
            "financial_summary": self.financial_summary,
            "ratio_summary": self.ratio_summary,
            "fraud_analysis": self.fraud_analysis,
            "key_findings": self.key_findings,
            "red_flags": self.red_flags,
            "recommendations": self.recommendations,
        }


class ReportGenerator:
    """Generates fundamental analysis reports.

    Creates various report formats from analysis results:
    - Executive summaries for quick decisions
    - Full reports for detailed review
    - JSON exports for system integration

    Example:
        generator = ReportGenerator()
        report = generator.generate_full_report(
            ticker="BBCA",
            company_name="Bank Central Asia",
            thesis=thesis,
            agent_reports=reports,
            financial_data=financial_data,
        )
        print(generator.to_markdown(report))
    """

    def __init__(self) -> None:
        """Initialize report generator."""

    def generate_full_report(
        self,
        ticker: str,
        company_name: str,
        thesis: InvestmentThesis,
        agent_reports: List[AgentReport],
        financial_data: Optional[Dict[str, Any]] = None,
        ratio_analysis: Optional[Dict[str, Any]] = None,
        fraud_analysis: Optional[Dict[str, Any]] = None,
    ) -> FundamentalReport:
        """Generate a complete fundamental analysis report.

        Args:
            ticker: Stock ticker symbol.
            company_name: Full company name.
            thesis: Investment thesis from synthesizer.
            agent_reports: All agent analysis reports.
            financial_data: Extracted financial data.
            ratio_analysis: Ratio analysis results.
            fraud_analysis: Fraud detection results.

        Returns:
            FundamentalReport with complete analysis.
        """
        report = FundamentalReport(
            ticker=ticker,
            company_name=company_name,
            analysis_date=datetime.now(),
            thesis=thesis,
            agent_reports=agent_reports,
            financial_summary=self._summarize_financials(financial_data),
            ratio_summary=ratio_analysis or {},
            fraud_analysis=fraud_analysis or {},
        )

        # Extract key findings
        report.key_findings = self._extract_key_findings(agent_reports)

        # Extract red flags
        report.red_flags = self._extract_red_flags(agent_reports, fraud_analysis)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(thesis, agent_reports)

        return report

    def generate_summary_report(
        self,
        ticker: str,
        company_name: str,
        thesis: InvestmentThesis,
        agent_reports: List[AgentReport],
    ) -> FundamentalReport:
        """Generate a summary report for quick reference.

        Args:
            ticker: Stock ticker symbol.
            company_name: Full company name.
            thesis: Investment thesis.
            agent_reports: Agent reports.

        Returns:
            Summary FundamentalReport.
        """
        report = FundamentalReport(
            ticker=ticker,
            company_name=company_name,
            analysis_date=datetime.now(),
            thesis=thesis,
            agent_reports=agent_reports,
        )

        report.key_findings = self._extract_key_findings(agent_reports)[:5]
        report.red_flags = self._extract_red_flags(agent_reports, None)[:3]

        return report

    def to_markdown(
        self,
        report: FundamentalReport,
        report_type: ReportType = ReportType.FULL
    ) -> str:
        """Convert report to markdown format.

        Args:
            report: Fundamental report.
            report_type: Type of report to generate.

        Returns:
            Markdown formatted string.
        """
        if report_type == ReportType.EXECUTIVE:
            return self._generate_executive_markdown(report)
        elif report_type == ReportType.SUMMARY:
            return self._generate_summary_markdown(report)
        else:
            return self._generate_full_markdown(report)

    def to_json(self, report: FundamentalReport) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary.

        Args:
            report: Fundamental report.

        Returns:
            Dictionary representation.
        """
        return report.to_dict()

    def _generate_full_markdown(self, report: FundamentalReport) -> str:
        """Generate full markdown report."""
        md = []

        # Header
        md.append(f"# Fundamental Analysis: {report.ticker}")
        md.append(f"## {report.company_name}")
        md.append(f"\n**Analysis Date:** {report.analysis_date.strftime('%Y-%m-%d %H:%M')}")
        md.append("")

        # Investment Thesis
        if report.thesis:
            md.append("## Investment Thesis")
            md.append("")
            md.append(f"**Rating:** {report.thesis.rating.value.replace('_', ' ').upper()}")
            md.append(f"**Confidence:** {report.thesis.confidence.value.upper()}")
            md.append(f"**Overall Score:** {report.thesis.overall_score:.1f}/100")
            md.append("")
            md.append(f"*{report.thesis.primary_thesis}*")
            md.append("")

            # Strengths and Risks
            if report.thesis.key_strengths:
                md.append("### Key Strengths")
                for strength in report.thesis.key_strengths:
                    md.append(f"- {strength}")
                md.append("")

            if report.thesis.key_risks:
                md.append("### Key Risks")
                for risk in report.thesis.key_risks:
                    md.append(f"- {risk}")
                md.append("")

        # Financial Summary
        if report.financial_summary:
            md.append("## Financial Summary")
            md.append(self._format_financial_summary(report.financial_summary))
            md.append("")

        # Ratio Analysis
        if report.ratio_summary:
            md.append("## Ratio Analysis")
            md.append(self._format_ratio_summary(report.ratio_summary))
            md.append("")

        # Fraud Analysis
        if report.fraud_analysis:
            md.append("## Fraud Analysis")
            md.append(self._format_fraud_analysis(report.fraud_analysis))
            md.append("")

        # Agent Reports
        md.append("## Detailed Analysis")
        for agent_report in report.agent_reports:
            md.append(self._format_agent_report(agent_report))
            md.append("")

        # Red Flags
        if report.red_flags:
            md.append("## Red Flags")
            for flag in report.red_flags:
                md.append(f"- {flag}")
            md.append("")

        # Recommendations
        if report.recommendations:
            md.append("## Recommendations")
            for rec in report.recommendations:
                md.append(f"- {rec}")
            md.append("")

        # Footer
        md.append("---")
        md.append("*This report was generated by the IDX Trading System Fundamental Analysis Module.*")

        return "\n".join(md)

    def _generate_summary_markdown(self, report: FundamentalReport) -> str:
        """Generate summary markdown report."""
        md = []

        md.append(f"# {report.ticker} - Analysis Summary")
        md.append("")

        if report.thesis:
            rating_emoji = self._get_rating_emoji(report.thesis.rating)
            md.append(f"**{rating_emoji} Rating: {report.thesis.rating.value.replace('_', ' ').upper()}**")
            md.append(f"Score: {report.thesis.overall_score:.1f}/100 | Confidence: {report.thesis.confidence.value.upper()}")
            md.append("")

        if report.key_findings:
            md.append("### Key Findings")
            for finding in report.key_findings[:5]:
                md.append(f"- {finding}")
            md.append("")

        if report.red_flags:
            md.append("### Red Flags")
            for flag in report.red_flags[:3]:
                md.append(f"- {flag}")

        return "\n".join(md)

    def _generate_executive_markdown(self, report: FundamentalReport) -> str:
        """Generate executive summary markdown."""
        md = []

        # One-line summary
        if report.thesis:
            rating = report.thesis.rating.value.replace('_', ' ').upper()
            score = report.thesis.overall_score
            md.append(f"## {report.ticker}: {rating} ({score:.0f}/100)")
            md.append("")
            md.append(report.thesis.primary_thesis)
            md.append("")

            # Quick bullets
            md.append("**Positives:**")
            for s in report.thesis.key_strengths[:2]:
                md.append(f"- {s}")
            md.append("")

            md.append("**Concerns:**")
            for r in report.thesis.key_risks[:2]:
                md.append(f"- {r}")

        return "\n".join(md)

    def _summarize_financials(
        self,
        financial_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary of financial data."""
        if not financial_data:
            return {}

        summary = {}

        income = financial_data.get("income_statement", {})
        if income:
            summary["revenue"] = income.get("revenue")
            summary["net_income"] = income.get("net_income")
            summary["operating_income"] = income.get("operating_income")

        balance = financial_data.get("balance_sheet", {})
        if balance:
            summary["total_assets"] = balance.get("total_assets")
            summary["total_equity"] = balance.get("total_equity")
            summary["total_liabilities"] = balance.get("total_liabilities")

        cashflow = financial_data.get("cash_flow", {})
        if cashflow:
            summary["operating_cash_flow"] = cashflow.get("operating_cash_flow")
            summary["free_cash_flow"] = cashflow.get("free_cash_flow")

        return summary

    def _extract_key_findings(
        self,
        agent_reports: List[AgentReport]
    ) -> List[str]:
        """Extract key findings from all agent reports."""
        findings = []

        for report in agent_reports:
            for finding in report.findings:
                if finding.score >= 70 or finding.severity == "info":
                    findings.append(f"[{report.agent_role.value}] {finding.description}")

        return findings

    def _extract_red_flags(
        self,
        agent_reports: List[AgentReport],
        fraud_analysis: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract red flags from reports."""
        flags = []

        # From agent reports
        for report in agent_reports:
            for finding in report.findings:
                if finding.severity in ("critical", "concern"):
                    flags.append(f"[{report.agent_role.value}] {finding.description}")

        # From fraud analysis
        if fraud_analysis:
            for indicator in fraud_analysis.get("indicators", []):
                flags.append(f"[Fraud] {indicator.get('description', '')}")

        return flags

    def _generate_recommendations(
        self,
        thesis: InvestmentThesis,
        agent_reports: List[AgentReport]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Based on rating
        if thesis.rating in (InvestmentRating.STRONG_BUY, InvestmentRating.BUY):
            recommendations.append("Consider establishing or increasing position")
            if thesis.key_strengths:
                recommendations.append(
                    f"Key catalyst: {thesis.key_strengths[0]}"
                )
        elif thesis.rating == InvestmentRating.HOLD:
            recommendations.append("Hold existing positions")
            recommendations.append("Monitor for changes in key metrics")
        elif thesis.rating == InvestmentRating.UNDERWEIGHT:
            recommendations.append("Consider reducing exposure")
            if thesis.key_risks:
                recommendations.append(
                    f"Primary concern: {thesis.key_risks[0]}"
                )
        else:  # SELL
            recommendations.append("Consider divesting position")
            recommendations.append("Wait for improved fundamentals before re-entry")

        # From agent recommendations
        for report in agent_reports:
            if report.recommendation and report.agent_role == AgentRole.FORENSIC_ANALYST:
                recommendations.append(f"Forensic note: {report.recommendation[:100]}")

        return recommendations[:5]

    def _format_financial_summary(
        self,
        summary: Dict[str, Any]
    ) -> str:
        """Format financial summary as markdown table."""
        lines = []
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        for key, value in summary.items():
            if value is not None:
                formatted_value = self._format_currency(value)
                lines.append(f"| {key.replace('_', ' ').title()} | {formatted_value} |")

        return "\n".join(lines)

    def _format_ratio_summary(
        self,
        ratios: Dict[str, Any]
    ) -> str:
        """Format ratio analysis as markdown."""
        lines = []

        for category, data in ratios.items():
            if isinstance(data, dict):
                lines.append(f"### {category.replace('_', ' ').title()}")
                lines.append("")
                for ratio_name, ratio_data in data.items():
                    if isinstance(ratio_data, dict):
                        value = ratio_data.get("value", "N/A")
                        interpretation = ratio_data.get("interpretation", "")
                        lines.append(f"- **{ratio_name}:** {value} - {interpretation}")
                lines.append("")

        return "\n".join(lines)

    def _format_fraud_analysis(
        self,
        fraud: Dict[str, Any]
    ) -> str:
        """Format fraud analysis as markdown."""
        lines = []

        risk_level = fraud.get("overall_risk", "unknown")
        fraud_prob = fraud.get("fraud_probability", 0)

        lines.append(f"**Overall Risk Level:** {risk_level.upper()}")
        lines.append(f"**Fraud Probability:** {fraud_prob:.1%}")
        lines.append("")

        red_flags = fraud.get("red_flags", [])
        if red_flags:
            lines.append("### Red Flags Detected")
            for flag in red_flags:
                lines.append(f"- {flag}")

        return "\n".join(lines)

    def _format_agent_report(
        self,
        report: AgentReport
    ) -> str:
        """Format individual agent report as markdown."""
        lines = []

        role_name = report.agent_role.value.replace('_', ' ').title()
        lines.append(f"### {role_name}")
        lines.append(f"**Score:** {report.overall_score:.1f} | **Confidence:** {report.confidence:.0%}")
        lines.append("")

        if report.findings:
            lines.append("| Category | Finding | Severity |")
            lines.append("|----------|---------|----------|")
            for finding in report.findings:
                lines.append(
                    f"| {finding.category} | {finding.description[:50]}... | {finding.severity} |"
                )
            lines.append("")

        if report.recommendation:
            lines.append(f"*{report.recommendation}*")

        return "\n".join(lines)

    def _format_currency(self, value: float) -> str:
        """Format currency value with appropriate scale."""
        abs_value = abs(value)

        if abs_value >= 1_000_000_000_000:  # Trillions
            return f"Rp {value / 1_000_000_000_000:.2f}T"
        elif abs_value >= 1_000_000_000:  # Billions
            return f"Rp {value / 1_000_000_000:.2f}B"
        elif abs_value >= 1_000_000:  # Millions
            return f"Rp {value / 1_000_000:.2f}M"
        else:
            return f"Rp {value:,.0f}"

    def _get_rating_emoji(self, rating: InvestmentRating) -> str:
        """Get emoji for investment rating."""
        emojis = {
            InvestmentRating.STRONG_BUY: "",
            InvestmentRating.BUY: "",
            InvestmentRating.HOLD: "",
            InvestmentRating.UNDERWEIGHT: "",
            InvestmentRating.SELL: "",
        }
        return emojis.get(rating, "")
