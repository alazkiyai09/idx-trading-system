"""
Prompt Manager Module

Manages prompt templates for LLM interactions.
Uses string.Template for simple, dependency-free templating.
"""

import logging
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Built-in prompt templates for fundamental analysis agents
BUILTIN_TEMPLATES: Dict[str, str] = {
    "auditor_analysis": """You are a financial auditor reviewing the financial statements of an Indonesian publicly listed company.

Analyze the following financial data and provide your assessment:

$financial_data

Focus on:
1. Audit opinion quality and any qualifications
2. Internal control weaknesses
3. Related party transaction concerns
4. Going concern indicators

Respond with valid JSON:
{
    "overall_score": <0-100>,
    "findings": [
        {
            "category": "<category>",
            "description": "<description>",
            "severity": "<info|warning|concern|critical>",
            "score": <0-100>
        }
    ],
    "recommendation": "<your recommendation>",
    "confidence": <0.0-1.0>
}""",

    "growth_analysis": """You are a growth-focused equity analyst reviewing an Indonesian publicly listed company.

Analyze the following financial data with an optimistic, forward-looking perspective:

$financial_data

Focus on:
1. Revenue growth trajectory and sustainability
2. Market share gains and competitive positioning
3. New product/market opportunities
4. Operating leverage potential
5. Capital allocation efficiency

Respond with valid JSON:
{
    "overall_score": <0-100>,
    "findings": [
        {
            "category": "<category>",
            "description": "<description>",
            "severity": "<info|warning|concern|critical>",
            "score": <0-100>
        }
    ],
    "recommendation": "<your recommendation>",
    "confidence": <0.0-1.0>
}""",

    "value_analysis": """You are a value-focused equity analyst reviewing an Indonesian publicly listed company.

Analyze the following financial data with a conservative, value-oriented perspective:

$financial_data

Focus on:
1. Price-to-earnings and price-to-book relative to sector
2. Free cash flow yield and dividend sustainability
3. Asset quality and balance sheet strength
4. Margin of safety assessment
5. Downside risk scenarios

Respond with valid JSON:
{
    "overall_score": <0-100>,
    "findings": [
        {
            "category": "<category>",
            "description": "<description>",
            "severity": "<info|warning|concern|critical>",
            "score": <0-100>
        }
    ],
    "recommendation": "<your recommendation>",
    "confidence": <0.0-1.0>
}""",

    "risk_analysis": """You are a risk analyst reviewing an Indonesian publicly listed company.

Analyze the following financial data with a pessimistic, risk-focused perspective:

$financial_data

Focus on:
1. Debt maturity profile and refinancing risk
2. Currency exposure and hedging adequacy
3. Regulatory and political risks
4. Concentration risks (customer, supplier, geography)
5. Liquidity stress scenarios
6. ESG and governance red flags

Respond with valid JSON:
{
    "overall_score": <0-100>,
    "findings": [
        {
            "category": "<category>",
            "description": "<description>",
            "severity": "<info|warning|concern|critical>",
            "score": <0-100>
        }
    ],
    "recommendation": "<your recommendation>",
    "confidence": <0.0-1.0>
}""",

    "forensic_analysis": """You are a forensic accountant reviewing the financial statements of an Indonesian publicly listed company.

Analyze the following financial data for signs of manipulation or fraud:

$financial_data

Focus on:
1. Revenue recognition irregularities
2. Unusual accrual patterns
3. Cash flow vs. earnings divergence
4. Related party transaction abuse
5. Inventory and receivable manipulation
6. Off-balance-sheet exposures
7. Benford's Law compliance of reported figures

Respond with valid JSON:
{
    "overall_score": <0-100>,
    "findings": [
        {
            "category": "<category>",
            "description": "<description>",
            "severity": "<info|warning|concern|critical>",
            "score": <0-100>
        }
    ],
    "recommendation": "<your recommendation>",
    "confidence": <0.0-1.0>
}""",

    "synthesis": """You are the Chief Investment Officer synthesizing multiple analyst reports on an Indonesian publicly listed company.

Previous analyst reports:
$analyst_reports

Provide a final synthesis that:
1. Weighs each analyst's perspective appropriately
2. Resolves any conflicts between analysts
3. Applies veto rules (if authenticity_score < 60, automatically reject)
4. Provides final investment recommendation
5. Assigns confidence level to the recommendation

Respond with valid JSON:
{
    "overall_score": <0-100>,
    "recommendation": "<STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL>",
    "confidence": <0.0-1.0>,
    "key_strengths": ["<strength1>", "<strength2>"],
    "key_risks": ["<risk1>", "<risk2>"],
    "veto_triggered": <true|false>,
    "veto_reason": "<reason if veto triggered>",
    "analyst_agreement": <0.0-1.0>,
    "summary": "<executive summary>"
}""",

    "sentiment_analysis": """Analyze the sentiment of the following news article about an Indonesian publicly listed company ($symbol):

Title: $title
Source: $source
Content: $content

Rate the sentiment from 0 (extremely negative) to 100 (extremely positive).
Consider the impact on stock price.

Respond with valid JSON:
{
    "sentiment_score": <0-100>,
    "confidence": <0.0-1.0>,
    "key_topics": ["<topic1>", "<topic2>"],
    "impact_assessment": "<description of likely market impact>",
    "time_horizon": "<short_term|medium_term|long_term>"
}""",
}


class PromptManager:
    """Manages prompt templates for LLM interactions.

    Supports built-in templates and custom templates loaded from files.

    Example:
        pm = PromptManager()
        prompt = pm.render("auditor_analysis", financial_data="...")
    """

    def __init__(self, templates_dir: Optional[Path] = None) -> None:
        """Initialize prompt manager.

        Args:
            templates_dir: Optional directory for custom prompt templates.
        """
        self._templates: Dict[str, str] = dict(BUILTIN_TEMPLATES)
        self._templates_dir = templates_dir

        if templates_dir and templates_dir.exists():
            self._load_custom_templates(templates_dir)

    def _load_custom_templates(self, templates_dir: Path) -> None:
        """Load custom templates from a directory.

        Args:
            templates_dir: Directory containing .txt template files.
        """
        for template_file in templates_dir.glob("*.txt"):
            name = template_file.stem
            try:
                content = template_file.read_text(encoding="utf-8")
                self._templates[name] = content
                logger.info(f"Loaded custom template: {name}")
            except Exception as e:
                logger.warning(f"Failed to load template {name}: {e}")

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a prompt template with variables.

        Args:
            template_name: Name of the template.
            **kwargs: Template variables.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If template not found.
        """
        if template_name not in self._templates:
            available = ", ".join(sorted(self._templates.keys()))
            raise KeyError(
                f"Template '{template_name}' not found. "
                f"Available templates: {available}"
            )

        template = Template(self._templates[template_name])
        try:
            return template.safe_substitute(**kwargs)
        except Exception as e:
            logger.error(f"Failed to render template '{template_name}': {e}")
            raise

    def register(self, name: str, template: str) -> None:
        """Register a new template.

        Args:
            name: Template name.
            template: Template string with $variable placeholders.
        """
        self._templates[name] = template
        logger.debug(f"Registered template: {name}")

    def list_templates(self) -> list:
        """List available template names.

        Returns:
            Sorted list of template names.
        """
        return sorted(self._templates.keys())

    def get_template(self, name: str) -> str:
        """Get raw template string.

        Args:
            name: Template name.

        Returns:
            Raw template string.

        Raises:
            KeyError: If template not found.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found.")
        return self._templates[name]
