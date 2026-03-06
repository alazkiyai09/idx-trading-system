"""
Ratio Calculator Module

Calculates comprehensive financial ratios from financial statements:
- Profitability ratios (ROE, ROA, margins)
- Liquidity ratios (current, quick)
- Leverage ratios (D/E, interest coverage)
- Efficiency ratios (receivable days, inventory days)
- Valuation ratios (P/E, P/B, EV/EBITDA)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class RatioCategory(Enum):
    """Categories of financial ratios."""
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"
    EFFICIENCY = "efficiency"
    VALUATION = "valuation"
    GROWTH = "growth"


@dataclass
class RatioResult:
    """Result of a single ratio calculation.

    Attributes:
        name: Ratio name.
        value: Calculated value.
        category: Ratio category.
        benchmark: Industry benchmark (if available).
        interpretation: Interpretation of the value.
    """

    name: str
    value: float
    category: RatioCategory
    benchmark: Optional[float] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "category": self.category.value,
            "benchmark": self.benchmark,
            "interpretation": self.interpretation,
        }


@dataclass
class RatioAnalysis:
    """Complete ratio analysis result.

    Attributes:
        ratios: List of calculated ratios.
        by_category: Ratios grouped by category.
        strengths: Identified strengths.
        weaknesses: Identified weaknesses.
        summary: Analysis summary.
    """

    ratios: List[RatioResult] = field(default_factory=list)
    by_category: Dict[str, List[RatioResult]] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ratios": [r.to_dict() for r in self.ratios],
            "by_category": {
                k: [r.to_dict() for r in v]
                for k, v in self.by_category.items()
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "summary": self.summary,
        }


class RatioCalculator:
    """Calculates financial ratios from statement data.

    Example:
        calculator = RatioCalculator()
        analysis = calculator.calculate(
            income_statement={...},
            balance_sheet={...},
            cash_flow={...},
            market_data={...},
        )
        for ratio in analysis.ratios:
            print(f"{ratio.name}: {ratio.value:.2f}")
    """

    # Industry benchmarks for Indonesian companies (approximate)
    BENCHMARKS = {
        "roe": 0.15,  # 15%
        "roa": 0.08,  # 8%
        "gross_margin": 0.30,  # 30%
        "operating_margin": 0.15,  # 15%
        "net_margin": 0.10,  # 10%
        "current_ratio": 1.5,
        "quick_ratio": 1.0,
        "debt_to_equity": 1.0,
        "interest_coverage": 3.0,
        "asset_turnover": 0.8,
        "inventory_turnover": 6.0,
        "receivable_turnover": 8.0,
    }

    def __init__(self) -> None:
        """Initialize ratio calculator."""
        pass

    def calculate(
        self,
        income_statement: Dict[str, Any],
        balance_sheet: Dict[str, Any],
        cash_flow: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        prior_year: Optional[Dict[str, Any]] = None,
    ) -> RatioAnalysis:
        """Calculate all financial ratios.

        Args:
            income_statement: Income statement data.
            balance_sheet: Balance sheet data.
            cash_flow: Cash flow statement data.
            market_data: Market data (price, shares, etc.).
            prior_year: Prior year data for growth calculations.

        Returns:
            RatioAnalysis with all ratios and interpretation.
        """
        analysis = RatioAnalysis()

        # Calculate each category
        self._calculate_profitability(income_statement, balance_sheet, analysis)
        self._calculate_liquidity(balance_sheet, analysis)
        self._calculate_leverage(income_statement, balance_sheet, analysis)
        self._calculate_efficiency(income_statement, balance_sheet, analysis)

        if market_data:
            self._calculate_valuation(income_statement, balance_sheet, market_data, analysis)

        if prior_year:
            self._calculate_growth(income_statement, balance_sheet, prior_year, analysis)

        # Group by category
        for ratio in analysis.ratios:
            cat = ratio.category.value
            if cat not in analysis.by_category:
                analysis.by_category[cat] = []
            analysis.by_category[cat].append(ratio)

        # Generate interpretations
        self._analyze_strengths_weaknesses(analysis)
        analysis.summary = self._generate_summary(analysis)

        return analysis

    def _calculate_profitability(
        self,
        income: Dict[str, Any],
        balance: Dict[str, Any],
        analysis: RatioAnalysis,
    ) -> None:
        """Calculate profitability ratios."""
        revenue = income.get("revenue", 0)
        gross_profit = income.get("gross_profit", 0)
        operating_income = income.get("operating_income", 0)
        net_income = income.get("net_income", 0)

        total_assets = balance.get("total_assets", 0)
        total_equity = balance.get("total_equity", 0)

        # Gross Margin
        if revenue > 0:
            gross_margin = gross_profit / revenue
            analysis.ratios.append(RatioResult(
                name="Gross Margin",
                value=gross_margin,
                category=RatioCategory.PROFITABILITY,
                benchmark=self.BENCHMARKS.get("gross_margin"),
                interpretation=self._interpret_margin("gross", gross_margin, 0.30),
            ))

        # Operating Margin
        if revenue > 0:
            operating_margin = operating_income / revenue
            analysis.ratios.append(RatioResult(
                name="Operating Margin",
                value=operating_margin,
                category=RatioCategory.PROFITABILITY,
                benchmark=self.BENCHMARKS.get("operating_margin"),
                interpretation=self._interpret_margin("operating", operating_margin, 0.15),
            ))

        # Net Margin
        if revenue > 0:
            net_margin = net_income / revenue
            analysis.ratios.append(RatioResult(
                name="Net Margin",
                value=net_margin,
                category=RatioCategory.PROFITABILITY,
                benchmark=self.BENCHMARKS.get("net_margin"),
                interpretation=self._interpret_margin("net", net_margin, 0.10),
            ))

        # ROE (Return on Equity)
        if total_equity > 0:
            roe = net_income / total_equity
            analysis.ratios.append(RatioResult(
                name="ROE",
                value=roe,
                category=RatioCategory.PROFITABILITY,
                benchmark=self.BENCHMARKS.get("roe"),
                interpretation=self._interpret_roe(roe),
            ))

        # ROA (Return on Assets)
        if total_assets > 0:
            roa = net_income / total_assets
            analysis.ratios.append(RatioResult(
                name="ROA",
                value=roa,
                category=RatioCategory.PROFITABILITY,
                benchmark=self.BENCHMARKS.get("roa"),
                interpretation=self._interpret_roa(roa),
            ))

    def _calculate_liquidity(
        self,
        balance: Dict[str, Any],
        analysis: RatioAnalysis,
    ) -> None:
        """Calculate liquidity ratios."""
        current_assets = balance.get("current_assets", 0)
        current_liabilities = balance.get("current_liabilities", 0)
        inventory = balance.get("inventory", 0)
        cash = balance.get("cash", 0)

        # Current Ratio
        if current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            analysis.ratios.append(RatioResult(
                name="Current Ratio",
                value=current_ratio,
                category=RatioCategory.LIQUIDITY,
                benchmark=self.BENCHMARKS.get("current_ratio"),
                interpretation=self._interpret_current_ratio(current_ratio),
            ))

        # Quick Ratio (Acid Test)
        if current_liabilities > 0:
            quick_assets = current_assets - inventory
            quick_ratio = quick_assets / current_liabilities
            analysis.ratios.append(RatioResult(
                name="Quick Ratio",
                value=quick_ratio,
                category=RatioCategory.LIQUIDITY,
                benchmark=self.BENCHMARKS.get("quick_ratio"),
                interpretation=self._interpret_quick_ratio(quick_ratio),
            ))

        # Cash Ratio
        if current_liabilities > 0:
            cash_ratio = cash / current_liabilities
            analysis.ratios.append(RatioResult(
                name="Cash Ratio",
                value=cash_ratio,
                category=RatioCategory.LIQUIDITY,
                interpretation=f"Cash covers {cash_ratio:.0%} of current liabilities",
            ))

    def _calculate_leverage(
        self,
        income: Dict[str, Any],
        balance: Dict[str, Any],
        analysis: RatioAnalysis,
    ) -> None:
        """Calculate leverage ratios."""
        total_assets = balance.get("total_assets", 0)
        total_liabilities = balance.get("total_liabilities", 0)
        total_equity = balance.get("total_equity", 0)
        interest_expense = income.get("interest_expense", 0)
        ebit = income.get("operating_income", 0)  # Approximation

        # Debt to Equity
        if total_equity > 0:
            debt_to_equity = total_liabilities / total_equity
            analysis.ratios.append(RatioResult(
                name="Debt to Equity",
                value=debt_to_equity,
                category=RatioCategory.LEVERAGE,
                benchmark=self.BENCHMARKS.get("debt_to_equity"),
                interpretation=self._interpret_debt_equity(debt_to_equity),
            ))

        # Debt to Assets
        if total_assets > 0:
            debt_to_assets = total_liabilities / total_assets
            analysis.ratios.append(RatioResult(
                name="Debt to Assets",
                value=debt_to_assets,
                category=RatioCategory.LEVERAGE,
                interpretation=f"{debt_to_assets:.0%} of assets financed by debt",
            ))

        # Interest Coverage
        if interest_expense > 0:
            interest_coverage = ebit / interest_expense
            analysis.ratios.append(RatioResult(
                name="Interest Coverage",
                value=interest_coverage,
                category=RatioCategory.LEVERAGE,
                benchmark=self.BENCHMARKS.get("interest_coverage"),
                interpretation=self._interpret_interest_coverage(interest_coverage),
            ))

        # Equity Multiplier
        if total_equity > 0:
            equity_multiplier = total_assets / total_equity
            analysis.ratios.append(RatioResult(
                name="Equity Multiplier",
                value=equity_multiplier,
                category=RatioCategory.LEVERAGE,
                interpretation=f"Assets are {equity_multiplier:.1f}x equity",
            ))

    def _calculate_efficiency(
        self,
        income: Dict[str, Any],
        balance: Dict[str, Any],
        analysis: RatioAnalysis,
    ) -> None:
        """Calculate efficiency ratios."""
        revenue = income.get("revenue", 0)
        cogs = income.get("cost_of_goods_sold", 0)
        total_assets = balance.get("total_assets", 0)
        inventory = balance.get("inventory", 0)
        receivables = balance.get("accounts_receivable", 0)
        payables = balance.get("accounts_payable", 0)

        # Asset Turnover
        if total_assets > 0:
            asset_turnover = revenue / total_assets
            analysis.ratios.append(RatioResult(
                name="Asset Turnover",
                value=asset_turnover,
                category=RatioCategory.EFFICIENCY,
                benchmark=self.BENCHMARKS.get("asset_turnover"),
                interpretation=self._interpret_asset_turnover(asset_turnover),
            ))

        # Inventory Turnover
        if inventory > 0:
            inventory_turnover = cogs / inventory
            days_inventory = 365 / inventory_turnover if inventory_turnover > 0 else 0
            analysis.ratios.append(RatioResult(
                name="Inventory Turnover",
                value=inventory_turnover,
                category=RatioCategory.EFFICIENCY,
                benchmark=self.BENCHMARKS.get("inventory_turnover"),
                interpretation=f"Inventory sold {inventory_turnover:.1f}x per year ({days_inventory:.0f} days)",
            ))

        # Receivables Turnover
        if receivables > 0:
            receivables_turnover = revenue / receivables
            days_receivable = 365 / receivables_turnover if receivables_turnover > 0 else 0
            analysis.ratios.append(RatioResult(
                name="Receivables Turnover",
                value=receivables_turnover,
                category=RatioCategory.EFFICIENCY,
                benchmark=self.BENCHMARKS.get("receivable_turnover"),
                interpretation=f"Receivables collected {receivables_turnover:.1f}x per year ({days_receivable:.0f} days)",
            ))

        # Payables Turnover
        if payables > 0:
            payables_turnover = cogs / payables
            days_payable = 365 / payables_turnover if payables_turnover > 0 else 0
            analysis.ratios.append(RatioResult(
                name="Payables Turnover",
                value=payables_turnover,
                category=RatioCategory.EFFICIENCY,
                interpretation=f"Payables paid {payables_turnover:.1f}x per year ({days_payable:.0f} days)",
            ))

    def _calculate_valuation(
        self,
        income: Dict[str, Any],
        balance: Dict[str, Any],
        market: Dict[str, Any],
        analysis: RatioAnalysis,
    ) -> None:
        """Calculate valuation ratios."""
        net_income = income.get("net_income", 0)
        total_equity = balance.get("total_equity", 0)

        market_cap = market.get("market_cap", 0)
        share_price = market.get("share_price", 0)
        shares_outstanding = market.get("shares_outstanding", 0)
        book_value = market.get("book_value", total_equity)
        ebitda = market.get("ebitda", income.get("operating_income", 0))
        enterprise_value = market.get("enterprise_value", market_cap)

        # P/E Ratio
        if net_income > 0 and shares_outstanding > 0:
            eps = net_income / shares_outstanding
            if eps > 0:
                pe_ratio = share_price / eps
                analysis.ratios.append(RatioResult(
                    name="P/E Ratio",
                    value=pe_ratio,
                    category=RatioCategory.VALUATION,
                    interpretation=self._interpret_pe(pe_ratio),
                ))

        # P/B Ratio
        if book_value > 0 and shares_outstanding > 0:
            book_per_share = book_value / shares_outstanding
            if book_per_share > 0:
                pb_ratio = share_price / book_per_share
                analysis.ratios.append(RatioResult(
                    name="P/B Ratio",
                    value=pb_ratio,
                    category=RatioCategory.VALUATION,
                    interpretation=self._interpret_pb(pb_ratio),
                ))

        # EV/EBITDA
        if ebitda > 0:
            ev_ebitda = enterprise_value / ebitda
            analysis.ratios.append(RatioResult(
                name="EV/EBITDA",
                value=ev_ebitda,
                category=RatioCategory.VALUATION,
                interpretation=self._interpret_ev_ebitda(ev_ebitda),
            ))

    def _calculate_growth(
        self,
        income: Dict[str, Any],
        balance: Dict[str, Any],
        prior: Dict[str, Any],
        analysis: RatioAnalysis,
    ) -> None:
        """Calculate growth ratios."""
        current_revenue = income.get("revenue", 0)
        prior_revenue = prior.get("income_statement", {}).get("revenue", 0)

        current_net_income = income.get("net_income", 0)
        prior_net_income = prior.get("income_statement", {}).get("net_income", 0)

        # Revenue Growth
        if prior_revenue > 0:
            revenue_growth = (current_revenue - prior_revenue) / prior_revenue
            analysis.ratios.append(RatioResult(
                name="Revenue Growth",
                value=revenue_growth,
                category=RatioCategory.GROWTH,
                interpretation=self._interpret_growth("revenue", revenue_growth),
            ))

        # Net Income Growth
        if prior_net_income > 0:
            income_growth = (current_net_income - prior_net_income) / prior_net_income
            analysis.ratios.append(RatioResult(
                name="Net Income Growth",
                value=income_growth,
                category=RatioCategory.GROWTH,
                interpretation=self._interpret_growth("net income", income_growth),
            ))

    def _analyze_strengths_weaknesses(self, analysis: RatioAnalysis) -> None:
        """Identify strengths and weaknesses from ratios."""
        for ratio in analysis.ratios:
            if ratio.benchmark is None:
                continue

            if ratio.value > ratio.benchmark * 1.2:
                if ratio.category in (RatioCategory.PROFITABILITY, RatioCategory.LIQUIDITY):
                    analysis.strengths.append(f"Strong {ratio.name} ({ratio.value:.2f})")
            elif ratio.value < ratio.benchmark * 0.8:
                if ratio.category in (RatioCategory.LEVERAGE,):
                    analysis.strengths.append(f"Conservative {ratio.name} ({ratio.value:.2f})")
                else:
                    analysis.weaknesses.append(f"Low {ratio.name} ({ratio.value:.2f})")

            # Specific checks
            if ratio.name == "Current Ratio" and ratio.value < 1.0:
                analysis.weaknesses.append("Potential liquidity concern (current ratio < 1)")
            if ratio.name == "Debt to Equity" and ratio.value > 2.0:
                analysis.weaknesses.append(f"High leverage (D/E = {ratio.value:.1f})")
            if ratio.name == "ROE" and ratio.value < 0.10:
                analysis.weaknesses.append(f"Low return on equity ({ratio.value:.0%})")

    def _generate_summary(self, analysis: RatioAnalysis) -> str:
        """Generate analysis summary."""
        lines = [
            "FINANCIAL RATIO ANALYSIS",
            "=" * 40,
            "",
        ]

        for category, ratios in analysis.by_category.items():
            lines.append(f"{category.upper()}:")
            for r in ratios:
                lines.append(f"  {r.name}: {r.value:.2f}")
            lines.append("")

        if analysis.strengths:
            lines.append("STRENGTHS:")
            for s in analysis.strengths:
                lines.append(f"  + {s}")
            lines.append("")

        if analysis.weaknesses:
            lines.append("CONCERNS:")
            for w in analysis.weaknesses:
                lines.append(f"  - {w}")

        return "\n".join(lines)

    # Interpretation helpers
    def _interpret_margin(self, margin_type: str, value: float, benchmark: float) -> str:
        if value < 0:
            return f"Negative {margin_type} margin - operational issues"
        elif value < benchmark * 0.5:
            return f"Low {margin_type} margin - below industry average"
        elif value > benchmark * 1.5:
            return f"Strong {margin_type} margin - above industry average"
        return f"Healthy {margin_type} margin - around industry average"

    def _interpret_roe(self, value: float) -> str:
        if value < 0:
            return "Negative ROE - company is unprofitable"
        elif value < 0.10:
            return "Low ROE - below cost of equity"
        elif value > 0.20:
            return "Strong ROE - excellent returns for shareholders"
        return "Moderate ROE - acceptable returns"

    def _interpret_roa(self, value: float) -> str:
        if value < 0:
            return "Negative ROA - assets not generating returns"
        elif value < 0.05:
            return "Low ROA - inefficient asset utilization"
        elif value > 0.10:
            return "Strong ROA - efficient asset utilization"
        return "Moderate ROA - reasonable asset efficiency"

    def _interpret_current_ratio(self, value: float) -> str:
        if value < 1.0:
            return "Liquidity risk - current assets don't cover liabilities"
        elif value < 1.5:
            return "Adequate liquidity but limited buffer"
        elif value > 3.0:
            return "High liquidity - may indicate excess cash"
        return "Healthy liquidity position"

    def _interpret_quick_ratio(self, value: float) -> str:
        if value < 0.5:
            return "Significant liquidity risk"
        elif value < 1.0:
            return "Limited quick liquidity"
        return "Good quick liquidity position"

    def _interpret_debt_equity(self, value: float) -> str:
        if value > 2.0:
            return "High leverage - significant financial risk"
        elif value > 1.0:
            return "Moderate leverage - manageable risk"
        return "Conservative leverage - low financial risk"

    def _interpret_interest_coverage(self, value: float) -> str:
        if value < 1.5:
            return "Concerning - limited ability to service debt"
        elif value < 3.0:
            return "Adequate interest coverage"
        return "Strong interest coverage - low default risk"

    def _interpret_asset_turnover(self, value: float) -> str:
        if value < 0.5:
            return "Low asset efficiency"
        elif value > 1.0:
            return "Strong asset utilization"
        return "Moderate asset efficiency"

    def _interpret_pe(self, value: float) -> str:
        if value < 10:
            return "Potentially undervalued (or declining earnings)"
        elif value > 25:
            return "Premium valuation (high growth expected)"
        return "Reasonable valuation"

    def _interpret_pb(self, value: float) -> str:
        if value < 1.0:
            return "Trading below book value - potential value"
        elif value > 3.0:
            return "Premium to book value"
        return "Trading around book value"

    def _interpret_ev_ebitda(self, value: float) -> str:
        if value < 8:
            return "Potentially undervalued"
        elif value > 15:
            return "Rich valuation"
        return "Reasonable valuation"

    def _interpret_growth(self, metric: str, value: float) -> str:
        if value < 0:
            return f"Declining {metric} ({value:.0%})"
        elif value < 0.05:
            return f"Slow {metric} growth ({value:.0%})"
        elif value > 0.20:
            return f"Strong {metric} growth ({value:.0%})"
        return f"Moderate {metric} growth ({value:.0%})"


def calculate_ratios(
    income_statement: Dict[str, Any],
    balance_sheet: Dict[str, Any],
    **kwargs,
) -> RatioAnalysis:
    """Convenience function to calculate ratios.

    Args:
        income_statement: Income statement data.
        balance_sheet: Balance sheet data.
        **kwargs: Additional arguments.

    Returns:
        RatioAnalysis.
    """
    calculator = RatioCalculator()
    return calculator.calculate(income_statement, balance_sheet, **kwargs)
