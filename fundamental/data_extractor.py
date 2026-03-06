"""
Data Extractor Module

Extracts structured financial data from parsed documents:
- Maps Indonesian/English terminology
- Handles multiple table formats
- Normalizes values and units
- Validates extracted data
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum

from .document_parser import ParsedDocument, ParsedTable, StatementType

logger = logging.getLogger(__name__)


class UnitScale(Enum):
    """Unit scale multipliers."""
    UNITS = 1
    THOUSANDS = 1_000
    MILLIONS = 1_000_000
    BILLIONS = 1_000_000_000
    TRILLIONS = 1_000_000_000_000


@dataclass
class ExtractedValue:
    """Extracted financial value with metadata."""
    value: float
    original_text: str
    unit_scale: UnitScale
    confidence: float
    source_table: str = ""
    row_index: int = -1


@dataclass
class FinancialData:
    """Structured financial data container."""
    # Income Statement
    revenue: Optional[float] = None
    cost_of_goods_sold: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    interest_expense: Optional[float] = None
    depreciation: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None

    # Balance Sheet
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    cash: Optional[float] = None
    inventory: Optional[float] = None
    accounts_receivable: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_equity: Optional[float] = None
    retained_earnings: Optional[float] = None

    # Cash Flow
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    dividends_paid: Optional[float] = None
    free_cash_flow: Optional[float] = None

    # Market Data
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    stock_price: Optional[float] = None

    # Metadata
    fiscal_year: Optional[int] = None
    period: str = ""
    currency: str = "IDR"
    unit_scale: UnitScale = UnitScale.MILLIONS

    # Prior period data
    prior_year: Dict[str, Any] = field(default_factory=dict)

    # Raw extracted values for verification
    raw_values: Dict[str, ExtractedValue] = field(default_factory=dict)


class DataExtractor:
    """Extracts structured financial data from parsed documents.

    Handles:
    - Indonesian and English terminology
    - Multiple number formats
    - Unit scale detection
    - Cross-statement validation

    Example:
        extractor = DataExtractor()
        financial_data = extractor.extract(parsed_document)
        print(f"Revenue: {financial_data.revenue:,.0f}")
    """

    # Income statement field mappings (Indonesian -> Standard)
    INCOME_STATEMENT_MAPPINGS = {
        # Revenue
        "revenue": ["pendapatan", "pendapatan usaha", "pendapatan operasional",
                   "revenue", "sales", "net sales", "penjualan bersih",
                   "total pendapatan", "total revenue"],
        # COGS
        "cost_of_goods_sold": ["hpp", "harga pokok penjualan", "biaya pokok penjualan",
                               "cost of goods sold", "cogs", "cost of sales"],
        # Gross Profit
        "gross_profit": ["laba kotor", "laba bruto", "gross profit",
                        "keuntungan kotor"],
        # Operating Income
        "operating_income": ["laba operasional", "laba usaha", "operating income",
                            "operating profit", "ebit", "laba sebelum bunga pajak"],
        # Net Income
        "net_income": ["laba bersih", "laba tahun berjalan", "net income",
                       "profit for the year", "keuntungan bersih", "net profit"],
        # Interest Expense
        "interest_expense": ["beban bunga", "biaya bunga", "interest expense",
                            "bunga dibayar"],
        # Depreciation
        "depreciation": ["penyusutan", "depresiasi", "depreciation",
                        "depreciation expense"],
    }

    # Balance sheet field mappings
    BALANCE_SHEET_MAPPINGS = {
        # Assets
        "total_assets": ["total aset", "jumlah aset", "total assets",
                        "jumlah harta", "total harta"],
        "current_assets": ["aset lancar", "aset current", "current assets",
                          "harta lancar", "aktiva lancar"],
        "cash": ["kas", "cash", "uang tunai", "kas dan setara kas",
                "cash and cash equivalents"],
        "inventory": ["persediaan", "inventaris", "inventory",
                     "stok", "barang dagangan"],
        "accounts_receivable": ["piutang", "piutang usaha", "accounts receivable",
                               "trade receivables", "piutang dagang"],
        # Liabilities
        "total_liabilities": ["total liabilitas", "jumlah liabilitas", "total liabilities",
                             "jumlah kewajiban", "total kewajiban", "total utang"],
        "current_liabilities": ["liabilitas jangka pendek", "kewajiban lancar",
                               "current liabilities", "utang lancar"],
        "long_term_debt": ["utang jangka panjang", "liabilitas jangka panjang",
                          "long term debt", "non-current liabilities"],
        # Equity
        "total_equity": ["total ekuitas", "jumlah ekuitas", "total equity",
                        "modal", "total modal", "net assets"],
        "retained_earnings": ["laba ditahan", "akumulasi laba", "retained earnings",
                             "saldo laba"],
    }

    # Cash flow field mappings
    CASH_FLOW_MAPPINGS = {
        "operating_cash_flow": ["arus kas operasional", "arus kas dari aktivitas operasi",
                               "operating cash flow", "cash from operations",
                               "net cash from operating activities"],
        "investing_cash_flow": ["arus kas investasi", "arus kas dari aktivitas investasi",
                               "investing cash flow", "cash from investing"],
        "financing_cash_flow": ["arus kas pendanaan", "arus kas dari aktivitas pendanaan",
                               "financing cash flow", "cash from financing"],
        "capital_expenditure": ["pengeluaran modal", "belanja modal", "capex",
                               "capital expenditure", "acquisition of assets"],
        "dividends_paid": ["dividen dibayar", "pembayaran dividen", "dividends paid"],
        "free_cash_flow": ["arus kas bebas", "free cash flow"],
    }

    # Unit scale indicators
    UNIT_PATTERNS = {
        UnitScale.TRILLIONS: [r"triliun", r"trillion", r"trn", r"t\.", r"ribu miliar"],
        UnitScale.BILLIONS: [r"miliar", r"billion", r"bln", r"m\.", r"milyar"],
        UnitScale.MILLIONS: [r"juta", r"million", r"mio", r"jt"],
        UnitScale.THOUSANDS: [r"ribu", r"thousand", r"ths", r"rb"],
    }

    def __init__(self) -> None:
        """Initialize data extractor."""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}

        for field_name, patterns in self.INCOME_STATEMENT_MAPPINGS.items():
            self._compiled_patterns[field_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        for field_name, patterns in self.BALANCE_SHEET_MAPPINGS.items():
            self._compiled_patterns[field_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        for field_name, patterns in self.CASH_FLOW_MAPPINGS.items():
            self._compiled_patterns[field_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Unit patterns
        self._compiled_unit_patterns: Dict[UnitScale, List[re.Pattern]] = {}
        for unit, patterns in self.UNIT_PATTERNS.items():
            self._compiled_unit_patterns[unit] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def extract(self, document: ParsedDocument) -> FinancialData:
        """Extract financial data from parsed document.

        Args:
            document: Parsed document from DocumentParser.

        Returns:
            FinancialData with extracted values.
        """
        financial_data = FinancialData()

        # Detect document unit scale
        financial_data.unit_scale = self._detect_unit_scale(document)

        # Extract from each statement type
        for table in document.tables:
            if table.statement_type == StatementType.INCOME_STATEMENT:
                self._extract_income_statement(table, financial_data)
            elif table.statement_type == StatementType.BALANCE_SHEET:
                self._extract_balance_sheet(table, financial_data)
            elif table.statement_type == StatementType.CASH_FLOW:
                self._extract_cash_flow(table, financial_data)

        # Calculate derived values
        self._calculate_derived_values(financial_data)

        # Extract metadata
        self._extract_metadata(document, financial_data)

        return financial_data

    def _detect_unit_scale(self, document: ParsedDocument) -> UnitScale:
        """Detect the unit scale used in the document."""
        full_text = document.raw_text.lower()

        for unit, patterns in self._compiled_unit_patterns.items():
            for pattern in patterns:
                if pattern.search(full_text):
                    return unit

        # Default to millions for Indonesian financial statements
        return UnitScale.MILLIONS

    def _extract_income_statement(
        self,
        table: ParsedTable,
        data: FinancialData
    ) -> None:
        """Extract income statement values."""
        for row_idx, row in enumerate(table.rows):
            if not row:
                continue

            # First column is typically the label
            label = str(row[0]) if row else ""
            values = [self._parse_number(cell) for cell in row[1:] if cell]
            current_value = values[0] if values else None

            if current_value is None:
                continue

            # Match against known field patterns
            for field_name, patterns in self._compiled_patterns.items():
                if field_name not in self.INCOME_STATEMENT_MAPPINGS:
                    continue

                for pattern in patterns:
                    if pattern.search(label):
                        setattr(data, field_name, current_value)
                        data.raw_values[field_name] = ExtractedValue(
                            value=current_value,
                            original_text=label,
                            unit_scale=data.unit_scale,
                            confidence=0.9,
                            source_table=table.name,
                            row_index=row_idx,
                        )
                        break

    def _extract_balance_sheet(
        self,
        table: ParsedTable,
        data: FinancialData
    ) -> None:
        """Extract balance sheet values."""
        for row_idx, row in enumerate(table.rows):
            if not row:
                continue

            label = str(row[0]) if row else ""
            values = [self._parse_number(cell) for cell in row[1:] if cell]
            current_value = values[0] if values else None

            if current_value is None:
                continue

            for field_name, patterns in self._compiled_patterns.items():
                if field_name not in self.BALANCE_SHEET_MAPPINGS:
                    continue

                for pattern in patterns:
                    if pattern.search(label):
                        setattr(data, field_name, current_value)
                        data.raw_values[field_name] = ExtractedValue(
                            value=current_value,
                            original_text=label,
                            unit_scale=data.unit_scale,
                            confidence=0.9,
                            source_table=table.name,
                            row_index=row_idx,
                        )
                        break

    def _extract_cash_flow(
        self,
        table: ParsedTable,
        data: FinancialData
    ) -> None:
        """Extract cash flow values."""
        for row_idx, row in enumerate(table.rows):
            if not row:
                continue

            label = str(row[0]) if row else ""
            values = [self._parse_number(cell) for cell in row[1:] if cell]
            current_value = values[0] if values else None

            if current_value is None:
                continue

            for field_name, patterns in self._compiled_patterns.items():
                if field_name not in self.CASH_FLOW_MAPPINGS:
                    continue

                for pattern in patterns:
                    if pattern.search(label):
                        setattr(data, field_name, current_value)
                        data.raw_values[field_name] = ExtractedValue(
                            value=current_value,
                            original_text=label,
                            unit_scale=data.unit_scale,
                            confidence=0.9,
                            source_table=table.name,
                            row_index=row_idx,
                        )
                        break

    def _parse_number(self, value: Any) -> Optional[float]:
        """Parse a number from various formats.

        Handles:
        - Indonesian format (1.234.567,89)
        - US format (1,234,567.89)
        - Parentheses for negative ((1,234))
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if not text or text == "-":
            return None

        # Check for negative in parentheses
        is_negative = text.startswith("(") and text.endswith(")")
        if is_negative:
            text = text[1:-1]

        # Remove currency symbols and spaces
        text = re.sub(r"[Rp$€£\s]", "", text)

        # Handle different decimal separators
        # Indonesian: 1.234.567,89 -> 1234567.89
        # US: 1,234,567.89 -> 1234567.89

        # Count dots and commas
        dots = text.count(".")
        commas = text.count(",")

        if dots > 1 or (dots == 1 and commas == 1 and text.rfind(".") < text.rfind(",")):
            # Indonesian format: dots as thousands, comma as decimal
            text = text.replace(".", "").replace(",", ".")
        elif commas > 1 or (dots == 1 and commas == 1 and text.rfind(".") > text.rfind(",")):
            # US format: commas as thousands, dot as decimal
            text = text.replace(",", "")
        elif commas == 1 and dots == 0:
            # Single comma, could be decimal separator
            text = text.replace(",", ".")

        try:
            result = float(text)
            return -result if is_negative else result
        except ValueError:
            return None

    def _calculate_derived_values(self, data: FinancialData) -> None:
        """Calculate derived financial values."""
        # Gross profit if not provided
        if data.gross_profit is None and data.revenue and data.cost_of_goods_sold:
            data.gross_profit = data.revenue - data.cost_of_goods_sold

        # EBITDA
        if data.ebitda is None:
            if data.ebit and data.depreciation:
                data.ebitda = data.ebit + data.depreciation
            elif data.operating_income and data.depreciation:
                data.ebitda = data.operating_income + data.depreciation

        # Free cash flow
        if data.free_cash_flow is None and data.operating_cash_flow:
            capex = data.capital_expenditure or 0
            data.free_cash_flow = data.operating_cash_flow - abs(capex)

    def _extract_metadata(
        self,
        document: ParsedDocument,
        data: FinancialData
    ) -> None:
        """Extract metadata from document."""
        # Extract fiscal year
        year_match = re.search(r"\b(20\d{2})\b", document.raw_text)
        if year_match:
            data.fiscal_year = int(year_match.group(1))

        # Extract period
        period_patterns = [
            (r"tahun.*?(?:buku|berjalan)", "Annual"),
            (r"semester", "Semi-Annual"),
            (r"triwulan", "Quarterly"),
            (r"quarter", "Quarterly"),
            (r"q[1-4]", "Quarterly"),
        ]

        for pattern, period_type in period_patterns:
            if re.search(pattern, document.raw_text, re.IGNORECASE):
                data.period = period_type
                break

    def extract_to_dict(self, document: ParsedDocument) -> Dict[str, Any]:
        """Extract financial data as dictionary.

        Args:
            document: Parsed document.

        Returns:
            Dictionary with extracted financial data.
        """
        data = self.extract(document)

        result = {
            "income_statement": {
                "revenue": data.revenue,
                "cost_of_goods_sold": data.cost_of_goods_sold,
                "gross_profit": data.gross_profit,
                "operating_income": data.operating_income,
                "net_income": data.net_income,
                "interest_expense": data.interest_expense,
                "depreciation": data.depreciation,
                "ebit": data.ebit,
                "ebitda": data.ebitda,
            },
            "balance_sheet": {
                "total_assets": data.total_assets,
                "current_assets": data.current_assets,
                "cash": data.cash,
                "inventory": data.inventory,
                "accounts_receivable": data.accounts_receivable,
                "total_liabilities": data.total_liabilities,
                "current_liabilities": data.current_liabilities,
                "long_term_debt": data.long_term_debt,
                "total_equity": data.total_equity,
                "retained_earnings": data.retained_earnings,
            },
            "cash_flow": {
                "operating_cash_flow": data.operating_cash_flow,
                "investing_cash_flow": data.investing_cash_flow,
                "financing_cash_flow": data.financing_cash_flow,
                "capital_expenditure": data.capital_expenditure,
                "dividends_paid": data.dividends_paid,
                "free_cash_flow": data.free_cash_flow,
            },
            "metadata": {
                "fiscal_year": data.fiscal_year,
                "period": data.period,
                "currency": data.currency,
                "unit_scale": data.unit_scale.value,
            },
        }

        if data.prior_year:
            result["prior_year"] = data.prior_year

        return result
