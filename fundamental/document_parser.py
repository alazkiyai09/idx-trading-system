"""
Document Parser Module

Parses PDF financial statements (annual reports, quarterly reports)
for Indonesian companies using pdfplumber for table extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Type of financial document."""
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    FINANCIAL_STATEMENT = "financial_statement"
    AUDITOR_REPORT = "auditor_report"
    UNKNOWN = "unknown"


class StatementType(Enum):
    """Type of financial statement."""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    EQUITY_CHANGE = "equity_change"
    NOTES = "notes"
    UNKNOWN = "unknown"


@dataclass
class ParsedTable:
    """A parsed table from a PDF.

    Attributes:
        headers: Column headers.
        rows: Table rows as lists of values.
        statement_type: Detected statement type.
        page_number: Page where table was found.
        confidence: Confidence score of parsing.
    """

    headers: List[str]
    rows: List[List[str]]
    statement_type: StatementType = StatementType.UNKNOWN
    page_number: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "headers": self.headers,
            "rows": self.rows,
            "statement_type": self.statement_type.value,
            "page_number": self.page_number,
            "confidence": self.confidence,
        }


@dataclass
class ParsedDocument:
    """Result of parsing a financial document.

    Attributes:
        file_path: Path to source file.
        document_type: Type of document.
        company_name: Company name.
        period: Reporting period.
        tables: Extracted tables.
        text_sections: Extracted text sections.
        metadata: Document metadata.
    """

    file_path: str
    document_type: DocumentType = DocumentType.UNKNOWN
    company_name: str = ""
    period: str = ""
    tables: List[ParsedTable] = field(default_factory=list)
    text_sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "document_type": self.document_type.value,
            "company_name": self.company_name,
            "period": self.period,
            "tables": [t.to_dict() for t in self.tables],
            "text_sections": self.text_sections,
            "metadata": self.metadata,
        }


class DocumentParser:
    """Parses financial documents from PDFs.

    Extracts tables and text from Indonesian financial statements,
    handling both Indonesian and English terminology.

    Example:
        parser = DocumentParser()
        result = parser.parse("annual_report_2023.pdf")
        for table in result.tables:
            print(f"Found {table.statement_type.value} on page {table.page_number}")
    """

    # Keywords for statement detection (Indonesian and English)
    INCOME_KEYWORDS = [
        "laba rugi", "laba-rugi", "pendapatan", "pendapatan usaha",
        "income statement", "profit loss", "revenue", "sales",
        "bebannya", "beban operasional", "operating expenses",
        "laba bersih", "net income", "net profit",
    ]

    BALANCE_KEYWORDS = [
        "neraca", "posisi keuangan", "aset", "kewajiban",
        "balance sheet", "financial position", "assets", "liabilities",
        "ekuitas", "equity", "modal", "capital",
    ]

    CASHFLOW_KEYWORDS = [
        "arus kas", "cash flow", "cashflow",
        "aktivitas operasi", "operating activities",
        "aktivitas investasi", "investing activities",
        "aktivitas pendanaan", "financing activities",
    ]

    def __init__(self) -> None:
        """Initialize document parser."""
        pass

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse a financial document.

        Args:
            file_path: Path to PDF file.

        Returns:
            ParsedDocument with extracted data.
        """
        result = ParsedDocument(file_path=file_path)

        # Check file exists
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return result

        # Try to import pdfplumber
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed, returning empty result")
            result.metadata["error"] = "pdfplumber not installed"
            return result

        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                result.metadata["num_pages"] = len(pdf.pages)

                # Detect document type from first few pages
                result.document_type = self._detect_document_type(pdf)

                # Extract company name and period
                result.company_name = self._extract_company_name(pdf)
                result.period = self._extract_period(pdf)

                # Extract tables from each page
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = self._extract_tables(page, page_num)
                    result.tables.extend(tables)

                    # Extract key text sections
                    text = page.extract_text() or ""
                    self._categorize_text(text, result.text_sections, page_num)

                logger.info(
                    f"Parsed {file_path}: {len(result.tables)} tables, "
                    f"{len(result.text_sections)} text sections"
                )

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            result.metadata["error"] = str(e)

        return result

    def _detect_document_type(self, pdf) -> DocumentType:
        """Detect document type from content.

        Args:
            pdf: pdfplumber PDF object.

        Returns:
            DocumentType.
        """
        # Check first 5 pages for document type indicators
        text = ""
        for page in pdf.pages[:5]:
            text += (page.extract_text() or "").lower()

        if "laporan tahunan" in text or "annual report" in text:
            return DocumentType.ANNUAL_REPORT
        elif "laporan keuangan" in text or "financial statement" in text:
            return DocumentType.FINANCIAL_STATEMENT
        elif "laporan kuartalan" in text or "quarterly" in text:
            return DocumentType.QUARTERLY_REPORT
        elif "auditor" in text or "akuntan publik" in text:
            return DocumentType.AUDITOR_REPORT

        return DocumentType.UNKNOWN

    def _extract_company_name(self, pdf) -> str:
        """Extract company name from document.

        Args:
            pdf: pdfplumber PDF object.

        Returns:
            Company name.
        """
        # Check first page for company name
        if pdf.pages:
            text = pdf.pages[0].extract_text() or ""

            # Common patterns for company names
            patterns = [
                r"PT\.?\s+([A-Z][A-Za-z\s]+(?:Tbk\.?|TBK)?)",
                r"([A-Z][A-Za-z\s]+)\s+(?:Tbk|TBK)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()

        return ""

    def _extract_period(self, pdf) -> str:
        """Extract reporting period from document.

        Args:
            pdf: pdfplumber PDF object.

        Returns:
            Period string (e.g., "2023", "Q3 2023").
        """
        # Check first few pages for period
        text = ""
        for page in pdf.pages[:3]:
            text += page.extract_text() or ""

        # Look for year
        year_match = re.search(r"\b(20\d{2})\b", text)
        if year_match:
            year = year_match.group(1)

            # Check for quarter
            quarter_match = re.search(r"(?:Q|Kuartal|Quarter)\s*([1-4])", text, re.I)
            if quarter_match:
                return f"Q{quarter_match.group(1)} {year}"

            return year

        return ""

    def _extract_tables(self, page, page_num: int) -> List[ParsedTable]:
        """Extract tables from a page.

        Args:
            page: pdfplumber page object.
            page_num: Page number.

        Returns:
            List of ParsedTable objects.
        """
        tables = []

        try:
            raw_tables = page.extract_tables()

            for raw_table in raw_tables:
                if not raw_table or len(raw_table) < 2:
                    continue

                # Clean up table
                headers = [self._clean_cell(cell) for cell in raw_table[0]]
                rows = [
                    [self._clean_cell(cell) for cell in row]
                    for row in raw_table[1:]
                    if any(cell for cell in row)  # Skip empty rows
                ]

                if not headers or not rows:
                    continue

                # Detect statement type
                table_text = " ".join(str(cell) for row in raw_table for cell in row if cell)
                statement_type = self._detect_statement_type(table_text)

                table = ParsedTable(
                    headers=headers,
                    rows=rows,
                    statement_type=statement_type,
                    page_number=page_num,
                    confidence=self._calculate_confidence(headers, rows),
                )

                tables.append(table)

        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")

        return tables

    def _clean_cell(self, cell: Optional[str]) -> str:
        """Clean a table cell value.

        Args:
            cell: Raw cell value.

        Returns:
            Cleaned cell value.
        """
        if cell is None:
            return ""

        # Remove extra whitespace
        cell = " ".join(str(cell).split())

        # Remove common PDF artifacts
        cell = cell.replace("\n", " ")
        cell = re.sub(r"\s+", " ", cell)

        return cell.strip()

    def _detect_statement_type(self, text: str) -> StatementType:
        """Detect financial statement type from text.

        Args:
            text: Table text content.

        Returns:
            StatementType.
        """
        text_lower = text.lower()

        # Count keyword matches for each type
        income_score = sum(1 for kw in self.INCOME_KEYWORDS if kw in text_lower)
        balance_score = sum(1 for kw in self.BALANCE_KEYWORDS if kw in text_lower)
        cashflow_score = sum(1 for kw in self.CASHFLOW_KEYWORDS if kw in text_lower)

        max_score = max(income_score, balance_score, cashflow_score)

        if max_score == 0:
            return StatementType.UNKNOWN

        if income_score == max_score:
            return StatementType.INCOME_STATEMENT
        elif balance_score == max_score:
            return StatementType.BALANCE_SHEET
        elif cashflow_score == max_score:
            return StatementType.CASH_FLOW

        return StatementType.UNKNOWN

    def _calculate_confidence(self, headers: List[str], rows: List[List[str]]) -> float:
        """Calculate parsing confidence score.

        Args:
            headers: Table headers.
            rows: Table rows.

        Returns:
            Confidence score (0-1).
        """
        if not headers or not rows:
            return 0.0

        score = 0.5  # Base score

        # Check for numeric content
        numeric_count = 0
        total_cells = 0

        for row in rows:
            for cell in row:
                total_cells += 1
                if self._is_numeric(cell):
                    numeric_count += 1

        if total_cells > 0:
            numeric_ratio = numeric_count / total_cells
            # Financial tables should have mostly numbers
            if numeric_ratio > 0.5:
                score += 0.2
            if numeric_ratio > 0.7:
                score += 0.1

        # Check for common financial headers
        header_text = " ".join(headers).lower()
        financial_terms = ["total", "jumlah", "period", "tahun", "year", "aset", "asset", "kewajiban"]
        if any(term in header_text for term in financial_terms):
            score += 0.2

        return min(score, 1.0)

    def _is_numeric(self, value: str) -> bool:
        """Check if value is numeric.

        Args:
            value: String value.

        Returns:
            True if numeric.
        """
        if not value:
            return False

        # Remove common formatting
        cleaned = value.replace(",", "").replace(".", "").replace("%", "")
        cleaned = cleaned.replace("(", "").replace(")", "").strip()

        if cleaned.startswith("-"):
            cleaned = cleaned[1:]

        return cleaned.isdigit()

    def _categorize_text(
        self,
        text: str,
        sections: Dict[str, str],
        page_num: int,
    ) -> None:
        """Categorize text into sections.

        Args:
            text: Page text.
            sections: Dictionary to store sections.
            page_num: Page number.
        """
        text_lower = text.lower()

        # Auditor report
        if "auditor" in text_lower or "akuntan publik" in text_lower:
            key = f"auditor_report_p{page_num}"
            sections[key] = text

        # Going concern
        if "going concern" in text_lower or "kelangsungan usaha" in text_lower:
            key = f"going_concern_p{page_num}"
            sections[key] = text

        # Risk factors
        if "faktor risiko" in text_lower or "risk factors" in text_lower:
            key = f"risk_factors_p{page_num}"
            sections[key] = text

        # Management discussion
        if "manajemen" in text_lower and ("diskusi" in text_lower or "analysis" in text_lower):
            key = f"management_discussion_p{page_num}"
            sections[key] = text

    def parse_text_content(self, text: str) -> Dict[str, Any]:
        """Parse financial data from raw text (no PDF).

        Args:
            text: Raw text content.

        Returns:
            Dictionary of extracted data.
        """
        result = {
            "numbers": [],
            "percentages": [],
            "dates": [],
            "currency_amounts": [],
        }

        # Extract numbers
        number_pattern = r"\b[\d,]+(?:\.\d+)?\b"
        result["numbers"] = re.findall(number_pattern, text)

        # Extract percentages
        percent_pattern = r"\b[\d,]+(?:\.\d+)?%"
        result["percentages"] = re.findall(percent_pattern, text)

        # Extract dates
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}\b",
        ]
        for pattern in date_patterns:
            result["dates"].extend(re.findall(pattern, text, re.I))

        # Extract currency amounts (IDR)
        currency_pattern = r"Rp\.?\s*[\d,]+(?:\.\d+)?(?:\s*(?:juta|miliiar|triliun))?"
        result["currency_amounts"] = re.findall(currency_pattern, text, re.I)

        return result


def parse_document(file_path: str) -> ParsedDocument:
    """Convenience function to parse a document.

    Args:
        file_path: Path to PDF file.

    Returns:
        ParsedDocument.
    """
    parser = DocumentParser()
    return parser.parse(file_path)
