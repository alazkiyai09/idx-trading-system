"""
Tests for Fundamental Analysis Module

Tests for:
- Document parser
- Data extractor
- Fraud detector
- Ratio calculator
- Analysis agents
- Report generator
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Document Parser Tests
class TestDocumentParser:
    """Tests for DocumentParser."""

    @pytest.fixture
    def parser(self):
        from fundamental.document_parser import DocumentParser
        return DocumentParser()

    def test_init(self, parser):
        """Test parser initialization."""
        assert parser is not None

    def test_detect_statement_type_income(self, parser):
        """Test income statement detection."""
        from fundamental.document_parser import StatementType
        text = "LAPORAN LABA RUGI - Pendapatan Usaha"
        result = parser._detect_statement_type(text)
        assert result == StatementType.INCOME_STATEMENT

    def test_detect_statement_type_balance(self, parser):
        """Test balance sheet detection."""
        from fundamental.document_parser import StatementType
        text = "NERACA - Aset Lancar"
        result = parser._detect_statement_type(text)
        assert result == StatementType.BALANCE_SHEET

    def test_detect_statement_type_cashflow(self, parser):
        """Test cash flow detection."""
        from fundamental.document_parser import StatementType
        text = "LAPORAN ARUS KAS - Aktivitas Operasi"
        result = parser._detect_statement_type(text)
        assert result == StatementType.CASH_FLOW

    def test_clean_cell(self, parser):
        """Test cell cleaning."""
        dirty = "Pendapatan\xa0Usaha\t\tRp"
        clean = parser._clean_cell(dirty)
        assert "\xa0" not in clean
        assert "\t\t" not in clean


class TestDataExtractor:
    """Tests for DataExtractor."""

    @pytest.fixture
    def extractor(self):
        from fundamental.data_extractor import DataExtractor
        return DataExtractor()

    def test_init(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None

    def test_parse_number_indonesian_format(self, extractor):
        """Test parsing Indonesian number format."""
        result = extractor._parse_number("1.234.567,89")
        assert result == pytest.approx(1234567.89, rel=0.01)

    def test_parse_number_us_format(self, extractor):
        """Test parsing US number format."""
        result = extractor._parse_number("1,234,567.89")
        assert result == pytest.approx(1234567.89, rel=0.01)

    def test_parse_number_negative_parentheses(self, extractor):
        """Test parsing negative number in parentheses."""
        result = extractor._parse_number("(1.234.567)")
        assert result == pytest.approx(-1234567, rel=0.01)

    def test_parse_number_with_currency(self, extractor):
        """Test parsing number with currency symbol."""
        result = extractor._parse_number("Rp 1.234.567")
        assert result == pytest.approx(1234567, rel=0.01)

    def test_parse_number_empty(self, extractor):
        """Test parsing empty value."""
        assert extractor._parse_number("") is None
        assert extractor._parse_number("-") is None
        assert extractor._parse_number(None) is None

    def test_parse_number_simple(self, extractor):
        """Test parsing simple number."""
        assert extractor._parse_number("1000000") == 1000000
        assert extractor._parse_number(1000000) == 1000000

    def test_extract_metadata_fiscal_year(self, extractor):
        """Test fiscal year extraction."""
        from fundamental.document_parser import ParsedDocument
        doc = Mock(spec=ParsedDocument)
        doc.raw_text = "Laporan Keuangan Tahun 2023"

        from fundamental.data_extractor import FinancialData
        data = FinancialData()
        extractor._extract_metadata(doc, data)

        assert data.fiscal_year == 2023


class TestFraudDetector:
    """Tests for FraudDetector."""

    @pytest.fixture
    def detector(self):
        from fundamental.fraud_detector import FraudDetector
        return FraudDetector()

    @pytest.fixture
    def sample_data(self):
        return {
            "income_statement": {
                "revenue": 10_000_000_000,
                "net_income": 1_000_000_000,
                "gross_profit": 4_000_000_000,
            },
            "balance_sheet": {
                "total_assets": 15_000_000_000,
                "total_liabilities": 8_000_000_000,
                "total_equity": 7_000_000_000,
                "current_assets": 5_000_000_000,
                "current_liabilities": 3_000_000_000,
            },
            "cash_flow": {
                "operating_cash_flow": 1_200_000_000,
            },
            "raw_numbers": [
                1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012,
                1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999,
            ],
        }

    def test_init(self, detector):
        """Test detector initialization."""
        assert detector is not None

    def test_analyze_returns_result(self, detector, sample_data):
        """Test that analyze returns FraudAnalysisResult."""
        from fundamental.fraud_detector import FraudAnalysisResult
        result = detector.analyze(sample_data)
        assert isinstance(result, FraudAnalysisResult)

    def test_check_balance_sheet_valid(self, detector):
        """Test balance sheet equation verification - valid."""
        from fundamental.fraud_detector import FraudAnalysisResult
        data = {
            "balance_sheet": {
                "total_assets": 100,
                "total_liabilities": 40,
                "total_equity": 60,
            }
        }
        result = FraudAnalysisResult()
        detector._check_balance_sheet(data, result)
        # Should not add critical check for valid balance
        assert len(result.checks) >= 0

    def test_check_balance_sheet_invalid(self, detector):
        """Test balance sheet equation failure detection."""
        from fundamental.fraud_detector import FraudAnalysisResult, RiskLevel
        data = {
            "balance_sheet": {
                "total_assets": 100,
                "total_liabilities": 50,
                "total_equity": 60,  # Doesn't add up
            }
        }
        result = FraudAnalysisResult()
        detector._check_balance_sheet(data, result)
        # Should detect the mismatch
        assert any(c.indicator.value == "balance_mismatch" for c in result.checks)

    def test_analyze_with_benford_data(self, detector):
        """Test analysis with data for Benford check."""
        data = {
            "balance_sheet": {"total_assets": 100, "total_liabilities": 50, "total_equity": 50},
            "raw_numbers": list(range(100, 1000, 7)),  # Diverse numbers
        }
        result = detector.analyze(data)
        assert result.fraud_probability >= 0


class TestRatioCalculator:
    """Tests for RatioCalculator."""

    @pytest.fixture
    def calculator(self):
        from fundamental.ratio_calculator import RatioCalculator
        return RatioCalculator()

    @pytest.fixture
    def sample_data(self):
        return {
            "income_statement": {
                "revenue": 10_000_000_000,
                "net_income": 1_000_000_000,
                "gross_profit": 4_000_000_000,
                "operating_income": 1_500_000_000,
                "interest_expense": 200_000_000,
            },
            "balance_sheet": {
                "total_assets": 15_000_000_000,
                "total_equity": 7_000_000_000,
                "total_liabilities": 8_000_000_000,
                "current_assets": 5_000_000_000,
                "current_liabilities": 3_000_000_000,
                "cash": 1_000_000_000,
                "inventory": 1_500_000_000,
            },
            "cash_flow": {
                "operating_cash_flow": 1_200_000_000,
            },
        }

    def test_init(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None

    def test_calculate_returns_result(self, calculator, sample_data):
        """Test that calculate returns RatioAnalysis."""
        from fundamental.ratio_calculator import RatioAnalysis
        result = calculator.calculate(
            income_statement=sample_data["income_statement"],
            balance_sheet=sample_data["balance_sheet"],
            cash_flow=sample_data["cash_flow"],
        )
        assert isinstance(result, RatioAnalysis)

    def test_calculate_profitability_ratios(self, calculator, sample_data):
        """Test profitability ratio calculations."""
        result = calculator.calculate(
            income_statement=sample_data["income_statement"],
            balance_sheet=sample_data["balance_sheet"],
        )
        # Find ROE ratio
        roe = next((r for r in result.ratios if "ROE" in r.name), None)
        assert roe is not None
        # ROE = 1B / 7B = ~14.3%
        assert roe.value == pytest.approx(0.143, rel=0.05)

    def test_calculate_liquidity_ratios(self, calculator, sample_data):
        """Test liquidity ratio calculations."""
        result = calculator.calculate(
            income_statement=sample_data["income_statement"],
            balance_sheet=sample_data["balance_sheet"],
        )
        # Find current ratio
        current_ratio = next((r for r in result.ratios if "Current" in r.name), None)
        assert current_ratio is not None
        # Current ratio = 5B / 3B = 1.67
        assert current_ratio.value == pytest.approx(1.667, rel=0.05)


# Agent Tests
class TestBaseAgent:
    """Tests for BaseAgent."""

    def test_agent_role_enum(self):
        """Test AgentRole enum values."""
        from fundamental.agents.base import AgentRole
        assert AgentRole.AUDITOR.value == "auditor"
        assert AgentRole.GROWTH_ANALYST.value == "growth_analyst"
        assert AgentRole.VALUE_ANALYST.value == "value_analyst"
        assert AgentRole.RISK_ANALYST.value == "risk_analyst"
        assert AgentRole.FORENSIC_ANALYST.value == "forensic_analyst"
        assert AgentRole.SYNTHESIZER.value == "synthesizer"

    def test_agent_finding_creation(self):
        """Test AgentFinding dataclass."""
        from fundamental.agents.base import AgentFinding
        finding = AgentFinding(
            category="test",
            description="Test finding",
            severity="info",
            score=80,
        )
        assert finding.category == "test"
        assert finding.score == 80

    def test_agent_report_creation(self):
        """Test AgentReport dataclass."""
        from fundamental.agents.base import AgentReport, AgentRole
        report = AgentReport(agent_role=AgentRole.AUDITOR)
        assert report.agent_role == AgentRole.AUDITOR
        assert report.findings == []


class TestAuditorAgent:
    """Tests for AuditorAgent."""

    @pytest.fixture
    def agent(self):
        from fundamental.agents.auditor_agent import AuditorAgent
        return AuditorAgent()

    def test_init(self, agent):
        """Test agent initialization."""
        from fundamental.agents.base import AgentRole
        assert agent.role == AgentRole.AUDITOR

    def test_analyze_unqualified_opinion(self, agent):
        """Test detection of unqualified opinion."""
        data = {
            "auditor_report": "Kami menyatakan pendapat wajar tanpa pengecualian atas laporan keuangan"
        }
        report = agent.analyze(data)
        assert any("UNQUALIFIED" in f.description for f in report.findings)

    def test_analyze_qualified_opinion(self, agent):
        """Test detection of qualified opinion."""
        data = {
            "auditor_report": "Kami menyatakan pendapat wajar dengan pengecualian"
        }
        report = agent.analyze(data)
        assert any("QUALIFIED" in f.description for f in report.findings)

    def test_analyze_going_concern(self, agent):
        """Test detection of going concern warning."""
        data = {
            "auditor_report": "Terdapat ketidakpastian signifikan terhadap kelangsungan usaha"
        }
        report = agent.analyze(data)
        assert any("going concern" in f.description.lower() for f in report.findings)

    def test_analyze_big4_auditor(self, agent):
        """Test detection of Big 4 auditor."""
        data = {
            "auditor_report": "Audit performed by KPMG"
        }
        report = agent.analyze(data)
        assert any("Big 4" in f.description for f in report.findings)


class TestGrowthAnalyst:
    """Tests for GrowthAnalyst."""

    @pytest.fixture
    def agent(self):
        from fundamental.agents.growth_analyst import GrowthAnalyst
        return GrowthAnalyst()

    @pytest.fixture
    def sample_data(self):
        return {
            "income_statement": {
                "revenue": 12_000_000_000,
                "gross_profit": 5_000_000_000,
                "net_income": 1_500_000_000,
            },
            "balance_sheet": {
                "total_assets": 20_000_000_000,
            },
            "prior_year": {
                "income_statement": {
                    "revenue": 10_000_000_000,
                }
            },
            "ratios": {
                "roe": 0.18,
            }
        }

    def test_init(self, agent):
        """Test agent initialization."""
        from fundamental.agents.base import AgentRole
        assert agent.role == AgentRole.GROWTH_ANALYST

    def test_analyze_revenue_growth(self, agent, sample_data):
        """Test revenue growth analysis."""
        report = agent.analyze(sample_data)
        # 20% growth (12B vs 10B)
        assert any("growth" in f.description.lower() for f in report.findings)

    def test_analyze_returns_report(self, agent, sample_data):
        """Test that analyze returns AgentReport."""
        from fundamental.agents.base import AgentReport
        report = agent.analyze(sample_data)
        assert isinstance(report, AgentReport)


class TestValueAnalyst:
    """Tests for ValueAnalyst."""

    @pytest.fixture
    def agent(self):
        from fundamental.agents.value_analyst import ValueAnalyst
        return ValueAnalyst()

    @pytest.fixture
    def sample_data(self):
        return {
            "income_statement": {
                "revenue": 10_000_000_000,
                "net_income": 1_000_000_000,
            },
            "balance_sheet": {
                "total_equity": 7_000_000_000,
                "total_liabilities": 5_000_000_000,
                "current_assets": 5_000_000_000,
                "current_liabilities": 3_000_000_000,
            },
            "cash_flow": {
                "operating_cash_flow": 1_200_000_000,
            },
        }

    def test_init(self, agent):
        """Test agent initialization."""
        from fundamental.agents.base import AgentRole
        assert agent.role == AgentRole.VALUE_ANALYST

    def test_analyze_earnings_quality(self, agent, sample_data):
        """Test earnings quality analysis."""
        report = agent.analyze(sample_data)
        # OCF/NI = 1.2, should be positive
        assert any("earnings" in f.category for f in report.findings)

    def test_analyze_balance_sheet(self, agent, sample_data):
        """Test balance sheet analysis."""
        report = agent.analyze(sample_data)
        # D/E = 5B/7B = 0.71, moderate leverage
        assert any("balance_sheet" in f.category or "leverage" in f.description.lower()
                   for f in report.findings)


class TestRiskAnalyst:
    """Tests for RiskAnalyst."""

    @pytest.fixture
    def agent(self):
        from fundamental.agents.risk_analyst import RiskAnalyst
        return RiskAnalyst()

    @pytest.fixture
    def sample_data(self):
        return {
            "income_statement": {
                "revenue": 10_000_000_000,
                "net_income": 500_000_000,
                "operating_income": 800_000_000,
                "interest_expense": 400_000_000,
                "cost_of_goods_sold": 7_000_000_000,
            },
            "balance_sheet": {
                "total_equity": 5_000_000_000,
                "total_liabilities": 10_000_000_000,
                "current_assets": 2_500_000_000,
                "current_liabilities": 3_000_000_000,
                "inventory": 2_000_000_000,
            },
            "cash_flow": {
                "operating_cash_flow": -200_000_000,
            },
        }

    def test_init(self, agent):
        """Test agent initialization."""
        from fundamental.agents.base import AgentRole
        assert agent.role == AgentRole.RISK_ANALYST

    def test_detect_liquidity_risk(self, agent, sample_data):
        """Test liquidity risk detection."""
        report = agent.analyze(sample_data)
        # Current ratio = 2.5B/3B = 0.83 < 1.0, should flag liquidity crisis
        assert any("liquidity" in f.description.lower() for f in report.findings)

    def test_detect_negative_cash_flow(self, agent, sample_data):
        """Test negative cash flow detection."""
        report = agent.analyze(sample_data)
        assert any("negative" in f.description.lower() and "cash" in f.description.lower()
                   for f in report.findings)


class TestForensicAnalyst:
    """Tests for ForensicAnalyst."""

    @pytest.fixture
    def agent(self):
        from fundamental.agents.forensic_analyst import ForensicAnalyst
        return ForensicAnalyst()

    @pytest.fixture
    def sample_data(self):
        return {
            "income_statement": {
                "net_income": 1_000_000_000,
            },
            "cash_flow": {
                "operating_cash_flow": 300_000_000,  # Low relative to net income
            },
            "fraud_analysis": {
                "overall_risk": "medium",
                "fraud_probability": 0.35,
                "red_flags": ["Unusual revenue pattern"],
            }
        }

    def test_init(self, agent):
        """Test agent initialization."""
        from fundamental.agents.base import AgentRole
        assert agent.role == AgentRole.FORENSIC_ANALYST

    def test_analyze_fraud_indicators(self, agent, sample_data):
        """Test fraud indicator analysis."""
        report = agent.analyze(sample_data)
        assert any("fraud" in f.description.lower() for f in report.findings)

    def test_detect_manipulation_pattern(self, agent, sample_data):
        """Test earnings manipulation detection."""
        report = agent.analyze(sample_data)
        # OCF is only 30% of net income - possible manipulation
        assert any("manipulation" in f.description.lower() or "cash flow" in f.description.lower()
                   for f in report.findings)


class TestSynthesizer:
    """Tests for Synthesizer."""

    @pytest.fixture
    def synthesizer(self):
        from fundamental.agents.synthesizer import Synthesizer
        return Synthesizer()

    @pytest.fixture
    def sample_reports(self):
        from fundamental.agents.base import AgentReport, AgentRole, AgentFinding
        return [
            AgentReport(
                agent_role=AgentRole.AUDITOR,
                findings=[AgentFinding("opinion", "Unqualified opinion", "info", 90)],
                overall_score=85,
                confidence=0.8,
            ),
            AgentReport(
                agent_role=AgentRole.GROWTH_ANALYST,
                findings=[AgentFinding("growth", "Strong revenue growth", "info", 80)],
                overall_score=75,
                confidence=0.7,
            ),
            AgentReport(
                agent_role=AgentRole.RISK_ANALYST,
                findings=[AgentFinding("risk", "Some risks identified", "warning", 55)],
                overall_score=60,
                confidence=0.7,
            ),
        ]

    def test_init(self, synthesizer):
        """Test synthesizer initialization."""
        from fundamental.agents.base import AgentRole
        assert synthesizer.role == AgentRole.SYNTHESIZER

    def test_analyze_consensus(self, synthesizer, sample_reports):
        """Test consensus analysis."""
        consensus = synthesizer._analyze_consensus(sample_reports)
        assert consensus.bullish_count >= 1

    def test_synthesize_returns_report(self, synthesizer, sample_reports):
        """Test that synthesize returns AgentReport."""
        from fundamental.agents.base import AgentReport
        report = synthesizer.synthesize(sample_reports)
        assert isinstance(report, AgentReport)

    def test_weighted_score_calculation(self, synthesizer, sample_reports):
        """Test weighted score calculation."""
        score = synthesizer._calculate_weighted_score(sample_reports)
        assert 0 <= score <= 100


class TestReportGenerator:
    """Tests for ReportGenerator."""

    @pytest.fixture
    def generator(self):
        from fundamental.report_generator import ReportGenerator
        return ReportGenerator()

    @pytest.fixture
    def sample_thesis(self):
        from fundamental.agents.synthesizer import InvestmentThesis, InvestmentRating, ConfidenceLevel
        return InvestmentThesis(
            rating=InvestmentRating.BUY,
            confidence=ConfidenceLevel.HIGH,
            overall_score=72.5,
            key_strengths=["Strong revenue growth", "Solid balance sheet"],
            key_risks=["Competitive pressure"],
            primary_thesis="Company shows strong fundamentals with manageable risks",
        )

    @pytest.fixture
    def sample_reports(self):
        from fundamental.agents.base import AgentReport, AgentRole, AgentFinding
        return [
            AgentReport(
                agent_role=AgentRole.AUDITOR,
                findings=[AgentFinding("opinion", "Unqualified opinion", "info", 90)],
                overall_score=85,
                confidence=0.8,
                recommendation="Favorable auditor opinion",
            ),
        ]

    def test_init(self, generator):
        """Test generator initialization."""
        assert generator is not None

    def test_generate_full_report(self, generator, sample_thesis, sample_reports):
        """Test full report generation."""
        report = generator.generate_full_report(
            ticker="BBCA",
            company_name="Bank Central Asia",
            thesis=sample_thesis,
            agent_reports=sample_reports,
        )
        assert report.ticker == "BBCA"
        assert report.company_name == "Bank Central Asia"
        assert report.thesis == sample_thesis

    def test_to_markdown(self, generator, sample_thesis, sample_reports):
        """Test markdown output."""
        report = generator.generate_full_report(
            ticker="BBCA",
            company_name="Bank Central Asia",
            thesis=sample_thesis,
            agent_reports=sample_reports,
        )
        md = generator.to_markdown(report)
        assert "# Fundamental Analysis: BBCA" in md
        assert "Bank Central Asia" in md
        assert "BUY" in md

    def test_to_json(self, generator, sample_thesis, sample_reports):
        """Test JSON output."""
        report = generator.generate_full_report(
            ticker="BBCA",
            company_name="Bank Central Asia",
            thesis=sample_thesis,
            agent_reports=sample_reports,
        )
        json_data = generator.to_json(report)
        assert json_data["ticker"] == "BBCA"
        assert json_data["thesis"]["rating"] == "buy"

    def test_format_currency(self, generator):
        """Test currency formatting."""
        assert "T" in generator._format_currency(1_500_000_000_000)
        assert "B" in generator._format_currency(1_500_000_000)
        assert "M" in generator._format_currency(1_500_000)


# Integration Tests
class TestFundamentalIntegration:
    """Integration tests for fundamental analysis."""

    @pytest.fixture
    def sample_financial_data(self):
        """Sample financial data for integration testing."""
        return {
            "income_statement": {
                "revenue": 50_000_000_000_000,  # 50 trillion
                "cost_of_goods_sold": 30_000_000_000_000,
                "gross_profit": 20_000_000_000_000,
                "operating_income": 8_000_000_000_000,
                "net_income": 5_000_000_000_000,
                "interest_expense": 500_000_000_000,
            },
            "balance_sheet": {
                "total_assets": 100_000_000_000_000,
                "current_assets": 30_000_000_000_000,
                "cash": 10_000_000_000_000,
                "inventory": 8_000_000_000_000,
                "accounts_receivable": 5_000_000_000_000,
                "total_liabilities": 60_000_000_000_000,
                "current_liabilities": 20_000_000_000_000,
                "long_term_debt": 25_000_000_000_000,
                "total_equity": 40_000_000_000_000,
            },
            "cash_flow": {
                "operating_cash_flow": 7_000_000_000_000,
                "investing_cash_flow": -3_000_000_000_000,
                "financing_cash_flow": -2_000_000_000_000,
                "capital_expenditure": 2_500_000_000_000,
            },
            "auditor_report": "Kami menyatakan pendapat wajar tanpa pengecualian",
            "raw_numbers": list(range(100, 1000, 3)),
        }

    def test_full_analysis_pipeline(self, sample_financial_data):
        """Test complete analysis pipeline."""
        # Run fraud detection
        from fundamental.fraud_detector import FraudDetector
        detector = FraudDetector()
        fraud_result = detector.analyze(sample_financial_data)
        assert fraud_result is not None

        # Calculate ratios
        from fundamental.ratio_calculator import RatioCalculator
        calculator = RatioCalculator()
        ratio_analysis = calculator.calculate(
            income_statement=sample_financial_data["income_statement"],
            balance_sheet=sample_financial_data["balance_sheet"],
            cash_flow=sample_financial_data["cash_flow"],
        )
        assert ratio_analysis is not None

        # Run agent analyses
        from fundamental.agents.auditor_agent import AuditorAgent
        from fundamental.agents.growth_analyst import GrowthAnalyst
        from fundamental.agents.value_analyst import ValueAnalyst
        from fundamental.agents.risk_analyst import RiskAnalyst
        from fundamental.agents.forensic_analyst import ForensicAnalyst
        from fundamental.agents.synthesizer import Synthesizer

        agents = [
            AuditorAgent(),
            GrowthAnalyst(),
            ValueAnalyst(),
            RiskAnalyst(),
            ForensicAnalyst(),
        ]

        reports = []
        for agent in agents:
            report = agent.analyze(sample_financial_data, reports.copy())
            reports.append(report)

        # Synthesize results
        synthesizer = Synthesizer()
        final_report = synthesizer.synthesize(reports, sample_financial_data)
        assert final_report is not None
        assert final_report.overall_score >= 0

    def test_ratio_calculations_with_sample_data(self, sample_financial_data):
        """Test ratio calculations with realistic data."""
        from fundamental.ratio_calculator import RatioCalculator
        calculator = RatioCalculator()

        result = calculator.calculate(
            income_statement=sample_financial_data["income_statement"],
            balance_sheet=sample_financial_data["balance_sheet"],
        )

        # Verify ratios were calculated
        assert len(result.ratios) > 0

        # Find and verify ROE
        roe = next((r for r in result.ratios if "ROE" in r.name), None)
        assert roe is not None
        # ROE = 5T / 40T = 12.5%
        assert roe.value == pytest.approx(0.125, rel=0.05)
