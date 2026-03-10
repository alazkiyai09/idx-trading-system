"""Tests for Sentiment page helper functions."""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from dashboard.pages.stock_detail_helpers import (
    format_sentiment_score,
    calculate_price_change,
)


class TestSentimentFormatting:
    """Tests for sentiment formatting functions."""

    def test_sentiment_label_bullish(self):
        """Test bullish sentiment label."""
        result = format_sentiment_score(75)
        assert result["label"] == "Bullish"
        assert result["emoji"] == "🟢"

    def test_sentiment_label_bearish(self):
        """Test bearish sentiment label."""
        result = format_sentiment_score(20)
        assert result["label"] == "Bearish"
        assert result["emoji"] == "🔴"

    def test_sentiment_label_neutral(self):
        """Test neutral sentiment label."""
        result = format_sentiment_score(50)
        assert result["label"] == "Neutral"
        assert result["emoji"] == "🟡"

    def test_sentiment_score_boundaries(self):
        """Test sentiment score boundary conditions."""
        # Exactly at boundaries
        assert format_sentiment_score(70)["label"] == "Bullish"
        assert format_sentiment_score(55)["label"] == "Slightly Bullish"
        assert format_sentiment_score(45)["label"] == "Neutral"
        assert format_sentiment_score(30)["label"] == "Slightly Bearish"

    def test_sentiment_score_extremes(self):
        """Test sentiment score extremes."""
        assert format_sentiment_score(0)["label"] == "Bearish"
        assert format_sentiment_score(100)["label"] == "Bullish"


class TestSectorSentimentAggregation:
    """Tests for sector sentiment aggregation logic."""

    def test_calculate_average_sentiment(self):
        """Test calculating average sentiment from articles."""
        articles = [
            {"sentiment_score": 70},
            {"sentiment_score": 60},
            {"sentiment_score": 80},
        ]

        avg = sum(a["sentiment_score"] for a in articles) / len(articles)
        assert avg == 70

    def test_weighted_sentiment_by_relevance(self):
        """Test weighted sentiment calculation by relevance."""
        articles = [
            {"sentiment_score": 70, "relevance": 0.9},
            {"sentiment_score": 50, "relevance": 0.3},
        ]

        total_weight = sum(a["relevance"] for a in articles)
        weighted_avg = sum(a["sentiment_score"] * a["relevance"] for a in articles) / total_weight
        assert weighted_avg == pytest.approx(65, rel=0.1)

    def test_sector_signal_generation(self):
        """Test generating signal from sector sentiment."""
        def generate_sector_signal(avg_score: float) -> str:
            if avg_score >= 65:
                return "BULLISH"
            elif avg_score <= 35:
                return "BEARISH"
            else:
                return "NEUTRAL"

        assert generate_sector_signal(70) == "BULLISH"
        assert generate_sector_signal(30) == "BEARISH"
        assert generate_sector_signal(50) == "NEUTRAL"


class TestArticleProcessing:
    """Tests for article processing logic."""

    def test_article_date_sorting(self):
        """Test that articles are sorted by date."""
        articles = [
            {"title": "Old", "published_at": "2024-01-01"},
            {"title": "New", "published_at": "2024-01-10"},
            {"title": "Mid", "published_at": "2024-01-05"},
        ]

        sorted_articles = sorted(
            articles,
            key=lambda x: x["published_at"],
            reverse=True
        )

        assert sorted_articles[0]["title"] == "New"
        assert sorted_articles[-1]["title"] == "Old"

    def test_article_deduplication(self):
        """Test deduplication of articles by URL."""
        articles = [
            {"url": "http://example.com/1", "title": "Article 1"},
            {"url": "http://example.com/2", "title": "Article 2"},
            {"url": "http://example.com/1", "title": "Article 1 Duplicate"},
        ]

        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)

        assert len(unique_articles) == 2

    def test_article_retention_filter(self):
        """Test filtering articles by retention days."""
        retention_days = 7
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        articles = [
            {"title": "Recent", "published_at": datetime.now() - timedelta(days=1)},
            {"title": "Old", "published_at": datetime.now() - timedelta(days=10)},
        ]

        filtered = [
            a for a in articles
            if a["published_at"] >= cutoff_date
        ]

        assert len(filtered) == 1
        assert filtered[0]["title"] == "Recent"


class TestThemeExtraction:
    """Tests for theme extraction from news."""

    def test_extract_themes_from_articles(self):
        """Test extracting themes from article titles."""
        articles = [
            {"title": "Bank Indonesia raises interest rates"},
            {"title": "BI rate hike impacts banking sector"},
            {"title": "Central bank maintains hawkish stance"},
        ]

        # Simple keyword counting
        themes = {}
        keywords = ["rate", "bank", "sector", "hike"]
        for article in articles:
            title_lower = article["title"].lower()
            for keyword in keywords:
                if keyword in title_lower:
                    themes[keyword] = themes.get(keyword, 0) + 1

        assert "rate" in themes
        assert "bank" in themes

    def test_theme_impact_direction(self):
        """Test determining theme impact direction."""
        def get_impact_direction(theme: str, sentiment: float) -> str:
            bullish_themes = ["earnings beat", "dividend", "expansion"]
            bearish_themes = ["loss", "fraud", "lawsuit"]

            if any(t in theme.lower() for t in bullish_themes) and sentiment > 50:
                return "positive"
            elif any(t in theme.lower() for t in bearish_themes) and sentiment < 50:
                return "negative"
            else:
                return "neutral"

        assert get_impact_direction("Earnings beat expectations", 70) == "positive"
        assert get_impact_direction("Company reports loss", 30) == "negative"
        assert get_impact_direction("Market update", 50) == "neutral"


class TestSentimentDataCleanup:
    """Tests for sentiment data cleanup logic."""

    def test_cleanup_by_days(self):
        """Test cleanup of old sentiment records."""
        retention_days = 30
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        records = [
            {"date": datetime.now() - timedelta(days=5), "score": 70},
            {"date": datetime.now() - timedelta(days=35), "score": 60},
            {"date": datetime.now() - timedelta(days=60), "score": 50},
        ]

        records_to_keep = [r for r in records if r["date"] >= cutoff_date]

        assert len(records_to_keep) == 1

    def test_cleanup_preserves_recent(self):
        """Test that cleanup preserves recent data."""
        retention_days = 7
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        records = [
            {"date": datetime.now() - timedelta(days=1)},
            {"date": datetime.now() - timedelta(days=3)},
            {"date": datetime.now() - timedelta(days=5)},
        ]

        records_to_keep = [r for r in records if r["date"] >= cutoff_date]

        assert len(records_to_keep) == 3


class TestHeatmapDataProcessing:
    """Tests for sector heatmap data processing."""

    def test_sector_aggregation(self):
        """Test aggregating sentiment by sector."""
        data = pd.DataFrame([
            {"sector": "Financials", "sentiment_score": 70},
            {"sector": "Financials", "sentiment_score": 60},
            {"sector": "Technology", "sentiment_score": 50},
            {"sector": "Technology", "sentiment_score": 80},
        ])

        sector_avg = data.groupby("sector")["sentiment_score"].mean()

        assert sector_avg["Financials"] == 65
        assert sector_avg["Technology"] == 65

    def test_sector_article_count(self):
        """Test counting articles per sector."""
        data = pd.DataFrame([
            {"sector": "Financials", "article_id": 1},
            {"sector": "Financials", "article_id": 2},
            {"sector": "Technology", "article_id": 3},
        ])

        counts = data.groupby("sector").size()

        assert counts["Financials"] == 2
        assert counts["Technology"] == 1

    def test_treemap_size_calculation(self):
        """Test calculating treemap block sizes."""
        sectors = pd.DataFrame([
            {"sector": "Financials", "avg_score": 70, "article_count": 20},
            {"sector": "Technology", "avg_score": 50, "article_count": 10},
        ])

        # Size based on article count
        total_articles = sectors["article_count"].sum()
        sectors["size_pct"] = sectors["article_count"] / total_articles * 100

        assert sectors.loc[sectors["sector"] == "Financials", "size_pct"].values[0] == pytest.approx(66.67, rel=0.1)
