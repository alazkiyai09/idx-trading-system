"""
News Analyzer Module

Analyzes sentiment of news articles using LLM.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.analysis.sentiment.news_fetcher import NewsArticle

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single article.

    Attributes:
        article_title: Title of the analyzed article.
        sentiment_score: Sentiment score (0-100, 50=neutral).
        confidence: Confidence in the assessment (0-1).
        key_topics: Key topics identified.
        impact_assessment: Description of likely market impact.
    """
    article_title: str
    sentiment_score: float = 50.0
    confidence: float = 0.5
    key_topics: List[str] = None
    impact_assessment: str = ""

    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []


class NewsAnalyzer:
    """Analyzes sentiment of news articles using LLM.

    Example:
        analyzer = NewsAnalyzer()
        results = analyzer.analyze_articles(articles)
    """

    def __init__(self, llm_client=None) -> None:
        """Initialize news analyzer.

        Args:
            llm_client: LLM client for sentiment analysis.
                        If None, creates default client.
        """
        self._client = llm_client

    def _get_client(self):
        """Lazily initialize LLM client."""
        if self._client is None:
            try:
                from llm import create_client
                self._client = create_client()
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {e}")
        return self._client

    def analyze_article(self, article: NewsArticle) -> SentimentResult:
        """Analyze sentiment of a single article.

        Args:
            article: News article to analyze.

        Returns:
            SentimentResult with score and analysis.
        """
        client = self._get_client()

        if client is None or not client.is_available():
            logger.debug("LLM not available, using neutral sentiment")
            return SentimentResult(
                article_title=article.title,
                sentiment_score=50.0,
                confidence=0.1,
            )

        try:
            from llm import PromptManager, LLMConfig
            from llm.response_validator import ResponseValidator

            pm = PromptManager()
            prompt = pm.render(
                "sentiment_analysis",
                symbol=article.symbol,
                title=article.title,
                source=article.source,
                content=article.content or article.title,
            )

            config = LLMConfig(temperature=0.3, max_tokens=500)
            data = client.generate_json(prompt, config=config)

            sentiment_score = ResponseValidator.validate_score_range(
                data.get("sentiment_score", 50.0),
                field_name="sentiment_score",
            )
            confidence = ResponseValidator.validate_score_range(
                data.get("confidence", 0.5),
                min_val=0.0, max_val=1.0,
                field_name="confidence",
            )

            return SentimentResult(
                article_title=article.title,
                sentiment_score=sentiment_score,
                confidence=confidence,
                key_topics=data.get("key_topics", []),
                impact_assessment=data.get("impact_assessment", ""),
            )

        except Exception as e:
            logger.warning(f"Sentiment analysis failed for '{article.title}': {e}")
            return SentimentResult(
                article_title=article.title,
                sentiment_score=50.0,
                confidence=0.1,
            )

    def analyze_articles(
        self,
        articles: List[NewsArticle],
    ) -> List[SentimentResult]:
        """Analyze sentiment of multiple articles.

        Args:
            articles: List of articles to analyze.

        Returns:
            List of SentimentResult objects.
        """
        results = []
        for article in articles:
            result = self.analyze_article(article)
            results.append(result)
        return results
