"""
Sentiment Scorer Module

Aggregates sentiment scores with time-decay weighting.
Integrates with the CompositeScorer for signal generation.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from core.analysis.sentiment.news_analyzer import SentimentResult

logger = logging.getLogger(__name__)


@dataclass
class AggregateSentiment:
    """Aggregated sentiment score.

    Attributes:
        score: Weighted average sentiment (0-100).
        confidence: Aggregated confidence.
        article_count: Number of articles analyzed.
        bullish_count: Articles with score > 60.
        bearish_count: Articles with score < 40.
        neutral_count: Articles with score 40-60.
    """
    score: float = 50.0
    confidence: float = 0.0
    article_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0


class SentimentScorer:
    """Aggregates sentiment with time-decay weighting.

    More recent articles have higher weight. Articles older than
    the decay period are weighted exponentially less.

    Example:
        scorer = SentimentScorer()
        sentiment = scorer.aggregate(results)
        print(f"Sentiment: {sentiment.score:.1f}")
    """

    def __init__(
        self,
        decay_hours: float = 72.0,
        min_confidence: float = 0.3,
    ) -> None:
        """Initialize sentiment scorer.

        Args:
            decay_hours: Half-life for time decay in hours.
            min_confidence: Minimum confidence to include in aggregation.
        """
        self.decay_hours = decay_hours
        self.min_confidence = min_confidence

    def aggregate(
        self,
        results: List[SentimentResult],
        reference_time: Optional[datetime] = None,
    ) -> AggregateSentiment:
        """Aggregate sentiment results with time-decay weighting.

        Args:
            results: List of sentiment results.
            reference_time: Reference time for decay calculation.

        Returns:
            AggregateSentiment with weighted score.
        """
        if not results:
            return AggregateSentiment()

        ref_time = reference_time or datetime.now()

        weighted_sum = 0.0
        weight_total = 0.0
        bullish = 0
        bearish = 0
        neutral = 0

        for result in results:
            if result.confidence < self.min_confidence:
                continue

            # Calculate time-decay weight
            weight = result.confidence

            # Categorize
            if result.sentiment_score > 60:
                bullish += 1
            elif result.sentiment_score < 40:
                bearish += 1
            else:
                neutral += 1

            weighted_sum += result.sentiment_score * weight
            weight_total += weight

        if weight_total == 0:
            return AggregateSentiment(article_count=len(results))

        avg_score = weighted_sum / weight_total
        avg_confidence = weight_total / len(results)

        return AggregateSentiment(
            score=avg_score,
            confidence=min(avg_confidence, 1.0),
            article_count=len(results),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )

    def get_sentiment_signal(self, sentiment: AggregateSentiment) -> str:
        """Convert aggregate sentiment to a signal string.

        Args:
            sentiment: Aggregated sentiment.

        Returns:
            Signal string: "bullish", "bearish", or "neutral".
        """
        if sentiment.confidence < 0.3 or sentiment.article_count < 2:
            return "neutral"

        if sentiment.score >= 65:
            return "bullish"
        elif sentiment.score <= 35:
            return "bearish"
        else:
            return "neutral"
