"""
News Fetcher Module

Fetches news articles from various Indonesian financial news sources.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """A fetched news article.

    Attributes:
        title: Article headline.
        source: News source name.
        url: Article URL.
        content: Article text content (may be excerpt).
        published_at: Publication timestamp.
        symbol: Related stock symbol.
    """
    title: str
    source: str
    url: str = ""
    content: str = ""
    published_at: Optional[datetime] = None
    symbol: str = ""


class NewsFetcher:
    """Fetches news from Indonesian financial news sources.

    Supports Google News RSS, and can be extended to add
    CNBC Indonesia, Kontan, and other sources.

    Example:
        fetcher = NewsFetcher()
        articles = fetcher.fetch("BBCA", max_articles=10)
    """

    def __init__(self, timeout: int = 10) -> None:
        """Initialize news fetcher.

        Args:
            timeout: Request timeout in seconds.
        """
        self._timeout = timeout

    def fetch(
        self,
        symbol: str,
        max_articles: int = 10,
        sources: Optional[List[str]] = None,
    ) -> List[NewsArticle]:
        """Fetch news articles for a symbol.

        Args:
            symbol: Stock symbol (e.g., "BBCA").
            max_articles: Maximum articles to fetch.
            sources: Optional list of sources to use.

        Returns:
            List of NewsArticle objects.
        """
        all_articles = []

        # Fetch from Google News RSS
        try:
            google_articles = self._fetch_google_news(symbol, max_articles)
            all_articles.extend(google_articles)
        except Exception as e:
            logger.warning(f"Google News fetch failed for {symbol}: {e}")

        # Limit to max_articles
        return all_articles[:max_articles]

    def _fetch_google_news(
        self,
        symbol: str,
        max_articles: int = 10,
    ) -> List[NewsArticle]:
        """Fetch from Google News RSS.

        Args:
            symbol: Stock symbol.
            max_articles: Maximum articles.

        Returns:
            List of NewsArticle objects.
        """
        try:
            import requests
            from xml.etree import ElementTree

            # Search for Indonesian stock news
            query = quote(f"{symbol} saham Indonesia")
            url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"

            response = requests.get(url, timeout=self._timeout)
            response.raise_for_status()

            root = ElementTree.fromstring(response.content)
            articles = []

            for item in root.findall('.//item')[:max_articles]:
                title = item.findtext('title', '')
                link = item.findtext('link', '')
                pub_date_str = item.findtext('pubDate', '')
                source = item.findtext('source', 'Google News')

                pub_date = None
                if pub_date_str:
                    try:
                        pub_date = datetime.strptime(
                            pub_date_str, "%a, %d %b %Y %H:%M:%S %Z"
                        )
                    except ValueError:
                        pass

                articles.append(NewsArticle(
                    title=title,
                    source=source,
                    url=link,
                    published_at=pub_date,
                    symbol=symbol,
                ))

            return articles

        except ImportError:
            logger.warning("requests package required for news fetching")
            return []
        except Exception as e:
            logger.warning(f"Google News RSS fetch error: {e}")
            return []
