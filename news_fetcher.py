"""
Sentiment Engine — CryptoPanic news fetcher.

Fetches the latest BTC and broader crypto news, filters by macro keywords
(Trump, Fed, Inflation, SEC), scores each article from its vote metadata,
and aggregates the results into a NewsSentiment object.

Usage:
    fetcher = NewsFetcher(auth_token=settings.cryptopanic_token)
    sentiment = await fetcher.fetch_sentiment()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List

import aiohttp


LOGGER = logging.getLogger(__name__)

CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1/posts"

# Articles must match at least one of these keywords to be counted.
FILTER_KEYWORDS: List[str] = ["trump", "fed", "inflation", "sec"]

# Crypto references used to identify Trump+Crypto positive articles.
_CRYPTO_REFS = {
    "crypto",
    "bitcoin",
    "btc",
    "defi",
    "blockchain",
    "digital asset",
    "digital assets",
    "coin",
    "token",
    "ethereum",
    "altcoin",
}


@dataclass
class NewsSentiment:
    """Aggregate sentiment derived from keyword-filtered news headlines."""

    # Normalised sentiment: -100 (very negative) to +100 (very positive).
    score: int
    # Human-readable label: "positive", "negative", or "neutral".
    label: str
    # Which filter keywords were matched at least once.
    matched_keywords: List[str] = field(default_factory=list)
    # True when ≥1 article mentions Trump positively alongside crypto.
    trump_crypto_boost_eligible: bool = False
    # Up to 5 matched headlines (for log context lines).
    headlines: List[str] = field(default_factory=list)


class NewsFetcher:
    """Async news sentiment fetcher backed by CryptoPanic's public API."""

    def __init__(self, auth_token: str = "", timeout_seconds: int = 10) -> None:
        self.auth_token = auth_token
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._cooldown_until: datetime | None = None
        self._warned_missing_token = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_posts(
        self,
        session: aiohttp.ClientSession,
        currencies: str | None = "BTC",
        filter_name: str = "hot",
    ) -> List[dict]:
        """
        Fetch the 'hot' post list from CryptoPanic.

        Pass currencies="" to get broader (non-currency-specific) posts.
        """
        params: dict = {"filter": filter_name}
        if self.auth_token:
            params["auth_token"] = self.auth_token
        else:
            params["public"] = "true"
        if currencies:
            params["currencies"] = currencies

        last_status = None
        last_url = ""
        try:
            # Try canonical endpoint first, then trailing-slash fallback.
            for url in (CRYPTOPANIC_BASE, f"{CRYPTOPANIC_BASE}/"):
                last_url = url
                async with session.get(url, params=params) as resp:
                    last_status = resp.status
                    if resp.status == 404:
                        continue
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                        cooldown_seconds = 60
                        if retry_after and retry_after.isdigit():
                            cooldown_seconds = max(10, int(retry_after))
                        self._cooldown_until = datetime.now(timezone.utc) + timedelta(
                            seconds=cooldown_seconds
                        )
                        LOGGER.warning(
                            "CryptoPanic Rate Limit hit. Cooling down for %ss...",
                            cooldown_seconds,
                        )
                        return []
                    if resp.status != 200:
                        LOGGER.warning("CryptoPanic returned HTTP %s for %s", resp.status, url)
                        return []
                    data = await resp.json(content_type=None)
                    return data.get("results", [])

            LOGGER.warning(
                "CryptoPanic endpoint failed (last_status=%s, last_url=%s)",
                last_status,
                last_url,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("CryptoPanic fetch error: %s", exc)
            return []

    @staticmethod
    def _article_score(article: dict) -> float:
        """
        Return a normalised sentiment score for one article in [-1, +1].

        Uses CryptoPanic votes: positive/liked/important vs
        negative/disliked/toxic.
        """
        votes = article.get("votes") or {}
        pos = (
            votes.get("positive", 0)
            + votes.get("liked", 0)
            + votes.get("important", 0)
        )
        neg = (
            votes.get("negative", 0)
            + votes.get("disliked", 0)
            + votes.get("toxic", 0)
        )
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total  # -1.0 … +1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_sentiment(self) -> NewsSentiment:
        """
        Fetch BTC-tagged and general crypto news, keep only articles whose
        title contains at least one FILTER_KEYWORD, then compute an
        aggregate sentiment score.

        Returns NewsSentiment(score=0, label="neutral") on any API failure.
        """
        if not self.auth_token:
            if not self._warned_missing_token:
                LOGGER.warning(
                    "CryptoPanic token missing. News sentiment disabled; using neutral sentiment."
                )
                self._warned_missing_token = True
            return NewsSentiment(score=0, label="neutral")

        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            return NewsSentiment(score=0, label="neutral")

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            btc_posts_result, general_posts_result = await asyncio.gather(
                self._fetch_posts(session, currencies="BTC", filter_name="hot"),
                self._fetch_posts(session, currencies=None, filter_name="important"),
                return_exceptions=True,
            )

        btc_posts: List[dict] = (
            [] if isinstance(btc_posts_result, Exception) else btc_posts_result
        )
        general_posts: List[dict] = (
            [] if isinstance(general_posts_result, Exception) else general_posts_result
        )

        if isinstance(btc_posts_result, Exception):
            LOGGER.warning("BTC news fetch failed: %s", btc_posts_result)
        if isinstance(general_posts_result, Exception):
            LOGGER.warning("General news fetch failed: %s", general_posts_result)

        # De-duplicate by article id before processing.
        seen_ids: set = set()
        all_posts: List[dict] = []
        for post in btc_posts + general_posts:
            pid = post.get("id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_posts.append(post)

        # Filter and score.
        scores: List[float] = []
        matched_keywords: List[str] = []
        headlines: List[str] = []
        trump_positive_count = 0

        for article in all_posts:
            title = article.get("title", "").lower()

            # Keep only articles relevant to our macro keywords.
            kws_found = [kw for kw in FILTER_KEYWORDS if kw in title]
            if not kws_found:
                continue

            article_score = self._article_score(article)
            scores.append(article_score)
            headlines.append(article.get("title", ""))

            for kw in kws_found:
                if kw not in matched_keywords:
                    matched_keywords.append(kw)

            # Trump + Crypto positive — eligible for the +20 confidence boost.
            has_trump = "trump" in title
            has_crypto_ref = any(ref in title for ref in _CRYPTO_REFS)
            if has_trump and has_crypto_ref and article_score > 0:
                trump_positive_count += 1

        if not scores:
            LOGGER.info("No news articles matched the filter keywords this cycle")
            return NewsSentiment(score=0, label="neutral")

        avg_score = sum(scores) / len(scores)          # -1.0 … +1.0
        normalised = int(avg_score * 100)               # -100 … +100

        label: str
        if normalised > 10:
            label = "positive"
        elif normalised < -10:
            label = "negative"
        else:
            label = "neutral"

        trump_eligible = trump_positive_count >= 1

        LOGGER.info(
            "News sentiment: score=%s label=%s keywords=%s trump_boost=%s "
            "matched_articles=%s",
            normalised,
            label,
            matched_keywords,
            trump_eligible,
            len(scores),
        )

        return NewsSentiment(
            score=normalised,
            label=label,
            matched_keywords=matched_keywords,
            trump_crypto_boost_eligible=trump_eligible,
            headlines=headlines[:5],
        )
