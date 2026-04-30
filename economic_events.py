import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import xml.etree.ElementTree as ET

import aiohttp

try:
    import brotli
except Exception:  # noqa: BLE001
    brotli = None


LOGGER = logging.getLogger(__name__)

FOREX_FACTORY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
EVENT_KEYWORDS = (
    "cpi",
    "inflation",
    "fomc",
    "fed",
    "federal reserve",
    "interest rate",
    "powell",
)


@dataclass
class EconomicEventStatus:
    caution_mode: bool
    reason: str = ""
    upcoming_events: List[str] = field(default_factory=list)


class EconomicCalendarChecker:
    """Checks for high-impact macro events in the next N hours."""

    def __init__(self, lookahead_hours: int = 2, timeout_seconds: int = 10,
                 cache_minutes: int = 60) -> None:
        self.lookahead_hours = max(1, int(lookahead_hours))
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._cache_ttl = timedelta(minutes=cache_minutes)
        self._cached_result: Optional[EconomicEventStatus] = None
        self._cached_at: Optional[datetime] = None

    async def check_high_impact_today(self) -> EconomicEventStatus:
        now_utc = datetime.now(timezone.utc)

        # Return cached result if still fresh.
        if (self._cached_result is not None
                and self._cached_at is not None
                and (now_utc - self._cached_at) < self._cache_ttl):
            return self._cached_result

        window_end = now_utc + timedelta(hours=self.lookahead_hours)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                headers = {
                    # Avoid unsupported content-encoding surprises.
                    "Accept-Encoding": "identity",
                    "User-Agent": "Mozilla/5.0 (compatible; EconomicCalendarChecker/1.0)",
                }
                async with session.get(FOREX_FACTORY_CALENDAR_URL, headers=headers) as resp:
                    if resp.status != 200:
                        LOGGER.warning("Economic calendar HTTP %s", resp.status)
                        return EconomicEventStatus(caution_mode=False)

                    body = await resp.read()
                    encoding = (resp.headers.get("Content-Encoding") or "").lower()
                    if encoding == "br":
                        if brotli is None:
                            LOGGER.warning("Calendar returned br encoding but brotli is unavailable")
                            return EconomicEventStatus(caution_mode=False)
                        body = brotli.decompress(body)

                    xml_body = body.decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Economic calendar fetch failed: %s", exc)
            return EconomicEventStatus(caution_mode=False)

        try:
            root = ET.fromstring(xml_body)
        except ET.ParseError as exc:
            LOGGER.warning("Economic calendar parse failed: %s", exc)
            return EconomicEventStatus(caution_mode=False)

        upcoming: List[str] = []

        for event in root.findall(".//event"):
            title = (event.findtext("title") or "").strip()
            impact = (event.findtext("impact") or event.findtext("impact_title") or "").strip()
            country = (event.findtext("country") or "").strip()

            if not self._is_high_impact(impact):
                continue
            if not self._matches_keywords(title):
                continue

            event_dt = self._extract_event_datetime(event, now_utc)
            if event_dt is None:
                continue

            # Requirement: only high-impact events happening today.
            if event_dt.date() != now_utc.date():
                continue

            if now_utc <= event_dt <= window_end:
                upcoming.append(f"{title} ({country}) at {event_dt.strftime('%H:%M')} UTC")

        if upcoming:
            reason = f"High-impact event within {self.lookahead_hours}h"
            result = EconomicEventStatus(caution_mode=True, reason=reason, upcoming_events=upcoming)
        else:
            result = EconomicEventStatus(caution_mode=False)

        self._cached_result = result
        self._cached_at = now_utc
        return result

    @staticmethod
    def _is_high_impact(impact: str) -> bool:
        impact_l = impact.lower()
        return "high" in impact_l or impact_l == "3"

    @staticmethod
    def _matches_keywords(title: str) -> bool:
        title_l = title.lower()
        return any(keyword in title_l for keyword in EVENT_KEYWORDS)

    def _extract_event_datetime(self, event: ET.Element, now_utc: datetime) -> Optional[datetime]:
        timestamp_text = (event.findtext("timestamp") or "").strip()
        if timestamp_text:
            maybe_dt = self._parse_timestamp(timestamp_text)
            if maybe_dt is not None:
                return maybe_dt

        date_text = (event.findtext("date") or "").strip()
        time_text = (event.findtext("time") or "").strip()
        return self._parse_date_time(date_text, time_text, now_utc)

    @staticmethod
    def _parse_timestamp(value: str) -> Optional[datetime]:
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return None

        try:
            ts = int(digits)
            # Handle millisecond timestamps.
            if ts > 10_000_000_000:
                ts = ts // 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _parse_date_time(date_text: str, time_text: str, now_utc: datetime) -> Optional[datetime]:
        if not date_text or not time_text:
            return None

        time_l = time_text.lower()
        if time_l in {"all day", "day 1", "day 2", "tentative"}:
            return None

        normalized_time = time_text.strip().replace(" ", "")

        # Try formats commonly seen in economic calendar feeds.
        candidate_strings = [
            f"{date_text} {normalized_time}",
            f"{date_text} {time_text}",
        ]
        formats = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %I:%M%p",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y %I:%M%p",
            "%b %d %Y %H:%M",
            "%b %d %Y %I:%M%p",
            "%b %d %H:%M",
            "%b %d %I:%M%p",
        ]

        for raw in candidate_strings:
            for fmt in formats:
                try:
                    parsed = datetime.strptime(raw, fmt)
                    if "%Y" not in fmt:
                        parsed = parsed.replace(year=now_utc.year)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None
