import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import aiohttp

from goinglass import fetch_all_heatmaps


LOGGER = logging.getLogger(__name__)

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"
COINALYZE_BASE = "https://api.coinalyze.net/v1"


class CoinalyzeRateLimitError(Exception):
    pass


class CoinalyzeFetcher:
    """Best-effort Coinalyze data client with soft rate-limiting and safe fallbacks."""

    def __init__(self, api_key: str, timeout_seconds: int = 10, cooldown_seconds: float = 1.5):
        self.api_key = api_key.strip()
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.cooldown_seconds = max(0.1, float(cooldown_seconds))
        self._last_request_at: Optional[datetime] = None
        self._request_lock = asyncio.Lock()
        self._global_lock = asyncio.Lock()
        self._global_cache: Optional[Dict[str, float]] = None
        self._skip_cycle_due_to_rate_limit = False
        self._cooldown_until: Optional[datetime] = None

    def start_new_cycle(self) -> None:
        # Called once per scan cycle to force fresh global context each loop.
        self._global_cache = None
        self._skip_cycle_due_to_rate_limit = False

    async def fetch_market_metrics(self, symbol: str) -> Dict[str, float | bool]:
        # If no key is provided, keep strategy running with neutral defaults.
        if not self.api_key:
            return self._neutral_metrics(coinalyze_available=False)

        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            return self._neutral_metrics(coinalyze_available=False)
        if self._skip_cycle_due_to_rate_limit:
            return self._neutral_metrics(coinalyze_available=False)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                oi_task = self._fetch_aggregated_oi_change(session, symbol)
                funding_task = self._fetch_predicted_funding(session, symbol)
                global_task = self._fetch_global_metrics_once(session)

                oi_change, pred_funding, global_metrics = await asyncio.gather(
                    oi_task,
                    funding_task,
                    global_task,
                    return_exceptions=True,
                )

            if not isinstance(global_metrics, Exception):
                liq_24h_val = float(global_metrics.get("coinalyze_liquidations_24h_usd", 0.0))
                liq_1h_val = float(global_metrics.get("coinalyze_liquidations_1h_usd", 0.0))
                short_liq_1h_val = float(
                    global_metrics.get("coinalyze_short_liquidations_1h_usd", 0.0)
                )
                global_ls_val = float(global_metrics.get("coinalyze_global_long_short_ratio", 1.0))
                global_ok = bool(global_metrics.get("coinalyze_global_ok", False))
            else:
                liq_24h_val = 0.0
                liq_1h_val = 0.0
                short_liq_1h_val = 0.0
                global_ls_val = 1.0
                global_ok = False

            oi_ok = (not isinstance(oi_change, Exception)) and (oi_change is not None)
            funding_ok = (not isinstance(pred_funding, Exception)) and (pred_funding is not None)
            coinalyze_available = bool(oi_ok and funding_ok and global_ok)

            return {
                "coinalyze_available": coinalyze_available,
                "coinalyze_agg_oi_change_pct": 0.0 if not oi_ok else float(oi_change),
                "coinalyze_pred_funding_rate": (
                    0.0 if not funding_ok else float(pred_funding)
                ),
                "coinalyze_liquidations_24h_usd": liq_24h_val,
                "coinalyze_liquidations_1h_usd": liq_1h_val,
                "coinalyze_short_liquidations_1h_usd": short_liq_1h_val,
                "coinalyze_global_long_short_ratio": global_ls_val,
            }
        except CoinalyzeRateLimitError:
            self._skip_cycle_due_to_rate_limit = True
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=60)
            LOGGER.warning("Coinalyze Rate Limit hit. Cooling down for 60s...")
            return self._neutral_metrics(coinalyze_available=False)
        except Exception as exc:
            LOGGER.warning("Coinalyze fetch failed for %s: %s", symbol, exc)
            return self._neutral_metrics(coinalyze_available=False)

    def _neutral_metrics(self, coinalyze_available: bool) -> Dict[str, float | bool]:
        return {
            "coinalyze_available": coinalyze_available,
            "coinalyze_agg_oi_change_pct": 0.0,
            "coinalyze_pred_funding_rate": 0.0,
            "coinalyze_liquidations_24h_usd": 0.0,
            "coinalyze_liquidations_1h_usd": 0.0,
            "coinalyze_short_liquidations_1h_usd": 0.0,
            "coinalyze_global_long_short_ratio": 1.0,
        }

    @staticmethod
    def _to_coinalyze_symbol(symbol: str) -> str:
        # Coinalyze aggregated perpetual symbol format, e.g. BTCUSDT_PERP.A
        return f"{symbol}_PERP.A"

    @staticmethod
    def _extract_history_points(data: Dict | List) -> List:
        # Expected payload shape: [{"symbol": "...", "history": [...]}]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            history = data[0].get("history")
            if isinstance(history, list):
                return history
        return []

    @staticmethod
    def _extract_point_value(point: object) -> Optional[float]:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                return float(point[1])
            except (TypeError, ValueError):
                return None
        if isinstance(point, dict):
            for key in (
                "value",
                "v",
                "c",
                "close",
                "long_short_ratio",
                "open_interest",
                "oi",
                "l",
                "s",
            ):
                if key in point:
                    try:
                        return float(point[key])
                    except (TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _extract_point_short_liq_value(point: object) -> Optional[float]:
        if isinstance(point, dict):
            for key in ("s", "short", "shorts", "shorts_usd", "short_liq", "short_liquidation"):
                if key in point:
                    try:
                        return float(point[key])
                    except (TypeError, ValueError):
                        continue
        return None

    async def _throttle(self) -> None:
        # Hard throttle to avoid Coinalyze rate-limit hits.
        if self._last_request_at is None:
            self._last_request_at = datetime.now(timezone.utc)
            return
        elapsed = (datetime.now(timezone.utc) - self._last_request_at).total_seconds()
        if elapsed < self.cooldown_seconds:
            await asyncio.sleep(self.cooldown_seconds - elapsed)
        self._last_request_at = datetime.now(timezone.utc)

    async def _request_json(self, session: aiohttp.ClientSession, path: str, params: Dict) -> Dict | List:
        headers = {
            "api_key": self.api_key,
            # Prevent Coinalyze from returning Brotli-encoded responses that aiohttp
            # cannot decompress without the optional brotli native library.
            "Accept-Encoding": "gzip, deflate",
        }
        url = f"{COINALYZE_BASE}/{path.lstrip('/')}"
        async with self._request_lock:
            await self._throttle()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise CoinalyzeRateLimitError(path)
                if response.status != 200:
                    LOGGER.warning("Coinalyze %s failed (status=%s)", path, response.status)
                    return {}
                return await response.json(content_type=None)

    async def _fetch_global_metrics_once(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        if self._global_cache is not None:
            return self._global_cache

        async with self._global_lock:
            if self._global_cache is not None:
                return self._global_cache

            liq_24h, liq_1h, short_liq_1h, liq_ok = await self._fetch_liquidations_global(session)
            global_ls, ls_ok = await self._fetch_long_short_ratio_global(session)
            self._global_cache = {
                "coinalyze_liquidations_24h_usd": float(liq_24h),
                "coinalyze_liquidations_1h_usd": float(liq_1h),
                "coinalyze_short_liquidations_1h_usd": float(short_liq_1h),
                "coinalyze_global_long_short_ratio": float(global_ls),
                "coinalyze_global_ok": bool(liq_ok and ls_ok),
            }
            return self._global_cache

    async def _fetch_aggregated_oi_change(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        now = datetime.now(timezone.utc)
        frm = int((now - timedelta(minutes=15)).timestamp())
        to = int(now.timestamp())
        co_symbol = self._to_coinalyze_symbol(symbol)
        data = await self._request_json(
            session,
            "open-interest-history",
            {
                "symbols": co_symbol,
                "interval": "5min",
                "from": frm,
                "to": to,
                "convert_to_usd": "true",
            },
        )
        points = self._extract_history_points(data)
        if len(points) < 2:
            return None

        prev_val = self._extract_point_value(points[-2])
        last_val = self._extract_point_value(points[-1])
        if prev_val is None or last_val is None:
            return None
        if prev_val <= 0:
            return None
        return ((last_val - prev_val) / prev_val) * 100.0

    async def _fetch_predicted_funding(self, session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
        co_symbol = self._to_coinalyze_symbol(symbol)
        data = await self._request_json(session, "predicted-funding-rate", {"symbols": co_symbol})
        rows = data if isinstance(data, list) else []
        if not rows:
            return None
        row = rows[0] if isinstance(rows[0], dict) else {}
        return float(row.get("value", row.get("predicted_funding_rate", 0.0)) or 0.0)

    async def _fetch_liquidations(self, session: aiohttp.ClientSession, symbol: str) -> tuple[float, float]:
        now = datetime.now(timezone.utc)
        since_24h = int((now - timedelta(hours=24)).timestamp())
        since_1h = int((now - timedelta(hours=1)).timestamp())

        data = await self._request_json(
            session,
            "liquidation-history",
            {"symbols": symbol, "from": since_24h},
        )
        rows = data if isinstance(data, list) else []
        total_24h = 0.0
        total_1h = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            amount = float(row.get("value", row.get("liquidation_usd", 0.0)) or 0.0)
            ts = int(row.get("t", row.get("timestamp", 0)) or 0)
            total_24h += amount
            if ts >= since_1h:
                total_1h += amount
        return total_24h, total_1h

    async def _fetch_long_short_ratio(self, session: aiohttp.ClientSession, symbol: str) -> float:
        now = datetime.now(timezone.utc)
        frm = int((now - timedelta(minutes=15)).timestamp())
        to = int(now.timestamp())
        co_symbol = self._to_coinalyze_symbol(symbol)
        data = await self._request_json(
            session,
            "long-short-ratio-history",
            {"symbols": co_symbol, "interval": "5min", "from": frm, "to": to},
        )
        points = self._extract_history_points(data)
        if not points:
            return 1.0
        value = self._extract_point_value(points[-1])
        return 1.0 if value is None else value

    async def _fetch_liquidations_global(
        self,
        session: aiohttp.ClientSession,
    ) -> tuple[float, float, float, bool]:
        now = datetime.now(timezone.utc)
        since_24h = int((now - timedelta(hours=24)).timestamp())
        since_1h = int((now - timedelta(hours=1)).timestamp())
        to = int(now.timestamp())

        # Use BTC aggregated as global proxy when account tier does not expose broad aggregate calls.
        data = await self._request_json(
            session,
            "liquidation-history",
            {
                "symbols": self._to_coinalyze_symbol("BTCUSDT"),
                "interval": "1hour",
                "from": since_24h,
                "to": to,
                "convert_to_usd": "true",
            },
        )
        points = self._extract_history_points(data)
        ok = bool(points)
        total_24h = 0.0
        total_1h = 0.0
        short_total_1h = 0.0
        for row in points:
            amount = self._extract_point_value(row) or 0.0
            short_amount = self._extract_point_short_liq_value(row) or 0.0
            ts = 0
            if isinstance(row, (list, tuple)) and len(row) >= 1:
                try:
                    ts = int(row[0])
                except (TypeError, ValueError):
                    ts = 0
            elif isinstance(row, dict):
                ts = int(row.get("t", row.get("timestamp", 0)) or 0)
            total_24h += amount
            if ts >= since_1h:
                total_1h += amount
                short_total_1h += short_amount
        return total_24h, total_1h, short_total_1h, ok

    async def _fetch_long_short_ratio_global(self, session: aiohttp.ClientSession) -> tuple[float, bool]:
        now = datetime.now(timezone.utc)
        frm = int((now - timedelta(minutes=15)).timestamp())
        to = int(now.timestamp())
        data = await self._request_json(
            session,
            "long-short-ratio-history",
            {
                "symbols": self._to_coinalyze_symbol("BTCUSDT"),
                "interval": "5min",
                "from": frm,
                "to": to,
            },
        )
        points = self._extract_history_points(data)
        if not points:
            return 1.0, False
        value = self._extract_point_value(points[-1])
        if value is None:
            return 1.0, False
        return value, True


class BinanceFetcher:
    def __init__(
        self,
        timeout_seconds: int = 10,
        coinalyze_api_key: str = "",
        coinalyze_cooldown_seconds: float = 1.5,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.coinalyze = CoinalyzeFetcher(
            coinalyze_api_key,
            timeout_seconds=timeout_seconds,
            cooldown_seconds=coinalyze_cooldown_seconds,
        )
        self._heatmap_cache: Dict[str, Dict] = {}

    def start_cycle_cache(self) -> None:
        self.coinalyze.start_new_cycle()

    async def fetch_coinglass_heatmaps(self, symbols: List[str]) -> None:
        """Fetch Coinglass heatmaps for all symbols once per scan cycle.
        Results are stored in self._heatmap_cache and injected into snapshots."""
        try:
            api_key = os.getenv("COINGLASS_API_KEY", "")
            self._heatmap_cache = await fetch_all_heatmaps(symbols, api_key=api_key)
            LOGGER.info("Coinglass heatmaps fetched for %s symbols", len(self._heatmap_cache))
        except Exception as exc:
            LOGGER.warning("Coinglass heatmap batch fetch failed: %s", exc)
            self._heatmap_cache = {}

    def get_heatmap_for(self, symbol: str) -> Dict:
        """Return cached heatmap data for *symbol*."""
        return self._heatmap_cache.get(symbol, {"high_density_zones": [], "clusters": []})

    async def fetch_pair_snapshot(
        self,
        symbol: str,
        interval: str,
        limit: int,
        higher_tf_interval: str,
        higher_tf_limit: int,
    ) -> Optional[Dict]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            klines_task = self._fetch_klines(session, symbol, interval, limit)
            htf_klines_task = self._fetch_klines(session, symbol, higher_tf_interval, higher_tf_limit)
            tf_4h_task = self._fetch_klines(session, symbol, "4h", 250)
            tf_1h_task = self._fetch_klines(session, symbol, "1h", 250)
            tf_5m_task = self._fetch_klines(session, symbol, "5m", 250)
            funding_task = self._fetch_funding_rate(session, symbol)
            oi_task = self._fetch_open_interest(session, symbol)
            oi_change_task = self._fetch_open_interest_change(session, symbol)
            top_ls_task = self._fetch_top_trader_ls_ratio(session, symbol)
            taker_task = self._fetch_taker_buy_sell_ratio(session, symbol)
            coinalyze_task = self.coinalyze.fetch_market_metrics(symbol)
            orderbook_task = self._fetch_orderbook_imbalance(session, symbol, depth=20)

            (
                klines,
                htf_klines,
                tf_4h_klines,
                tf_1h_klines,
                tf_5m_klines,
                funding_rate,
                open_interest,
                oi_change_pct,
                top_trader_ls_ratio,
                taker_buy_sell_ratio,
                coinalyze_metrics,
                orderbook_metrics,
            ) = await asyncio.gather(
                klines_task,
                htf_klines_task,
                tf_4h_task,
                tf_1h_task,
                tf_5m_task,
                funding_task,
                oi_task,
                oi_change_task,
                top_ls_task,
                taker_task,
                coinalyze_task,
                orderbook_task,
                return_exceptions=True,
            )

        if isinstance(klines, Exception) or not klines:
            LOGGER.warning("Failed to fetch klines for %s", symbol)
            return None

        if isinstance(funding_rate, Exception):
            LOGGER.warning("Funding fetch error for %s: %s", symbol, funding_rate)
            funding_rate = 0.0
        if isinstance(open_interest, Exception):
            LOGGER.warning("Open interest fetch error for %s: %s", symbol, open_interest)
            open_interest = 0.0
        if isinstance(oi_change_pct, Exception):
            LOGGER.warning("OI change fetch error for %s: %s", symbol, oi_change_pct)
            oi_change_pct = 0.0
        if isinstance(top_trader_ls_ratio, Exception):
            LOGGER.warning("Top trader ratio fetch error for %s: %s", symbol, top_trader_ls_ratio)
            top_trader_ls_ratio = 1.0
        if isinstance(taker_buy_sell_ratio, Exception):
            LOGGER.warning("Taker ratio fetch error for %s: %s", symbol, taker_buy_sell_ratio)
            taker_buy_sell_ratio = 1.0
        if isinstance(coinalyze_metrics, Exception):
            LOGGER.warning("Coinalyze metrics error for %s: %s", symbol, coinalyze_metrics)
            coinalyze_metrics = self.coinalyze._neutral_metrics(coinalyze_available=False)
        if isinstance(htf_klines, Exception):
            htf_klines = []
        if isinstance(tf_4h_klines, Exception):
            tf_4h_klines = []
        if isinstance(tf_1h_klines, Exception):
            tf_1h_klines = []
        if isinstance(tf_5m_klines, Exception):
            tf_5m_klines = []
        if isinstance(orderbook_metrics, Exception):
            LOGGER.warning("Order book metrics error for %s: %s", symbol, orderbook_metrics)
            orderbook_metrics = {"ob_bid_usd": 0.0, "ob_ask_usd": 0.0, "ob_bid_pct": 50.0, "ob_ask_pct": 50.0}

        if float(open_interest) == 0.0:
            LOGGER.warning("Open interest is zero or missing for %s", symbol)
        if float(funding_rate) == 0.0:
            LOGGER.warning("Funding rate is zero or missing for %s", symbol)

        # Fallback rule: if Coinalyze funding is missing, keep Binance funding as source of truth.
        coinalyze_pred_funding_rate = float(coinalyze_metrics.get("coinalyze_pred_funding_rate", 0.0))
        if coinalyze_pred_funding_rate == 0.0:
            coinalyze_pred_funding_rate = float(funding_rate)

        # Inject Coinglass heatmap data from the per-cycle cache.
        heatmap = self.get_heatmap_for(symbol)

        return {
            "symbol": symbol,
            "klines": klines,
            "higher_tf_klines": htf_klines,
            "tf_4h_klines": tf_4h_klines,
            "tf_1h_klines": tf_1h_klines,
            "tf_5m_klines": tf_5m_klines,
            "funding_rate": float(funding_rate),
            "open_interest": float(open_interest),
            "oi_change_pct": float(oi_change_pct),
            "top_trader_ls_ratio": float(top_trader_ls_ratio),
            "taker_buy_sell_ratio": float(taker_buy_sell_ratio),
            "coinalyze_available": bool(coinalyze_metrics.get("coinalyze_available", False)),
            "coinalyze_agg_oi_change_pct": float(
                coinalyze_metrics.get("coinalyze_agg_oi_change_pct", 0.0)
            ),
            "coinalyze_pred_funding_rate": coinalyze_pred_funding_rate,
            "coinalyze_liquidations_24h_usd": float(
                coinalyze_metrics.get("coinalyze_liquidations_24h_usd", 0.0)
            ),
            "coinalyze_liquidations_1h_usd": float(
                coinalyze_metrics.get("coinalyze_liquidations_1h_usd", 0.0)
            ),
            "coinalyze_short_liquidations_1h_usd": float(
                coinalyze_metrics.get("coinalyze_short_liquidations_1h_usd", 0.0)
            ),
            "coinalyze_global_long_short_ratio": float(
                coinalyze_metrics.get("coinalyze_global_long_short_ratio", 1.0)
            ),
            # Coinglass liquidation heatmap data.
            "coinglass_clusters": heatmap.get("clusters", []),
            "coinglass_high_density_zones": heatmap.get("high_density_zones", []),
            # Defaults — overridden by main.py after economic-event check.
            "caution_mode": False,
            "caution_reason": "",
            # Order book imbalance (top-20 Binance Futures depth).
            "ob_bid_pct": float(orderbook_metrics.get("ob_bid_pct", 50.0)),
            "ob_ask_pct": float(orderbook_metrics.get("ob_ask_pct", 50.0)),
            "ob_bid_usd": float(orderbook_metrics.get("ob_bid_usd", 0.0)),
            "ob_ask_usd": float(orderbook_metrics.get("ob_ask_usd", 0.0)),
        }

    async def fetch_fear_and_greed(self) -> int:
        timeout = aiohttp.ClientTimeout(total=10)
        url = "https://api.alternative.me/fng/?limit=1"
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        LOGGER.warning("Fear & Greed fetch failed (status=%s)", response.status)
                        return 50
                    data = await response.json()
        except Exception as exc:
            LOGGER.warning("Fear & Greed fetch error: %s", exc)
            return 50

        try:
            return int(data["data"][0]["value"])
        except Exception:
            return 50

    async def fetch_klines(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            return await self._fetch_klines(session, symbol, interval, limit)

    async def rank_symbols_by_24h_volume(
        self,
        symbols: List[str],
        limit: int,
    ) -> List[str]:
        if not symbols:
            return []

        symbol_set = set(symbols)
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            url = f"{FUTURES_BASE}/fapi/v1/ticker/24hr"
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

        scored = []
        for row in data:
            sym = row.get("symbol")
            if sym not in symbol_set:
                continue
            try:
                quote_volume = float(row.get("quoteVolume", 0.0))
            except (TypeError, ValueError):
                quote_volume = 0.0
            scored.append((sym, quote_volume))

        if not scored:
            return symbols[:limit] if limit > 0 else symbols

        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [sym for sym, _ in scored]

        # Preserve any missing configured symbols at the end in original order.
        missing = [sym for sym in symbols if sym not in set(ranked)]
        ordered = ranked + missing

        if limit > 0:
            return ordered[:limit]
        return ordered

    async def _binance_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Dict,
        max_retries: int = 3,
    ) -> Dict | List:
        """HTTP GET against Binance with automatic 429 / 418 backoff."""
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        # Respect Retry-After header when present.
                        retry_after = float(
                            response.headers.get("Retry-After", 2 ** (attempt + 1))
                        )
                        wait = min(retry_after, 60.0)
                        LOGGER.warning(
                            "Binance 429 rate-limit on %s — backing off %.1fs (attempt %s/%s)",
                            url, wait, attempt + 1, max_retries,
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(wait)
                            continue
                        return []
                    if response.status == 418:
                        LOGGER.error("Binance IP ban (418) on %s. Cooling down 60s.", url)
                        await asyncio.sleep(60)
                        return []
                    if response.status != 200:
                        LOGGER.warning("Binance non-200 (%s) on %s", response.status, url)
                        return []
                    return await response.json()
            except Exception as exc:
                wait = 2.0 ** attempt
                LOGGER.warning(
                    "Binance request error on %s (attempt %s/%s): %s — retry in %.1fs",
                    url, attempt + 1, max_retries, exc, wait,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    raise
        return []

    async def _fetch_klines(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        interval: str,
        limit: int,
    ) -> List[Dict]:
        url = f"{FUTURES_BASE}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        raw = await self._binance_request(session, url, params)
        if not raw:
            return []

        klines = []
        for k in raw:
            klines.append(
                {
                    "open_time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": int(k[6]),
                }
            )
        return klines

    async def _fetch_funding_rate(self, session: aiohttp.ClientSession, symbol: str) -> float:
        url = f"{FUTURES_BASE}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": 1}
        async with session.get(url, params=params) as response:
            if response.status != 200:
                LOGGER.warning("Funding rate fetch failed for %s (status=%s)", symbol, response.status)
                return await self._fetch_funding_fallback(session, symbol)
            data = await response.json()

        if not data:
            return await self._fetch_funding_fallback(session, symbol)

        try:
            return float(data[-1].get("fundingRate", 0.0))
        except (TypeError, ValueError):
            return await self._fetch_funding_fallback(session, symbol)

    async def _fetch_funding_fallback(self, session: aiohttp.ClientSession, symbol: str) -> float:
        # Fallback to premium index lastFundingRate when /fundingRate is empty/delayed.
        url = f"{FUTURES_BASE}/fapi/v1/premiumIndex"
        params = {"symbol": symbol}
        async with session.get(url, params=params) as response:
            if response.status != 200:
                LOGGER.warning("Funding fallback failed for %s (status=%s)", symbol, response.status)
                return 0.0
            data = await response.json()

        try:
            return float(data.get("lastFundingRate", 0.0))
        except (TypeError, ValueError):
            return 0.0

    async def _fetch_open_interest(self, session: aiohttp.ClientSession, symbol: str) -> float:
        url = f"{FUTURES_BASE}/fapi/v1/openInterest"
        params = {"symbol": symbol}
        async with session.get(url, params=params) as response:
            if response.status != 200:
                LOGGER.warning("Open interest fetch failed for %s (status=%s)", symbol, response.status)
                return 0.0
            data = await response.json()

        try:
            return float(data.get("openInterest", 0.0))
        except (TypeError, ValueError):
            return 0.0

    async def _fetch_open_interest_change(self, session: aiohttp.ClientSession, symbol: str) -> float:
        url = f"{FUTURES_BASE}/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": "5m", "limit": 2}
        data = await self._binance_request(session, url, params)
        if not data:
            return 0.0

        if len(data) < 2:
            return 0.0

        prev_oi = float(data[-2].get("sumOpenInterest", 0.0))
        latest_oi = float(data[-1].get("sumOpenInterest", 0.0))
        if prev_oi <= 0:
            return 0.0
        return ((latest_oi - prev_oi) / prev_oi) * 100.0

    async def _fetch_top_trader_ls_ratio(self, session: aiohttp.ClientSession, symbol: str) -> float:
        url = f"{FUTURES_BASE}/futures/data/topLongShortPositionRatio"
        params = {"symbol": symbol, "period": "5m", "limit": 1}
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    LOGGER.warning("Top trader LS ratio fetch failed for %s (status=%s)", symbol, response.status)
                    return 1.0
                data = await response.json()
            if not data:
                return 1.0
            return float(data[0].get("longShortRatio", 1.0))
        except Exception as exc:
            LOGGER.warning("Top trader LS ratio fetch error for %s: %s", symbol, exc)
            return 1.0

    async def _fetch_taker_buy_sell_ratio(self, session: aiohttp.ClientSession, symbol: str) -> float:
        url = f"{FUTURES_BASE}/futures/data/takerlongshortRatio"
        params = {"symbol": symbol, "period": "5m", "limit": 1}
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    LOGGER.warning("Taker buy/sell ratio fetch failed for %s (status=%s)", symbol, response.status)
                    return 1.0
                data = await response.json()
            if not data:
                return 1.0
            return float(data[0].get("buySellRatio", 1.0))
        except Exception as exc:
            LOGGER.warning("Taker buy/sell ratio fetch error for %s: %s", symbol, exc)
            return 1.0

    async def _fetch_orderbook_imbalance(
        self, session: aiohttp.ClientSession, symbol: str, depth: int = 20
    ) -> Dict[str, float]:
        """
        Fetches the top-`depth` order book levels from Binance Futures and
        returns bid/ask cumulative USD volume and the bid percentage of total liquidity.

        Returns:
            {
                "ob_bid_usd":  float,   # cumulative bid notional (price × qty)
                "ob_ask_usd":  float,   # cumulative ask notional
                "ob_bid_pct":  float,   # bid_usd / (bid_usd + ask_usd) × 100
                "ob_ask_pct":  float,   # 100 − ob_bid_pct
            }
        Falls back to neutral 50/50 on any error so downstream logic is never blocked.
        """
        neutral = {"ob_bid_usd": 0.0, "ob_ask_usd": 0.0, "ob_bid_pct": 50.0, "ob_ask_pct": 50.0}
        url = f"{FUTURES_BASE}/fapi/v1/depth"
        params = {"symbol": symbol, "limit": depth}
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    LOGGER.warning("Order book fetch failed for %s (status=%s)", symbol, response.status)
                    return neutral
                data = await response.json()
        except Exception as exc:
            LOGGER.warning("Order book fetch error for %s: %s", symbol, exc)
            return neutral

        try:
            bid_usd = sum(float(price) * float(qty) for price, qty in data.get("bids", []))
            ask_usd = sum(float(price) * float(qty) for price, qty in data.get("asks", []))
            total = bid_usd + ask_usd
            if total <= 0:
                return neutral
            bid_pct = (bid_usd / total) * 100.0
            ask_pct = 100.0 - bid_pct
            return {
                "ob_bid_usd": bid_usd,
                "ob_ask_usd": ask_usd,
                "ob_bid_pct": bid_pct,
                "ob_ask_pct": ask_pct,
            }
        except Exception as exc:
            LOGGER.warning("Order book calculation error for %s: %s", symbol, exc)
            return neutral
