"""
Coinglass Data Provider — pure async data module.

Provides liquidation-heatmap clusters and high-density zones for the main
trading bot.  No standalone main(), no Telegram, no duplicate indicators.
"""

import logging
import os
from typing import Dict, List

import aiohttp

LOGGER = logging.getLogger(__name__)

COINGLASS_BASE_URL = "https://open-api-v4.coinglass.com"


async def fetch_liquidation_heatmap(
    session: aiohttp.ClientSession,
    symbol: str,
    api_key: str,
) -> Dict:
    """
    Fetch liquidation heatmap for *one* symbol from Coinglass.

    Returns {"high_density_zones": [...], "clusters": [...]}.
    On failure returns empty lists so callers never crash.
    """
    url = f"{COINGLASS_BASE_URL}/public/v2/liquidation_heatmap"
    headers = {"coinglassSecret": api_key}
    params = {"symbol": symbol}

    try:
        async with session.get(url, headers=headers, params=params) as resp:
            if resp.status != 200:
                LOGGER.warning("Coinglass heatmap HTTP %s for %s", resp.status, symbol)
                return {"high_density_zones": [], "clusters": []}
            data = (await resp.json()).get("data", {})
            return {
                "high_density_zones": data.get("high_density_zones", []),
                "clusters": data.get("clusters", []),
            }
    except Exception as exc:
        LOGGER.warning("Coinglass heatmap error for %s: %s", symbol, exc)
        return {"high_density_zones": [], "clusters": []}


async def fetch_all_heatmaps(
    symbols: List[str],
    api_key: str | None = None,
    timeout_seconds: int = 10,
) -> Dict[str, Dict]:
    """
    Batch-fetch heatmaps for every symbol in *symbols*.

    Returns {symbol: {"high_density_zones": [...], "clusters": [...]}}.
    Safe to call even when no API key is set — returns empty dicts.
    """
    key = (api_key or os.getenv("COINGLASS_API_KEY", "")).strip()
    if not key:
        return {s: {"high_density_zones": [], "clusters": []} for s in symbols}

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    results: Dict[str, Dict] = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for symbol in symbols:
            results[symbol] = await fetch_liquidation_heatmap(session, symbol, key)

    return results

