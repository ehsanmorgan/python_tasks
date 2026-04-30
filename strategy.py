from dataclasses import dataclass
import html
import logging
from typing import Dict, List, Optional

import pandas as pd

from config import Settings, SCORING_WEIGHTS
from indicators import (
    compute_indicators,
    detect_trend,
    find_support_resistance,
    is_sideways_market,
    nearest_level,
    to_dataframe,
)
from risk_manager import build_trade_levels, calculate_position_size, calculate_kelly_fraction
from smart_money import liquidity_context


LOGGER = logging.getLogger(__name__)

# Re-export for backward compatibility (backtester.py, etc.)
WEIGHTS = SCORING_WEIGHTS

# ── Coinglass Liquidation Cluster Proximity ──────────────────────────────────

def _cluster_proximity_boost(clusters: list, price: float, side: str = "") -> tuple:
    """Return (buy_boost, sell_boost) based on directional cluster proximity.

    Cluster above price → shorts will be liquidated → bullish (+20 buy).
    Cluster below price → longs will be liquidated → bearish (+20 sell).
    """
    buy_boost = 0
    sell_boost = 0
    for cluster_price in clusters:
        if not isinstance(cluster_price, (int, float)) or cluster_price <= 0:
            continue
        dist = abs(price - cluster_price) / cluster_price
        if dist <= 0.015:
            if cluster_price > price:
                buy_boost = max(buy_boost, 20)   # Short liq above → bullish
            else:
                sell_boost = max(sell_boost, 20)  # Long liq below → bearish
    return buy_boost, sell_boost


@dataclass
class Signal:
    pair: str
    side: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    position_size: float
    binance_sentiment: str
    global_sentiment: str
    context_lines: List[str]
    global_context_lines: List[str]
    tier: str = "A"                 # "A" (>=85) or "B" (70-84)
    break_even_price: float = 0.0   # price at which SL moves to entry


def _is_near_level(price: float, level: Optional[float], atr: float, factor: float) -> bool:
    if level is None:
        return False
    return abs(price - level) <= (atr * factor)


def _has_nearby_block(
    side: str,
    price: float,
    support: Optional[float],
    resistance: Optional[float],
    atr: float,
    block_factor: float,
) -> bool:
    min_distance = atr * block_factor
    if side == "BUY" and resistance is not None:
        return (resistance - price) < min_distance
    if side == "SELL" and support is not None:
        return (price - support) < min_distance
    return False


def _has_rejection_confirmation(last: pd.Series, side: str, min_wick_ratio: float) -> bool:
    candle_range = max(float(last["high"] - last["low"]), 1e-9)
    open_price = float(last["open"])
    close_price = float(last["close"])

    lower_wick = min(open_price, close_price) - float(last["low"])
    upper_wick = float(last["high"]) - max(open_price, close_price)

    if side == "BUY":
        return close_price > open_price and (lower_wick / candle_range) >= min_wick_ratio
    return close_price < open_price and (upper_wick / candle_range) >= min_wick_ratio


def _log_market_insight(symbol: str, snapshot: Dict, extras: Dict) -> None:
    """
    Prints a structured 'Market Snapshot' block for every scanned coin —
    always visible before the Signal Rejected / Signal Generated message.
    """
    taker_bs      = float(snapshot.get("taker_buy_sell_ratio", 1.0))
    oi_change_pct = float(snapshot.get("oi_change_pct", 0.0))
    co_oi_chg     = float(snapshot.get("coinalyze_agg_oi_change_pct", 0.0))
    funding       = float(snapshot.get("funding_rate", 0.0))
    co_pred_fund  = float(snapshot.get("coinalyze_pred_funding_rate", funding))
    liq_1h        = float(snapshot.get("coinalyze_liquidations_1h_usd", 0.0))
    liq_24h       = float(snapshot.get("coinalyze_liquidations_24h_usd", 0.0))
    short_liq_1h  = float(snapshot.get("coinalyze_short_liquidations_1h_usd", 0.0))
    global_ls     = float(snapshot.get("coinalyze_global_long_short_ratio", 1.0))
    top_ls        = float(snapshot.get("top_trader_ls_ratio", 1.0))
    co_avail      = bool(snapshot.get("coinalyze_available", False))

    rsi       = extras.get("rsi", 0.0)
    adx       = extras.get("adx", 0.0)
    vol_accel = extras.get("vol_accel", 0.0)
    price     = extras.get("price", 0.0)
    trend     = extras.get("trend", "?")
    htf_trend = extras.get("htf_trend", "?")
    candle_mv = extras.get("candle_move_pct", 0.0)
    st_dir    = int(extras.get("supertrend_dir", 0))
    ob_bid_pct = float(extras.get("ob_bid_pct", 50.0))
    ob_ask_pct = float(extras.get("ob_ask_pct", 50.0))
    ob_bid_usd = float(extras.get("ob_bid_usd", 0.0))
    ob_ask_usd = float(extras.get("ob_ask_usd", 0.0))

    # ── Derived labels ─────────────────────────────────────────────────────────
    co_src       = "Coinalyze ✓" if co_avail else "Binance-only ⚠"
    taker_label  = "BUY  ▲" if taker_bs >= 1.0 else "SELL ▼"
    oi_arrow     = "↑" if co_oi_chg > 0 else ("↓" if co_oi_chg < 0 else "→")
    fund_label   = "LONG bias" if funding < 0 else ("SHORT bias" if funding > 0 else "neutral")
    trend_label  = f"{trend.upper()[:4]}/{htf_trend.upper()[:4]}"
    st_label     = "▲ BULL" if st_dir == 1 else ("▼ BEAR" if st_dir == -1 else "? N/A")
    # Order book imbalance label with wall warnings.
    if ob_ask_pct >= 60.0:
        ob_label = f"Bid {ob_bid_pct:.1f}% / Ask {ob_ask_pct:.1f}%  ⛔ SELL WALL"
    elif ob_bid_pct >= 55.0:
        ob_label = f"Bid {ob_bid_pct:.1f}% / Ask {ob_ask_pct:.1f}%  ⬆ BUY PRESSURE"
    else:
        ob_label = f"Bid {ob_bid_pct:.1f}% / Ask {ob_ask_pct:.1f}%  → BALANCED"

    # CVD-proxy: taker ratio vs OI divergence tells us who is driving price.
    if taker_bs > 1.02 and co_oi_chg > 0:
        cvd_status = "BULLISH CONFLUENCE ↑↑"
    elif taker_bs < 0.98 and co_oi_chg < 0:
        cvd_status = "BEARISH CONFLUENCE ↓↓"
    elif taker_bs > 1.02 and co_oi_chg < 0:
        cvd_status = "LONG SQUEEZE RISK  ↑↓"
    elif taker_bs < 0.98 and co_oi_chg > 0:
        cvd_status = "SHORT SQUEEZE RISK ↓↑"
    else:
        cvd_status = "NEUTRAL / MIXED    →→"

    # Squeeze alert when short liquidations spike.
    squeeze_flag = ""
    if short_liq_1h >= 5_000_000:
        squeeze_flag = f"  ⚡ SHORT SQUEEZE ALERT ${short_liq_1h:,.0f} liquidated"

    # Bollinger Band values for display (snapshot passes them via extras if available).
    bb_upper_val = extras.get("bb_upper", float("nan"))
    bb_lower_val = extras.get("bb_lower", float("nan"))
    bb_upper_str = f"{bb_upper_val:.2f}" if bb_upper_val == bb_upper_val else " n/a"
    bb_lower_str = f"{bb_lower_val:.2f}" if bb_lower_val == bb_lower_val else " n/a"

    sep = "─" * 72
    block = (
        f"\n{sep}\n"
        f"  Market Snapshot  [{symbol}]   Price: {price:.4f}   "
        f"Source: {co_src}\n"
        f"{sep}\n"
        f"  Trend  : {trend_label:<16}  RSI   : {rsi:>5.1f}      "
        f"ADX   : {adx:>5.1f}\n"
        f"  Candle : {candle_mv:>+6.2f}%           Vol   : {vol_accel:>5.2f}x     "
        f"Funding: {funding:>+.6f}  ({fund_label})\n"
        f"  SuperTrend (7×3) : {st_label}            "
        f"BB Upper: {bb_upper_str}  Lower: {bb_lower_str}\n"
        f"  Order Book (top-20): {ob_label}\n"
        f"    Bid Liq: ${ob_bid_usd:>12,.0f}   Ask Liq: ${ob_ask_usd:>12,.0f}\n"
        f"  Taker BS Ratio   : {taker_bs:.4f}  ({taker_label})\n"
        f"  Top Trader L/S   : {top_ls:.4f}   Global L/S : {global_ls:.4f}\n"
        f"  OI Δ Binance     : {oi_change_pct:>+.4f}%\n"
        f"  OI Δ Aggregated  : {co_oi_chg:>+.4f}%  {oi_arrow}   "
        f"Pred. Funding: {co_pred_fund:>+.6f}\n"
        f"  Liq 1h (Global)  : ${liq_1h:>12,.0f}   "
        f"Short Liq 1h: ${short_liq_1h:>10,.0f}\n"
        f"  Liq 24h (Global) : ${liq_24h:>12,.0f}\n"
        f"  CVD Status       : {cvd_status}{squeeze_flag}\n"
        f"{sep}"
    )
    LOGGER.info("Market Snapshot [%s] price=%.4f adx=%.1f rsi=%.1f vol=%.2fx "
                "taker=%.4f oi_delta=%+.4f%% liq1h=$%.0f short_liq1h=$%.0f cvd=%s",
                symbol, price, adx, rsi, vol_accel,
                taker_bs, co_oi_chg, liq_1h, short_liq_1h, cvd_status)
    print(block)


def _log_signal_rejection(symbol: str, reasons: List[str], snapshot: Dict) -> None:
    if not reasons:
        return
    message = f"Signal Rejected [{symbol}]: " + " | ".join(reasons)
    LOGGER.info(message)
    print(message)
    LOGGER.debug(
        "%s rejection diagnostics: taker_bs=%.3f oi_change=%.3f co_oi_change=%.3f co_short_liq_1h=%.0f",
        symbol,
        float(snapshot.get("taker_buy_sell_ratio", 1.0)),
        float(snapshot.get("oi_change_pct", 0.0)),
        float(snapshot.get("coinalyze_agg_oi_change_pct", 0.0)),
        float(snapshot.get("coinalyze_short_liquidations_1h_usd", 0.0)),
    )


def _bollinger_breakout_flags(df: pd.DataFrame, sigma: float = 2.0) -> Dict[str, float]:
    close = df["close"]
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    bb_upper = bb_mid + (sigma * bb_std)
    bb_lower = bb_mid - (sigma * bb_std)

    if len(df) < 21 or bb_upper.iloc[-1] != bb_upper.iloc[-1] or bb_lower.iloc[-1] != bb_lower.iloc[-1]:
        return {
            "upper": float("nan"),
            "lower": float("nan"),
            "breakout_up": False,
            "breakout_down": False,
        }

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    upper_now = float(bb_upper.iloc[-1])
    lower_now = float(bb_lower.iloc[-1])
    upper_prev = float(bb_upper.iloc[-2])
    lower_prev = float(bb_lower.iloc[-2])

    breakout_up = last_close > upper_now and prev_close <= upper_prev
    breakout_down = last_close < lower_now and prev_close >= lower_prev
    return {
        "upper": upper_now,
        "lower": lower_now,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: compute_scores — data extraction, indicator computation, scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scores(
    snapshot: Dict,
    settings: Settings,
    btc_snapshot: Optional[Dict] = None,
) -> Optional[Dict]:
    """Extract data, compute indicators, and calculate raw buy/sell scores.

    Returns a context dict with all computed values, or None if rejected early.
    """
    symbol = snapshot["symbol"]
    df = to_dataframe(snapshot["klines"])
    if len(df) < 210:
        _log_signal_rejection(symbol, ["Insufficient 15m candles (<210)"], snapshot)
        return None

    higher_tf_klines = snapshot.get("higher_tf_klines", [])
    if not higher_tf_klines:
        _log_signal_rejection(symbol, ["Missing higher timeframe klines"], snapshot)
        return None

    htf_df = to_dataframe(higher_tf_klines)
    if len(htf_df) < 210:
        _log_signal_rejection(symbol, ["Insufficient higher timeframe candles (<210)"], snapshot)
        return None

    df = compute_indicators(df)
    htf_df = compute_indicators(htf_df)
    if df[["ema50", "ema200", "rsi14", "atr14", "adx14", "vol_ma20"]].tail(1).isnull().any().any():
        _log_signal_rejection(symbol, ["Primary indicator values unavailable"], snapshot)
        return None
    if htf_df[["ema50", "ema200"]].tail(1).isnull().any().any():
        _log_signal_rejection(symbol, ["Higher timeframe EMA values unavailable"], snapshot)
        return None

    liq = liquidity_context(df)
    supports, resistances = find_support_resistance(df)

    # ── Quad-TF Analysis: 4H (macro), 1H, 15M (alignment), 5M (surgical entry) ──
    tf_4h_klines = snapshot.get("tf_4h_klines", [])
    tf_1h_klines = snapshot.get("tf_1h_klines", [])
    tf_5m_klines = snapshot.get("tf_5m_klines", [])

    tf_4h_trend = "sideways"
    tf_1h_trend = "sideways"
    tf_5m_rsi = 50.0
    tf_5m_ema9_pullback = False

    if tf_4h_klines:
        tf_4h_df = to_dataframe(tf_4h_klines)
        if len(tf_4h_df) >= 210:
            tf_4h_df = compute_indicators(tf_4h_df)
            if not tf_4h_df[["ema50", "ema200"]].tail(1).isnull().any().any():
                tf_4h_trend = detect_trend(tf_4h_df.iloc[-1])

    if tf_1h_klines:
        tf_1h_df = to_dataframe(tf_1h_klines)
        if len(tf_1h_df) >= 210:
            tf_1h_df = compute_indicators(tf_1h_df)
            if not tf_1h_df[["ema50", "ema200"]].tail(1).isnull().any().any():
                tf_1h_trend = detect_trend(tf_1h_df.iloc[-1])

    tf_5m_swing_low = 0.0
    tf_5m_swing_high = 0.0
    if tf_5m_klines:
        tf_5m_df = to_dataframe(tf_5m_klines)
        if len(tf_5m_df) >= 20:
            ema9_series = tf_5m_df["close"].ewm(span=9, adjust=False).mean()
            rsi_series = tf_5m_df["close"].diff()
            gain = rsi_series.clip(lower=0).rolling(14).mean()
            loss = (-rsi_series.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi_5m_series = 100 - (100 / (1 + rs))
            if not rsi_5m_series.empty and rsi_5m_series.iloc[-1] == rsi_5m_series.iloc[-1]:
                tf_5m_rsi = float(rsi_5m_series.iloc[-1])
            if not ema9_series.empty:
                ema9_val = float(ema9_series.iloc[-1])
                close_5m = float(tf_5m_df.iloc[-1]["close"])
                tf_5m_ema9_pullback = abs(close_5m - ema9_val) / max(ema9_val, 1e-9) <= 0.0015
            tf_5m_swing_low = float(tf_5m_df["low"].tail(10).min()) if len(tf_5m_df) >= 10 else 0.0
            tf_5m_swing_high = float(tf_5m_df["high"].tail(10).max()) if len(tf_5m_df) >= 10 else 0.0

    quad_4h_long_boost = 15 if tf_4h_trend == "uptrend" else 0
    quad_4h_short_boost = 15 if tf_4h_trend == "downtrend" else 0

    # BTC Sentinel: if BTC 1H trend opposes altcoin trade direction, penalize
    btc_1h_trend = "sideways"
    if btc_snapshot and symbol != "BTCUSDT":
        btc_1h_klines = btc_snapshot.get("tf_1h_klines", [])
        if btc_1h_klines:
            btc_1h_df = to_dataframe(btc_1h_klines)
            if len(btc_1h_df) >= 210:
                btc_1h_df = compute_indicators(btc_1h_df)
                if not btc_1h_df[["ema50", "ema200"]].tail(1).isnull().any().any():
                    btc_1h_trend = detect_trend(btc_1h_df.iloc[-1])

    # ── Market variables ─────────────────────────────────────────────────────
    last = df.iloc[-1]
    htf_last = htf_df.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr14"])
    adx = float(last["adx14"])
    trend = detect_trend(last)
    htf_trend = detect_trend(htf_last)

    triple_align_long = tf_4h_trend == "uptrend" and tf_1h_trend == "uptrend" and trend == "uptrend"
    triple_align_short = tf_4h_trend == "downtrend" and tf_1h_trend == "downtrend" and trend == "downtrend"
    triple_alignment_boost = 20 if (triple_align_long or triple_align_short) else 0

    is_uptrend = trend == "uptrend" and htf_trend == "uptrend"
    is_downtrend = trend == "downtrend" and htf_trend == "downtrend"

    ema_gap_ratio = abs(float(last["ema50"]) - float(last["ema200"])) / max(price, 1e-9)
    htf_ema_gap_ratio = abs(float(htf_last["ema50"]) - float(htf_last["ema200"])) / max(
        float(htf_last["close"]), 1e-9
    )

    levels = nearest_level(price, supports, resistances)
    support = levels["support"]
    resistance = levels["resistance"]

    rsi = float(last["rsi14"])
    vol = float(last["volume"])
    vol_ma = float(last["vol_ma20"])
    volume_spike = vol_ma > 0 and (vol / vol_ma) >= settings.volume_spike_factor

    funding = float(snapshot["funding_rate"])
    open_interest = float(snapshot.get("open_interest", 0.0))
    oi_change_pct = float(snapshot["oi_change_pct"])
    top_ls = float(snapshot.get("top_trader_ls_ratio", 1.0))
    taker_bs = float(snapshot.get("taker_buy_sell_ratio", 1.0))
    coinalyze_available = bool(snapshot.get("coinalyze_available", False))
    co_oi_change_pct = float(snapshot.get("coinalyze_agg_oi_change_pct", 0.0))
    co_pred_funding = float(snapshot.get("coinalyze_pred_funding_rate", funding))
    co_global_ls = float(snapshot.get("coinalyze_global_long_short_ratio", 1.0))
    co_liq_24h_usd = float(snapshot.get("coinalyze_liquidations_24h_usd", 0.0))
    co_liq_1h_usd = float(snapshot.get("coinalyze_liquidations_1h_usd", 0.0))
    co_short_liq_1h_usd = float(snapshot.get("coinalyze_short_liquidations_1h_usd", 0.0))
    oi_increasing = oi_change_pct >= settings.min_oi_change_pct

    ob_bid_pct = float(snapshot.get("ob_bid_pct", 50.0))
    ob_ask_pct = float(snapshot.get("ob_ask_pct", 50.0))
    ob_bid_usd = float(snapshot.get("ob_bid_usd", 0.0))
    ob_ask_usd = float(snapshot.get("ob_ask_usd", 0.0))
    ob_buy_wall  = ob_bid_pct >= 55.0
    ob_sell_wall = ob_ask_pct >= 60.0

    bb = _bollinger_breakout_flags(df)
    bb_breakout_up = bool(bb["breakout_up"])
    bb_breakout_down = bool(bb["breakout_down"])

    supertrend_dir = int(last.get("supertrend_dir", 0))
    supertrend_bull = supertrend_dir == 1
    supertrend_bear = supertrend_dir == -1

    candle_move_pct = abs((float(last["close"]) - float(last["open"])) / max(float(last["open"]), 1e-9)) * 100
    volume_accel = (vol / vol_ma) if vol_ma > 0 else 0.0
    fast_track = (
        candle_move_pct >= settings.fast_track_candle_move_pct
        and volume_accel >= settings.fast_track_volume_mult
    )
    price_rising = float(last["close"]) > float(last["open"])
    price_falling = float(last["close"]) < float(last["open"])

    _log_market_insight(symbol, snapshot, {
        "rsi": rsi, "adx": adx, "vol_accel": volume_accel,
        "price": price, "trend": trend, "htf_trend": htf_trend,
        "candle_move_pct": candle_move_pct, "supertrend_dir": supertrend_dir,
        "bb_upper": bb["upper"], "bb_lower": bb["lower"],
        "ob_bid_pct": ob_bid_pct, "ob_ask_pct": ob_ask_pct,
        "ob_bid_usd": ob_bid_usd, "ob_ask_usd": ob_ask_usd,
    })

    liq_volume_override = (
        co_short_liq_1h_usd >= settings.coinalyze_short_liq_spike_1h_usd
        or co_liq_1h_usd >= settings.coinalyze_liquidation_spike_1h_usd
    )
    effective_volume_spike = volume_spike or liq_volume_override

    coinalyze_liq_long_override = (
        price_rising
        and volume_spike
        and co_short_liq_1h_usd >= settings.coinalyze_short_liq_spike_1h_usd
    )

    volatility_breakout_long = (
        bb_breakout_up
        and volume_spike
        and co_liq_1h_usd >= settings.coinalyze_liquidation_spike_1h_usd
    )
    volatility_breakout_short = (
        bb_breakout_down
        and volume_spike
        and co_liq_1h_usd >= settings.coinalyze_liquidation_spike_1h_usd
    )

    # ── Early gates ──────────────────────────────────────────────────────────
    if settings.require_non_zero_derivatives and (open_interest == 0.0 or funding == 0.0):
        _log_signal_rejection(symbol, ["Missing derivatives data (OI or funding = 0)"], snapshot)
        return None

    if is_sideways_market(last, price):
        _log_signal_rejection(symbol, ["Market sideways regime"], snapshot)
        return None

    if adx < settings.min_adx and not (fast_track or volatility_breakout_long or volatility_breakout_short):
        _log_signal_rejection(symbol, [f"ADX too low ({adx:.2f} < {settings.min_adx:.2f})"], snapshot)
        return None

    if (
        (ema_gap_ratio < settings.min_ema_gap_ratio or htf_ema_gap_ratio < settings.min_htf_ema_gap_ratio)
        and not (fast_track or volatility_breakout_long or volatility_breakout_short)
    ):
        _log_signal_rejection(
            symbol,
            [f"EMA structure too weak (15m={ema_gap_ratio:.4f}, HTF={htf_ema_gap_ratio:.4f})"],
            snapshot,
        )
        return None

    candle_range = float(last["high"] - last["low"])
    if atr > 0 and candle_range >= (settings.strong_spike_atr_mult * atr) and not (
        volatility_breakout_long or volatility_breakout_short
    ):
        _log_signal_rejection(
            symbol,
            [f"Abnormal candle range ({candle_range:.4f}) exceeds {settings.strong_spike_atr_mult:.2f}x ATR"],
            snapshot,
        )
        return None

    # ── Condition flags ──────────────────────────────────────────────────────
    buy_key_level = _is_near_level(price, support, atr, settings.level_proximity_atr_factor)
    sell_key_level = _is_near_level(price, resistance, atr, settings.level_proximity_atr_factor)

    if is_uptrend and (fast_track or volatility_breakout_long or coinalyze_liq_long_override):
        buy_rsi_ok = rsi <= 78
        sell_rsi_ok = rsi >= 22
    elif fast_track:
        buy_rsi_ok = rsi <= 75
        sell_rsi_ok = rsi >= 25
    else:
        buy_rsi_ok = rsi <= 68
        sell_rsi_ok = rsi >= 32

    buy_funding_ok = funding < 0
    sell_funding_ok = funding > 0

    buy_rejection = _has_rejection_confirmation(last, "BUY", settings.min_rejection_wick_ratio)
    sell_rejection = _has_rejection_confirmation(last, "SELL", settings.min_rejection_wick_ratio)

    ema50_val = float(last["ema50"])
    ema50_dist_pct = abs(price - ema50_val) / max(ema50_val, 1e-9) * 100

    momentum_override = taker_bs > 1.05 and volume_accel > 1.5
    if momentum_override:
        buy_rsi_ok = True
        sell_rsi_ok = True

    buy_blocked = _has_nearby_block(
        "BUY", price, support, resistance, atr, settings.nearby_block_atr_factor
    )
    sell_blocked = _has_nearby_block(
        "SELL", price, support, resistance, atr, settings.nearby_block_atr_factor
    )

    buy_taker_threshold = min(settings.min_taker_buy_sell_long, 1.02)
    sell_taker_threshold = max(settings.max_taker_buy_sell_short, 0.98)
    buy_taker_positive = taker_bs >= buy_taker_threshold
    sell_taker_positive = taker_bs <= sell_taker_threshold

    if coinalyze_liq_long_override:
        buy_taker_positive = True

    strong_long_confirmed = taker_bs > 1.05 and co_oi_change_pct > 0

    single_source_mode = not coinalyze_available
    oi_divergence = (oi_change_pct > 0 and co_oi_change_pct < 0) or (
        oi_change_pct < 0 and co_oi_change_pct > 0
    )

    co_bullish = (
        co_oi_change_pct >= settings.min_coinalyze_agg_oi_long
        and co_global_ls >= settings.min_coinalyze_ls_long
        and co_pred_funding <= settings.max_predicted_funding_for_long
    )
    co_bearish = (
        co_oi_change_pct <= settings.max_coinalyze_agg_oi_short
        and co_global_ls <= settings.max_coinalyze_ls_short
        and co_pred_funding >= settings.min_predicted_funding_for_short
    )

    # ── Build condition dictionaries ─────────────────────────────────────────
    buy_major = {
        "trend": is_uptrend or volatility_breakout_long,
        "aggregated_sentiment": co_bullish,
        "key_level": buy_key_level or volatility_breakout_long,
        "rsi": buy_rsi_ok,
        "volume": effective_volume_spike or (buy_taker_positive and taker_bs > 1.0 and volume_accel >= 0.5),
        "oi": oi_increasing or co_oi_change_pct > 0,
        "funding": buy_funding_ok,
        "top_trader": top_ls >= settings.min_top_trader_ls_long,
        "taker_flow": buy_taker_positive,
        "liquidations": coinalyze_liq_long_override,
        "volatility_breakout": volatility_breakout_long,
        "supertrend": supertrend_bull or volatility_breakout_long or coinalyze_liq_long_override
            or (buy_taker_positive and taker_bs > 1.0 and (oi_increasing or co_oi_change_pct > 0)),
    }

    sell_major = {
        "trend": is_downtrend or volatility_breakout_short,
        "aggregated_sentiment": co_bearish,
        "key_level": sell_key_level or volatility_breakout_short,
        "rsi": sell_rsi_ok,
        "volume": effective_volume_spike or (sell_taker_positive and taker_bs < 1.0 and volume_accel >= 0.5),
        "oi": oi_increasing or co_oi_change_pct < 0,
        "funding": sell_funding_ok,
        "top_trader": top_ls <= settings.max_top_trader_ls_short,
        "taker_flow": sell_taker_positive,
        "liquidations": co_liq_1h_usd >= settings.coinalyze_liquidation_spike_1h_usd,
        "volatility_breakout": volatility_breakout_short,
        "supertrend": supertrend_bear or volatility_breakout_short
            or (sell_taker_positive and taker_bs < 1.0 and (oi_increasing or co_oi_change_pct < 0)),
    }

    buy_conf = sum(WEIGHTS[k] for k, ok in buy_major.items() if ok)
    sell_conf = sum(WEIGHTS[k] for k, ok in sell_major.items() if ok)

    # ── Pack context ─────────────────────────────────────────────────────────
    return {
        "symbol": symbol, "df": df, "htf_df": htf_df, "last": last, "htf_last": htf_last,
        "price": price, "atr": atr, "adx": adx, "trend": trend, "htf_trend": htf_trend,
        "liq": liq, "supports": supports, "resistances": resistances,
        "support": support, "resistance": resistance,
        "tf_4h_trend": tf_4h_trend, "tf_1h_trend": tf_1h_trend,
        "tf_5m_rsi": tf_5m_rsi, "tf_5m_ema9_pullback": tf_5m_ema9_pullback,
        "tf_5m_swing_low": tf_5m_swing_low, "tf_5m_swing_high": tf_5m_swing_high,
        "btc_1h_trend": btc_1h_trend, "btc_snapshot": btc_snapshot,
        "quad_4h_long_boost": quad_4h_long_boost, "quad_4h_short_boost": quad_4h_short_boost,
        "triple_alignment_boost": triple_alignment_boost,
        "triple_align_long": triple_align_long, "triple_align_short": triple_align_short,
        "is_uptrend": is_uptrend, "is_downtrend": is_downtrend,
        "rsi": rsi, "vol": vol, "vol_ma": vol_ma, "volume_spike": volume_spike,
        "funding": funding, "open_interest": open_interest, "oi_change_pct": oi_change_pct,
        "top_ls": top_ls, "taker_bs": taker_bs,
        "coinalyze_available": coinalyze_available, "co_oi_change_pct": co_oi_change_pct,
        "co_pred_funding": co_pred_funding, "co_global_ls": co_global_ls,
        "co_liq_24h_usd": co_liq_24h_usd, "co_liq_1h_usd": co_liq_1h_usd,
        "co_short_liq_1h_usd": co_short_liq_1h_usd, "oi_increasing": oi_increasing,
        "ob_bid_pct": ob_bid_pct, "ob_ask_pct": ob_ask_pct,
        "ob_bid_usd": ob_bid_usd, "ob_ask_usd": ob_ask_usd,
        "ob_buy_wall": ob_buy_wall, "ob_sell_wall": ob_sell_wall,
        "bb": bb, "bb_breakout_up": bb_breakout_up, "bb_breakout_down": bb_breakout_down,
        "supertrend_dir": supertrend_dir, "supertrend_bull": supertrend_bull, "supertrend_bear": supertrend_bear,
        "candle_move_pct": candle_move_pct, "volume_accel": volume_accel,
        "fast_track": fast_track, "price_rising": price_rising, "price_falling": price_falling,
        "effective_volume_spike": effective_volume_spike,
        "coinalyze_liq_long_override": coinalyze_liq_long_override,
        "volatility_breakout_long": volatility_breakout_long,
        "volatility_breakout_short": volatility_breakout_short,
        "buy_key_level": buy_key_level, "sell_key_level": sell_key_level,
        "buy_rsi_ok": buy_rsi_ok, "sell_rsi_ok": sell_rsi_ok,
        "buy_funding_ok": buy_funding_ok, "sell_funding_ok": sell_funding_ok,
        "buy_rejection": buy_rejection, "sell_rejection": sell_rejection,
        "ema50_val": ema50_val, "ema50_dist_pct": ema50_dist_pct,
        "momentum_override": momentum_override,
        "buy_blocked": buy_blocked, "sell_blocked": sell_blocked,
        "buy_taker_threshold": buy_taker_threshold, "sell_taker_threshold": sell_taker_threshold,
        "buy_taker_positive": buy_taker_positive, "sell_taker_positive": sell_taker_positive,
        "strong_long_confirmed": strong_long_confirmed,
        "single_source_mode": single_source_mode, "oi_divergence": oi_divergence,
        "co_bullish": co_bullish, "co_bearish": co_bearish,
        "buy_major": buy_major, "sell_major": sell_major,
        "buy_conf": buy_conf, "sell_conf": sell_conf,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: apply_gates — validity checks, penalties, and boosts
# ═══════════════════════════════════════════════════════════════════════════════

def apply_gates(ctx: Dict, snapshot: Dict, settings: Settings) -> None:
    """Apply validity gates, penalties, and boosts. Modifies ctx in-place."""
    buy_major = ctx["buy_major"]
    sell_major = ctx["sell_major"]
    buy_conf = ctx["buy_conf"]
    sell_conf = ctx["sell_conf"]

    volatility_breakout_long = ctx["volatility_breakout_long"]
    volatility_breakout_short = ctx["volatility_breakout_short"]
    coinalyze_liq_long_override = ctx["coinalyze_liq_long_override"]
    momentum_override = ctx["momentum_override"]
    fast_track = ctx["fast_track"]

    buy_min_conditions = 3 if (volatility_breakout_long or coinalyze_liq_long_override) else 4
    sell_min_conditions = 3 if volatility_breakout_short else 4
    buy_valid = sum(buy_major.values()) >= buy_min_conditions and not ctx["buy_blocked"]
    sell_valid = sum(sell_major.values()) >= sell_min_conditions and not ctx["sell_blocked"]

    buy_conditions_met = sum(buy_major.values())
    sell_conditions_met = sum(sell_major.values())
    if not (volatility_breakout_long or coinalyze_liq_long_override or momentum_override):
        if buy_conditions_met < 6:
            buy_valid = buy_valid and ctx["buy_rejection"]
    if not (volatility_breakout_short or momentum_override):
        if sell_conditions_met < 6:
            sell_valid = sell_valid and ctx["sell_rejection"]

    # Fast-track momentum override
    if fast_track and ctx["price_rising"] and ctx["effective_volume_spike"] and ctx["buy_taker_positive"]:
        buy_valid = True
    if fast_track and ctx["price_falling"] and ctx["effective_volume_spike"] and ctx["sell_taker_positive"]:
        sell_valid = True
    if volatility_breakout_long:
        buy_valid = True
    if volatility_breakout_short:
        sell_valid = True

    # Trap adjustments
    if ctx["liq"]["bull_trap"]:
        buy_conf -= 10
        buy_valid = False
    if ctx["liq"]["bear_trap"]:
        sell_conf -= 10
        sell_valid = False
        buy_conf += 15
    if ctx["liq"]["bull_trap"] and sell_valid:
        sell_conf += 5

    # EMA50 extension penalty
    if ctx["ema50_dist_pct"] > 1.5 and not momentum_override:
        if ctx["price"] > ctx["ema50_val"]:
            buy_conf -= 10
        else:
            sell_conf -= 10

    # Quad-TF boosts
    buy_conf += ctx["quad_4h_long_boost"]
    sell_conf += ctx["quad_4h_short_boost"]
    if ctx["triple_align_long"]:
        buy_conf += ctx["triple_alignment_boost"]
    if ctx["triple_align_short"]:
        sell_conf += ctx["triple_alignment_boost"]

    # BTC Sentinel: scale penalty by BTC trend conviction
    btc_sentinel_penalty = 0
    symbol = ctx["symbol"]
    btc_1h_trend = ctx["btc_1h_trend"]
    btc_snapshot = ctx["btc_snapshot"]
    if symbol != "BTCUSDT" and btc_1h_trend != "sideways" and btc_snapshot:
        btc_1h_klines_for_adx = btc_snapshot.get("tf_1h_klines", [])
        btc_adx = 20.0
        if btc_1h_klines_for_adx:
            try:
                _btc_df = to_dataframe(btc_1h_klines_for_adx)
                if len(_btc_df) >= 30:
                    _btc_df = compute_indicators(_btc_df)
                    btc_adx = float(_btc_df.iloc[-1]["adx14"])
            except Exception:
                pass
        if btc_adx >= 25:
            btc_penalty = 25
        elif btc_adx >= 18:
            btc_penalty = 18
        else:
            btc_penalty = 10
        if btc_1h_trend == "downtrend":
            buy_conf -= btc_penalty
            btc_sentinel_penalty = -btc_penalty
        elif btc_1h_trend == "uptrend":
            sell_conf -= btc_penalty
            btc_sentinel_penalty = -btc_penalty

    # 5M surgical entry gate
    if ctx["tf_5m_rsi"] > 70 and not momentum_override:
        buy_valid = False
    if ctx["tf_5m_rsi"] < 30 and not momentum_override:
        sell_valid = False

    # Coinglass cluster proximity: directional liquidation boost
    coinglass_clusters = snapshot.get("coinglass_clusters", [])
    heatmap_buy_boost, heatmap_sell_boost = _cluster_proximity_boost(coinglass_clusters, ctx["price"])
    buy_conf += heatmap_buy_boost
    sell_conf += heatmap_sell_boost

    # OI divergence penalties
    oi_divergence = ctx["oi_divergence"]
    if oi_divergence:
        buy_conf -= settings.coinalyze_oi_divergence_penalty
        sell_conf -= settings.coinalyze_oi_divergence_penalty

    if buy_valid and not oi_divergence and ctx["oi_increasing"] and ctx["co_oi_change_pct"] < 0 and not coinalyze_liq_long_override:
        buy_conf -= settings.coinalyze_oi_divergence_penalty

    if buy_valid and ctx["coinalyze_available"] and not ctx["strong_long_confirmed"] and not (volatility_breakout_long or coinalyze_liq_long_override):
        buy_conf -= settings.coinalyze_oi_divergence_penalty

    # Breakout confluence
    resistance = ctx["resistance"]
    support = ctx["support"]
    price = ctx["price"]
    breakout_up = resistance is not None and price > resistance
    breakout_down = support is not None and price < support
    if ctx["co_oi_change_pct"] > 0 and breakout_up:
        buy_conf += settings.breakout_confluence_boost
    if ctx["co_oi_change_pct"] < 0 and breakout_down:
        sell_conf += settings.breakout_confluence_boost

    # Short squeeze detector
    if ctx["price_rising"] and ctx["co_short_liq_1h_usd"] >= settings.coinalyze_short_liq_spike_1h_usd:
        buy_conf += 25

    # Order book imbalance adjustments
    if ctx["ob_buy_wall"]:
        buy_conf += 8
    if ctx["ob_sell_wall"] and not (volatility_breakout_long or coinalyze_liq_long_override):
        buy_valid = False

    # Strong setups must include both volume and taker pressure
    if buy_conf >= 85 and not (ctx["effective_volume_spike"] and ctx["buy_taker_positive"]) and not volatility_breakout_long:
        buy_valid = False
    if sell_conf >= 85 and not (ctx["effective_volume_spike"] and ctx["sell_taker_positive"]) and not volatility_breakout_short:
        sell_valid = False

    # High-confidence promotion guard
    if ctx["coinalyze_available"] and buy_conf >= 85 and not ctx["co_bullish"] and not (
        volatility_breakout_long or coinalyze_liq_long_override
    ):
        buy_conf -= 8
    if ctx["coinalyze_available"] and sell_conf >= 85 and not ctx["co_bearish"] and not volatility_breakout_short:
        sell_conf -= 8

    # Store results back into context
    ctx["buy_conf"] = buy_conf
    ctx["sell_conf"] = sell_conf
    ctx["buy_valid"] = buy_valid
    ctx["sell_valid"] = sell_valid
    ctx["buy_conditions_met"] = buy_conditions_met
    ctx["sell_conditions_met"] = sell_conditions_met
    ctx["btc_sentinel_penalty"] = btc_sentinel_penalty
    ctx["heatmap_buy_boost"] = heatmap_buy_boost
    ctx["heatmap_sell_boost"] = heatmap_sell_boost


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: generate_signal — candidate selection, logging, Signal creation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signal(
    ctx: Dict,
    snapshot: Dict,
    settings: Settings,
    _debug_msgs: Optional[List[str]] = None,
) -> Optional[Signal]:
    """Select best candidate and build the Signal object."""
    symbol = ctx["symbol"]
    buy_conf = ctx["buy_conf"]
    sell_conf = ctx["sell_conf"]
    buy_valid = ctx["buy_valid"]
    sell_valid = ctx["sell_valid"]
    min_conf_required = settings.min_confidence

    buy_direction_ok = ctx["is_uptrend"] or ctx["volatility_breakout_long"] or ctx["coinalyze_liq_long_override"]
    sell_direction_ok = ctx["is_downtrend"] or ctx["volatility_breakout_short"]

    # Verbose final score
    LOGGER.info(
        "[%s] Final Score — BUY: %.0f / %.0f (valid=%s, dir_ok=%s) | "
        "SELL: %.0f / %.0f (valid=%s, dir_ok=%s)",
        symbol, buy_conf, min_conf_required, buy_valid, buy_direction_ok,
        sell_conf, min_conf_required, sell_valid, sell_direction_ok,
    )
    print(
        f"[{symbol}]  Final Score "
        f"BUY: {buy_conf:.0f} / {min_conf_required:.0f}  "
        f"(valid={buy_valid}, dir_ok={buy_direction_ok})  |  "
        f"SELL: {sell_conf:.0f} / {min_conf_required:.0f}  "
        f"(valid={sell_valid}, dir_ok={sell_direction_ok})"
    )

    candidate = None
    if buy_valid and buy_direction_ok and buy_conf >= min_conf_required:
        candidate = ("BUY", float(buy_conf), ctx["support"], ctx["resistance"])
    if sell_valid and sell_direction_ok and sell_conf >= min_conf_required:
        sell_candidate = ("SELL", float(sell_conf), ctx["support"], ctx["resistance"])
        if candidate is None or sell_candidate[1] > candidate[1]:
            candidate = sell_candidate

    if candidate is None:
        _log_rejection_details(ctx, snapshot, settings, _debug_msgs, min_conf_required,
                               buy_direction_ok, sell_direction_ok)
        return None

    side, confidence, sup, res = candidate
    return _build_signal(ctx, snapshot, settings, side, confidence, sup, res)


def _log_rejection_details(
    ctx: Dict,
    snapshot: Dict,
    settings: Settings,
    _debug_msgs: Optional[List[str]],
    min_conf_required: float,
    buy_direction_ok: bool,
    sell_direction_ok: bool,
) -> None:
    """Log detailed rejection reasons for both sides."""
    symbol = ctx["symbol"]
    buy_conf = ctx["buy_conf"]
    sell_conf = ctx["sell_conf"]
    buy_valid = ctx["buy_valid"]
    sell_valid = ctx["sell_valid"]
    buy_major = ctx["buy_major"]
    sell_major = ctx["sell_major"]
    rsi = ctx["rsi"]
    volume_accel = ctx["volume_accel"]
    taker_bs = ctx["taker_bs"]
    momentum_override = ctx["momentum_override"]
    btc_sentinel_penalty = ctx["btc_sentinel_penalty"]
    btc_1h_trend = ctx["btc_1h_trend"]
    supertrend_dir = ctx["supertrend_dir"]
    trend = ctx["trend"]

    buy_reasons: List[str] = []
    sell_reasons: List[str] = []

    if not buy_direction_ok:
        buy_reasons.append("trend mismatch")
    if not ctx["buy_rsi_ok"]:
        buy_reasons.append(f"RSI too high for long ({rsi:.1f} > 78 in momentum regime)")
    if not ctx["effective_volume_spike"]:
        buy_reasons.append(f"low volume ({volume_accel:.2f}x < {settings.volume_spike_factor:.2f}x)")
    if not ctx["buy_taker_positive"]:
        buy_reasons.append(f"low taker flow ({taker_bs:.3f} < {ctx['buy_taker_threshold']:.2f})")
    if ctx["buy_blocked"]:
        buy_reasons.append("nearby resistance block")
    if ctx["ema50_dist_pct"] > 1.5 and not momentum_override and ctx["price"] > ctx["ema50_val"]:
        buy_reasons.append(f"EMA50 extension penalty -10 (dist {ctx['ema50_dist_pct']:.1f}%)")
    if ctx["tf_5m_rsi"] > 70 and not momentum_override:
        buy_reasons.append(f"5M RSI overbought ({ctx['tf_5m_rsi']:.1f} > 70)")
    if btc_sentinel_penalty < 0 and btc_1h_trend == "downtrend":
        buy_reasons.append(f"BTC Sentinel: BTC 1H bearish ({btc_sentinel_penalty})")
    if not ctx["buy_rejection"] and not (ctx["volatility_breakout_long"] or ctx["coinalyze_liq_long_override"] or momentum_override) and ctx["buy_conditions_met"] < 6:
        buy_reasons.append("missing bullish rejection candle (marginal setup)")
    if ctx["ob_sell_wall"] and not (ctx["volatility_breakout_long"] or ctx["coinalyze_liq_long_override"]):
        buy_reasons.append(
            f"OB sell wall ({ctx['ob_ask_pct']:.1f}% ask ≥ 60%) — long blocked by order book safety filter"
        )
    if not ctx["supertrend_bull"] and not (ctx["volatility_breakout_long"] or ctx["coinalyze_liq_long_override"])\
            and not (ctx["buy_taker_positive"] and taker_bs > 1.0 and (ctx["oi_increasing"] or ctx["co_oi_change_pct"] > 0)):
        buy_reasons.append(f"SuperTrend bearish/neutral (dir={supertrend_dir:+d})")
    if buy_conf < min_conf_required:
        buy_reasons.append(f"confidence too low ({buy_conf:.0f} < {min_conf_required:.0f})")

    if not sell_direction_ok:
        sell_reasons.append("trend mismatch")
    if not ctx["sell_rsi_ok"]:
        sell_reasons.append(f"RSI too low for short ({rsi:.1f})")
    if not ctx["effective_volume_spike"]:
        sell_reasons.append(f"low volume ({volume_accel:.2f}x < {settings.volume_spike_factor:.2f}x)")
    if not ctx["sell_taker_positive"]:
        sell_reasons.append(f"weak taker sell flow ({taker_bs:.3f} > {ctx['sell_taker_threshold']:.2f})")
    if ctx["sell_blocked"]:
        sell_reasons.append("nearby support block")
    if ctx["ema50_dist_pct"] > 1.5 and not momentum_override and ctx["price"] < ctx["ema50_val"]:
        sell_reasons.append(f"EMA50 extension penalty -10 (dist {ctx['ema50_dist_pct']:.1f}%)")
    if ctx["tf_5m_rsi"] < 30 and not momentum_override:
        sell_reasons.append(f"5M RSI oversold ({ctx['tf_5m_rsi']:.1f} < 30)")
    if btc_sentinel_penalty < 0 and btc_1h_trend == "uptrend":
        sell_reasons.append(f"BTC Sentinel: BTC 1H bullish ({btc_sentinel_penalty})")
    if not ctx["sell_rejection"] and not (ctx["volatility_breakout_short"] or momentum_override) and ctx["sell_conditions_met"] < 6:
        sell_reasons.append("missing bearish rejection candle (marginal setup)")
    if not ctx["supertrend_bear"] and not ctx["volatility_breakout_short"]\
            and not (ctx["sell_taker_positive"] and taker_bs < 1.0 and (ctx["oi_increasing"] or ctx["co_oi_change_pct"] < 0)):
        sell_reasons.append(f"SuperTrend bullish/neutral (dir={supertrend_dir:+d})")
    if sell_conf < min_conf_required:
        sell_reasons.append(f"confidence too low ({sell_conf:.0f} < {min_conf_required:.0f})")

    _log_signal_rejection(symbol, [f"BUY fail: {', '.join(buy_reasons) or 'n/a'}"], snapshot)
    _log_signal_rejection(symbol, [f"SELL fail: {', '.join(sell_reasons) or 'n/a'}"], snapshot)
    LOGGER.debug(
        "%s buy_major=%s sell_major=%s buy_valid=%s sell_valid=%s conf_buy=%.0f conf_sell=%.0f",
        symbol, buy_major, sell_major, buy_valid, sell_valid, buy_conf, sell_conf,
    )

    # ── Intelligent Telegram Output (Category B / C) ─────────────────────
    if _debug_msgs is not None:
        buy_met = [k for k, v in buy_major.items() if v]
        sell_met = [k for k, v in sell_major.items() if v]
        best_side = "BUY" if buy_conf >= sell_conf else "SELL"
        best_conf = max(buy_conf, sell_conf)
        best_met = buy_met if best_side == "BUY" else sell_met
        best_reasons = buy_reasons if best_side == "BUY" else sell_reasons

        if best_conf >= 60 and len(best_met) >= 3:
            safe_met = html.escape(', '.join(best_met))
            safe_blockers = html.escape(', '.join(best_reasons) or 'none')
            ob_str = f"OB: {ctx['ob_bid_pct']:.0f}%/{ctx['ob_ask_pct']:.0f}%"
            mo_flag = "  ⚡ <b>MOMENTUM OVERRIDE</b>" if momentum_override else ""
            tf_info = f"4H:{ctx['tf_4h_trend'][:2].upper()} 1H:{ctx['tf_1h_trend'][:2].upper()} 15M:{trend[:2].upper()}"
            dbg = (
                f"\U0001f50d <b>POTENTIAL SETUP {best_side}</b>{mo_flag}\n"
                f"<b>{html.escape(symbol)}</b>  Score: {best_conf:.0f} / {min_conf_required:.0f}\n"
                f"\U0001f4ca TF: {tf_info}  |  5M RSI: {ctx['tf_5m_rsi']:.1f}\n"
                f"\u2705 Met ({len(best_met)}): {safe_met}\n"
                f"\u274c Blockers: {safe_blockers}\n"
                f"\U0001f4d6 {ob_str}  |  Taker: {taker_bs:.3f}  |  RSI: {rsi:.1f}"
            )
            _debug_msgs.append(dbg)
            LOGGER.warning("POTENTIAL %s [%s] score=%.0f/%s met=%s",
                           best_side, symbol, best_conf, min_conf_required, best_met)
        elif best_conf >= 55:
            LOGGER.info("QUIET [%s] %s score=%.0f (below Category B threshold 60)",
                        symbol, best_side, best_conf)


def _build_signal(
    ctx: Dict,
    snapshot: Dict,
    settings: Settings,
    side: str,
    confidence: float,
    sup: Optional[float],
    res: Optional[float],
) -> Optional[Signal]:
    """Build the final Signal object with trade levels, position sizing, and context."""
    symbol = ctx["symbol"]
    price = ctx["price"]
    atr = ctx["atr"]

    # Aggressive TP for high-confidence setups
    rr_fallback = 3.0 if confidence > 80 else 2.0

    # 5M tight SL
    tight_sup = sup
    tight_res = res
    if side == "BUY" and ctx["tf_5m_swing_low"] > 0:
        tight_sup = ctx["tf_5m_swing_low"]
    elif side == "SELL" and ctx["tf_5m_swing_high"] > 0:
        tight_res = ctx["tf_5m_swing_high"]

    levels = build_trade_levels(side, price, tight_sup, tight_res, atr, risk_reward_fallback=rr_fallback)

    # Liquidity Magnet TP
    heatmap_clusters = snapshot.get("coinglass_clusters", [])
    if heatmap_clusters and side == "BUY":
        for cp in sorted(heatmap_clusters):
            if isinstance(cp, (int, float)) and cp > price:
                if cp < levels["take_profit"]:
                    levels["take_profit"] = float(cp)
                    levels["risk_reward"] = max((cp - price) / max(levels["risk_per_unit"], 1e-9), 1.0)
                break
    elif heatmap_clusters and side == "SELL":
        for cp in sorted(heatmap_clusters, reverse=True):
            if isinstance(cp, (int, float)) and cp < price:
                if cp > levels["take_profit"]:
                    levels["take_profit"] = float(cp)
                    levels["risk_reward"] = max((price - cp) / max(levels["risk_per_unit"], 1e-9), 1.0)
                break

    if levels["risk_reward"] < 2.0:
        _log_signal_rejection(symbol, [f"Risk/Reward too low ({levels['risk_reward']:.2f} < 2.0)"], snapshot)
        return None

    # Auto break-even price
    if side == "BUY":
        be_price = price + (levels["take_profit"] - price) * 0.5
    else:
        be_price = price - (price - levels["take_profit"]) * 0.5

    # Tiered signal classification
    tier = "A" if confidence >= 85 else "B"

    # Position sizing with Kelly criterion
    caution_mode = bool(snapshot.get("caution_mode", False))
    caution_reason = str(snapshot.get("caution_reason", "High-impact economic event soon"))

    kelly_override = None
    if settings.kelly_enabled:
        kelly_override = calculate_kelly_fraction(
            win_rate=settings.kelly_default_win_rate,
            avg_win_loss_ratio=settings.kelly_default_avg_rr,
            fraction=settings.kelly_fraction,
        )

    size = calculate_position_size(
        capital_usdt=settings.capital_usdt,
        risk_per_trade=settings.risk_per_trade,
        risk_per_unit=levels["risk_per_unit"],
        caution_mode=caution_mode,
        caution_mode_scale=settings.caution_mode_position_scale,
        kelly_fraction_override=kelly_override,
    )
    if tier == "B":
        size *= 0.6

    # ── Context lines ────────────────────────────────────────────────────────
    trend = ctx["trend"]
    htf_trend = ctx["htf_trend"]
    context_lines = [f"{trend.title()} + {htf_trend.title()} confirmed"]
    context_lines.append(f"Quad-TF: 4H={ctx['tf_4h_trend']} 1H={ctx['tf_1h_trend']} 15M={trend} 5M-RSI={ctx['tf_5m_rsi']:.1f}")
    if ctx["triple_align_long"] or ctx["triple_align_short"]:
        context_lines.append("✅ Triple Alignment (4H+1H+15M) +20")
    if ctx["quad_4h_long_boost"] > 0 and side == "BUY":
        context_lines.append("✅ 4H Macro Bullish +15")
    if ctx["quad_4h_short_boost"] > 0 and side == "SELL":
        context_lines.append("✅ 4H Macro Bearish +15")
    if ctx["btc_sentinel_penalty"] < 0:
        context_lines.append(f"⚠️ BTC Sentinel: BTC 1H={ctx['btc_1h_trend']} ({ctx['btc_sentinel_penalty']:+d})")
    if ctx["tf_5m_ema9_pullback"]:
        context_lines.append("✅ 5M EMA9 pullback entry")

    is_uptrend = ctx["is_uptrend"]
    is_downtrend = ctx["is_downtrend"]
    buy_taker_positive = ctx["buy_taker_positive"]
    sell_taker_positive = ctx["sell_taker_positive"]
    oi_increasing = ctx["oi_increasing"]
    oi_change_pct = ctx["oi_change_pct"]
    taker_bs = ctx["taker_bs"]
    funding = ctx["funding"]
    top_ls = ctx["top_ls"]
    rsi = ctx["rsi"]
    volume_spike = ctx["volume_spike"]
    co_oi_change_pct = ctx["co_oi_change_pct"]
    co_global_ls = ctx["co_global_ls"]
    co_liq_24h_usd = ctx["co_liq_24h_usd"]
    co_liq_1h_usd = ctx["co_liq_1h_usd"]
    co_short_liq_1h_usd = ctx["co_short_liq_1h_usd"]
    coinalyze_available = ctx["coinalyze_available"]
    volume_accel = ctx["volume_accel"]

    if is_uptrend and buy_taker_positive and oi_increasing:
        binance_sentiment = "Bullish"
    elif is_downtrend and sell_taker_positive and oi_change_pct <= 0:
        binance_sentiment = "Bearish"
    else:
        binance_sentiment = "Neutral"

    if not coinalyze_available:
        global_sentiment = "Unavailable"
    elif ctx["co_bullish"]:
        global_sentiment = "Bullish"
    elif ctx["co_bearish"]:
        global_sentiment = "Bearish"
    else:
        global_sentiment = "Neutral"

    global_context_lines: List[str] = [
        f"Aggregated OI change: {co_oi_change_pct:+.2f}%",
        f"Global Long/Short ratio: {co_global_ls:.2f}",
    ]
    if side == "BUY":
        context_lines.append("Support bounce")
    else:
        context_lines.append("Resistance rejection")
    if volume_spike:
        context_lines.append("Volume spike detected")
    if oi_increasing:
        context_lines.append(f"OI increasing ({oi_change_pct:.2f}%)")
    if (side == "BUY" and funding < 0) or (side == "SELL" and funding > 0):
        context_lines.append(f"Funding {'negative' if funding < 0 else 'positive'} bias")
    if top_ls >= settings.min_top_trader_ls_long and side == "BUY":
        context_lines.append(f"Top traders net long (ratio={top_ls:.2f})")
    if top_ls <= settings.max_top_trader_ls_short and side == "SELL":
        context_lines.append(f"Top traders net short (ratio={top_ls:.2f})")
    if taker_bs >= settings.min_taker_buy_sell_long and side == "BUY":
        context_lines.append(f"Taker buy pressure active (ratio={taker_bs:.2f})")
    if taker_bs <= settings.max_taker_buy_sell_short and side == "SELL":
        context_lines.append(f"Taker sell pressure active (ratio={taker_bs:.2f})")
    if ctx["oi_divergence"]:
        context_lines.append(
            f"OI_Divergence: Binance OI {oi_change_pct:+.2f}% vs Global OI {co_oi_change_pct:+.2f}%"
        )
    if co_liq_1h_usd >= settings.coinalyze_liquidation_spike_1h_usd:
        context_lines.append("VOLATILITY WARNING: liquidation spike in last 1h")
    if ctx["price_rising"] and co_short_liq_1h_usd >= settings.coinalyze_short_liq_spike_1h_usd:
        context_lines.append(
            f"Short squeeze detected: short liquidations 1h=${co_short_liq_1h_usd:,.0f} (+25)"
        )
    if ctx["volatility_breakout_long"]:
        context_lines.append(
            f"Volatility Breakout LONG: close {price:.4f} > BB upper {ctx['bb']['upper']:.4f} with liquidation spike"
        )
    if ctx["volatility_breakout_short"]:
        context_lines.append(
            f"Volatility Breakout SHORT: close {price:.4f} < BB lower {ctx['bb']['lower']:.4f} with liquidation spike"
        )
    if ctx["coinalyze_liq_long_override"]:
        context_lines.append(
            f"Coinalyze override: short liquidations 1h=${co_short_liq_1h_usd:,.0f} prioritized for LONG"
        )
    if ctx["fast_track"]:
        context_lines.append(
            f"Fast-Track momentum: candle move {ctx['candle_move_pct']:.2f}% with volume x{volume_accel:.2f}"
        )
    if ctx["single_source_mode"]:
        context_lines.append("Single Source - Proceed with Caution")
    if co_liq_24h_usd > 0:
        global_context_lines.append(f"24h liquidations: ${co_liq_24h_usd:,.0f}")
    context_lines.append(f"RSI {'valid for long' if side == 'BUY' else 'valid for short'} ({rsi:.1f})")
    if kelly_override is not None:
        context_lines.append(f"Kelly Criterion active (f={kelly_override:.4f}, fraction={settings.kelly_fraction})")
    if caution_mode:
        reduction_pct = (1.0 - settings.caution_mode_position_scale) * 100
        context_lines.append(
            f"CAUTION MODE active ({caution_reason}) - position size reduced by {reduction_pct:.0f}%"
        )

    return Signal(
        pair=snapshot["symbol"],
        side=side,
        confidence=confidence,
        entry=levels["entry"],
        stop_loss=levels["stop_loss"],
        take_profit=levels["take_profit"],
        risk_reward=levels["risk_reward"],
        position_size=size,
        binance_sentiment=binance_sentiment,
        global_sentiment=global_sentiment,
        context_lines=context_lines,
        global_context_lines=global_context_lines,
        tier=tier,
        break_even_price=be_price,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator: evaluate_pair — thin wrapper calling the 3 phases
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_pair(
    snapshot: Dict,
    settings: Settings,
    _debug_msgs: Optional[List[str]] = None,
    btc_snapshot: Optional[Dict] = None,
) -> Optional[Signal]:
    """Evaluate a single trading pair. Thin orchestrator calling compute_scores,
    apply_gates, and generate_signal in sequence."""
    ctx = compute_scores(snapshot, settings, btc_snapshot)
    if ctx is None:
        return None
    apply_gates(ctx, snapshot, settings)
    return generate_signal(ctx, snapshot, settings, _debug_msgs)


def build_telegram_message(signal: Signal) -> str:
    side_emoji = "🟢" if signal.side == "BUY" else "🔴"
    action_label = "LONG" if signal.side == "BUY" else "SHORT"

    # Tier label
    if signal.tier == "A":
        tier_label = "🅰️ FULL POSITION"
    else:
        tier_label = "🅱️ MEDIUM POSITION"

    # Star rating by confidence tier.
    if signal.confidence >= 90:
        stars = "⭐⭐⭐⭐⭐"
    elif signal.confidence >= 85:
        stars = "⭐⭐⭐⭐"
    elif signal.confidence >= 80:
        stars = "⭐⭐⭐"
    elif signal.confidence >= 75:
        stars = "⭐⭐"
    else:
        stars = "⭐"

    divider = "─" * 24

    context_lines = "\n".join(f"• {line}" for line in signal.context_lines)
    global_context_lines = "\n".join(f"• {line}" for line in signal.global_context_lines)

    return (
        f"🚨 <b>SIGNAL ALERT</b> {side_emoji} <b>{action_label}</b>  {tier_label}\n"
        f"<b>{signal.pair}</b>\n"
        f"Confidence: <b>{signal.confidence:.0f}%</b>  {stars}\n"
        f"\n{divider}\n"
        f"📐 <b>Trade Levels</b>\n"
        f"Entry:       <code>{signal.entry:.4f}</code>\n"
        f"Stop Loss:   <code>{signal.stop_loss:.4f}</code>\n"
        f"Take Profit: <code>{signal.take_profit:.4f}</code>\n"
        f"Break-Even:  <code>{signal.break_even_price:.4f}</code>\n"
        f"R:R          <b>1:{signal.risk_reward:.1f}</b>\n"
        f"Position:    <b>{signal.position_size:.2f} USDT</b>\n"
        f"\n{divider}\n"
        f"📊 Binance Sentiment:  <b>{signal.binance_sentiment}</b>\n"
        f"🌐 Global Sentiment:   <b>{signal.global_sentiment}</b>\n"
        f"\n{divider}\n"
        f"🌐 <b>Global Market Context (Coinalyze)</b>\n"
        f"{global_context_lines}\n"
        f"\n{divider}\n"
        f"📋 <b>Why this signal fired?</b>\n"
        f"{context_lines}\n"
        f"\n{divider}\n"
        f"🛡️ <b>Risk Management</b>\n"
        f"• Move SL → Entry when price reaches <code>{signal.break_even_price:.4f}</code> (50% to TP)"
    )
