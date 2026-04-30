from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


def to_dataframe(klines: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(klines)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def _compute_supertrend(
    df: pd.DataFrame, length: int = 7, multiplier: float = 3.0
) -> pd.Series:
    """
    Calculates SuperTrend direction for each row.
    Returns a Series of int: +1 = bullish (price above ST), -1 = bearish.
    Falls back to 0 when insufficient rows exist.
    """
    high = df["high"].reset_index(drop=True)
    low = df["low"].reset_index(drop=True)
    close = df["close"].reset_index(drop=True)

    atr_series = AverageTrueRange(
        high=high, low=low, close=close, window=length
    ).average_true_range()

    hl2 = (high + low) / 2.0
    raw_upper = hl2 + multiplier * atr_series
    raw_lower = hl2 - multiplier * atr_series

    n = len(close)
    final_upper = raw_upper.copy()
    final_lower = raw_lower.copy()
    direction = pd.Series(0, index=range(n), dtype=int)
    supertrend = pd.Series(np.nan, index=range(n))

    for i in range(1, n):
        if np.isnan(atr_series.iloc[i]):
            continue

        # Clamp upper band downward / lower band upward to avoid band flip
        if raw_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = raw_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if raw_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = raw_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Direction: flip from previous SuperTrend value
        prev_st = supertrend.iloc[i - 1]
        if np.isnan(prev_st):
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
        elif prev_st == final_upper.iloc[i - 1]:
            # Was bearish — stay bearish unless close breaks above upper
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
        else:
            # Was bullish — stay bullish unless close breaks below lower
            direction.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i] else 1

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    direction.index = df.index
    return direction


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ema50"] = EMAIndicator(close=result["close"], window=50).ema_indicator()
    result["ema200"] = EMAIndicator(close=result["close"], window=200).ema_indicator()
    result["rsi14"] = RSIIndicator(close=result["close"], window=14).rsi()
    result["atr14"] = AverageTrueRange(
        high=result["high"], low=result["low"], close=result["close"], window=14
    ).average_true_range()
    result["adx14"] = ADXIndicator(
        high=result["high"], low=result["low"], close=result["close"], window=14
    ).adx()
    result["vol_ma20"] = result["volume"].rolling(20).mean()
    result["supertrend_dir"] = _compute_supertrend(result, length=7, multiplier=3.0)
    return result


def detect_trend(last_row: pd.Series) -> str:
    if last_row["ema50"] > last_row["ema200"]:
        return "uptrend"
    if last_row["ema50"] < last_row["ema200"]:
        return "downtrend"
    return "sideways"


def find_support_resistance(df: pd.DataFrame, lookback: int = 120) -> Tuple[List[float], List[float]]:
    section = df.tail(lookback).reset_index(drop=True)
    supports: List[float] = []
    resistances: List[float] = []

    for i in range(2, len(section) - 2):
        low = section.loc[i, "low"]
        high = section.loc[i, "high"]

        if (
            low < section.loc[i - 1, "low"]
            and low < section.loc[i - 2, "low"]
            and low < section.loc[i + 1, "low"]
            and low < section.loc[i + 2, "low"]
        ):
            supports.append(float(low))

        if (
            high > section.loc[i - 1, "high"]
            and high > section.loc[i - 2, "high"]
            and high > section.loc[i + 1, "high"]
            and high > section.loc[i + 2, "high"]
        ):
            resistances.append(float(high))

    return dedupe_levels(supports), dedupe_levels(resistances)


def dedupe_levels(levels: List[float], tolerance: float = 0.003) -> List[float]:
    if not levels:
        return []
    levels = sorted(levels)
    merged = [levels[0]]
    for level in levels[1:]:
        if abs(level - merged[-1]) / merged[-1] <= tolerance:
            merged[-1] = (merged[-1] + level) / 2.0
        else:
            merged.append(level)
    return merged


def nearest_level(
    price: float,
    supports: List[float],
    resistances: List[float],
) -> Dict[str, Optional[float]]:
    below_supports = [s for s in supports if s <= price]
    above_resistances = [r for r in resistances if r >= price]

    nearest_support = max(below_supports) if below_supports else None
    nearest_resistance = min(above_resistances) if above_resistances else None

    return {
        "support": nearest_support,
        "resistance": nearest_resistance,
    }


def is_sideways_market(last_row: pd.Series, price: float) -> bool:
    ema_gap_ratio = abs(last_row["ema50"] - last_row["ema200"]) / max(price, 1e-9)
    atr_ratio = last_row["atr14"] / max(price, 1e-9)
    # Use OR: either flat EMAs or compressed volatility signal choppy market.
    return ema_gap_ratio < 0.0015 and atr_ratio < 0.003
