from typing import Dict, List

import pandas as pd


def detect_equal_highs_lows(df: pd.DataFrame, tolerance: float = 0.0008) -> Dict[str, List[float]]:
    highs = df["high"].tail(80).tolist()
    lows = df["low"].tail(80).tolist()

    equal_highs: List[float] = []
    equal_lows: List[float] = []

    for i in range(len(highs) - 1):
        for j in range(i + 5, len(highs)):
            if abs(highs[i] - highs[j]) / max(highs[i], 1e-9) <= tolerance:
                equal_highs.append((highs[i] + highs[j]) / 2.0)

    for i in range(len(lows) - 1):
        for j in range(i + 5, len(lows)):
            if abs(lows[i] - lows[j]) / max(lows[i], 1e-9) <= tolerance:
                equal_lows.append((lows[i] + lows[j]) / 2.0)

    return {
        "equal_highs": _compress(equal_highs),
        "equal_lows": _compress(equal_lows),
    }


def detect_fake_breakout(df: pd.DataFrame) -> Dict[str, bool]:
    if len(df) < 30:
        return {"bull_trap": False, "bear_trap": False}

    recent = df.tail(25)
    prev_high = recent.iloc[:-1]["high"].max()
    prev_low = recent.iloc[:-1]["low"].min()
    last = recent.iloc[-1]

    bull_trap = last["high"] > prev_high and last["close"] < prev_high
    bear_trap = last["low"] < prev_low and last["close"] > prev_low

    return {"bull_trap": bool(bull_trap), "bear_trap": bool(bear_trap)}


def liquidity_context(df: pd.DataFrame) -> Dict:
    traps = detect_fake_breakout(df)
    return {
        "bull_trap": traps["bull_trap"],
        "bear_trap": traps["bear_trap"],
    }


def _compress(levels: List[float], tolerance: float = 0.0015) -> List[float]:
    if not levels:
        return []
    levels = sorted(levels)
    merged = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - merged[-1]) / max(merged[-1], 1e-9) <= tolerance:
            merged[-1] = (merged[-1] + lvl) / 2.0
        else:
            merged.append(lvl)
    return merged[-5:]
