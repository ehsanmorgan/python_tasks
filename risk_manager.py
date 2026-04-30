from typing import Dict, Optional


def calculate_kelly_fraction(
    win_rate: float,
    avg_win_loss_ratio: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly criterion for position sizing.

    Returns the fraction of capital to risk per trade.
    Uses quarter-Kelly by default for safety.
    """
    if win_rate <= 0 or win_rate >= 1.0 or avg_win_loss_ratio <= 0:
        return 0.0
    q = 1.0 - win_rate
    kelly = (win_rate * avg_win_loss_ratio - q) / avg_win_loss_ratio
    # Fractional Kelly, capped at 5% of capital per trade
    return max(min(kelly * fraction, 0.05), 0.0)


def build_trade_levels(
    side: str,
    entry: float,
    support: Optional[float],
    resistance: Optional[float],
    atr: float,
    risk_reward_fallback: float = 2.0,
    risk_reward_preferred: float = 3.0,
) -> Dict[str, float]:
    atr_buffer = max(atr * 0.2, entry * 0.001)

    if side == "BUY":
        stop_loss = (support - atr_buffer) if support else (entry - atr)
        risk_per_unit = max(entry - stop_loss, entry * 0.001)
        preferred_tp = entry + risk_per_unit * risk_reward_preferred
        fallback_tp = entry + risk_per_unit * risk_reward_fallback

        if resistance and resistance > entry and (resistance - entry) >= (risk_per_unit * risk_reward_preferred):
            take_profit = preferred_tp
            rr = risk_reward_preferred
        else:
            take_profit = fallback_tp
            rr = risk_reward_fallback

    else:
        stop_loss = (resistance + atr_buffer) if resistance else (entry + atr)
        risk_per_unit = max(stop_loss - entry, entry * 0.001)
        preferred_tp = entry - risk_per_unit * risk_reward_preferred
        fallback_tp = entry - risk_per_unit * risk_reward_fallback

        if support and support < entry and (entry - support) >= (risk_per_unit * risk_reward_preferred):
            take_profit = preferred_tp
            rr = risk_reward_preferred
        else:
            take_profit = fallback_tp
            rr = risk_reward_fallback

    return {
        "entry": float(entry),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "risk_reward": float(rr),
        "risk_per_unit": float(risk_per_unit),
    }


def calculate_position_size(
    capital_usdt: float,
    risk_per_trade: float,
    risk_per_unit: float,
    caution_mode: bool = False,
    caution_mode_scale: float = 0.5,
    kelly_fraction_override: Optional[float] = None,
) -> float:
    if risk_per_unit <= 0:
        return 0.0

    effective_risk_per_trade = risk_per_trade
    if kelly_fraction_override is not None and kelly_fraction_override > 0:
        effective_risk_per_trade = min(kelly_fraction_override, risk_per_trade * 2)
    if caution_mode:
        effective_risk_per_trade *= max(min(caution_mode_scale, 1.0), 0.0)

    risk_budget = capital_usdt * effective_risk_per_trade
    raw_size = risk_budget / risk_per_unit
    max_size = capital_usdt * 10
    return max(min(raw_size, max_size), 0.0)
