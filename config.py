import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


@dataclass
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    scan_pairs: List[str] = field(
        default_factory=lambda: os.getenv(
            "SCAN_PAIRS",
            "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,SUIUSDT,DOGEUSDT,BCHUSDT,TAOUSDT,ZECUSDT,AVAXUSDT,LINKUSDT,FARTCOINUSDT,1000PEPEUSDT,HYPEUSDT",
        ).split(",")
    )
    kline_interval: str = os.getenv("KLINE_INTERVAL", "15m")
    higher_tf_interval: str = os.getenv("HIGHER_TF_INTERVAL", "1h")
    klines_limit: int = int(os.getenv("KLINES_LIMIT", "250"))
    scan_interval_seconds: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
    pair_fetch_delay_seconds: float = float(os.getenv("PAIR_FETCH_DELAY_SECONDS", "1.5"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    prioritize_high_volume: bool = os.getenv("PRIORITIZE_HIGH_VOLUME", "true").lower() == "true"
    high_volume_scan_limit: int = int(os.getenv("HIGH_VOLUME_SCAN_LIMIT", "0"))

    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "60"))
    max_signals_per_day: int = int(os.getenv("MAX_SIGNALS_PER_DAY", "3"))
    max_signals_per_cycle: int = int(os.getenv("MAX_SIGNALS_PER_CYCLE", "1"))
    min_confidence_floor: float = float(os.getenv("MIN_CONFIDENCE_FLOOR", "60"))

    capital_usdt: float = float(os.getenv("CAPITAL_USDT", "6000"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.01"))

    volume_spike_factor: float = float(os.getenv("VOLUME_SPIKE_FACTOR", "1.05"))
    level_proximity_atr_factor: float = float(os.getenv("LEVEL_PROXIMITY_ATR_FACTOR", "0.35"))
    nearby_block_atr_factor: float = float(os.getenv("NEARBY_BLOCK_ATR_FACTOR", "1.2"))

    funding_neutral_abs: float = float(os.getenv("FUNDING_NEUTRAL_ABS", "0.0001"))
    strong_spike_atr_mult: float = float(os.getenv("STRONG_SPIKE_ATR_MULT", "3.0"))
    min_oi_change_pct: float = float(os.getenv("MIN_OI_CHANGE_PCT", "0.2"))
    require_non_zero_derivatives: bool = (
        os.getenv("REQUIRE_NON_ZERO_DERIVATIVES", "true").lower() == "true"
    )
    min_adx: float = float(os.getenv("MIN_ADX", "12"))
    min_ema_gap_ratio: float = float(os.getenv("MIN_EMA_GAP_RATIO", "0.0025"))
    min_htf_ema_gap_ratio: float = float(os.getenv("MIN_HTF_EMA_GAP_RATIO", "0.0020"))
    min_rejection_wick_ratio: float = float(os.getenv("MIN_REJECTION_WICK_RATIO", "0.35"))
    max_extension_atr: float = float(os.getenv("MAX_EXTENSION_ATR", "1.2"))
    fear_greed_min_long: int = int(os.getenv("FEAR_GREED_MIN_LONG", "20"))
    fear_greed_max_short: int = int(os.getenv("FEAR_GREED_MAX_SHORT", "80"))
    extreme_fear_threshold: int = int(os.getenv("EXTREME_FEAR_THRESHOLD", "25"))
    extreme_fear_long_penalty: float = float(os.getenv("EXTREME_FEAR_LONG_PENALTY", "10"))
    extreme_fear_short_boost: float = float(os.getenv("EXTREME_FEAR_SHORT_BOOST", "5"))
    min_top_trader_ls_long: float = float(os.getenv("MIN_TOP_TRADER_LS_LONG", "0.9"))
    max_top_trader_ls_short: float = float(os.getenv("MAX_TOP_TRADER_LS_SHORT", "1.1"))
    min_taker_buy_sell_long: float = float(os.getenv("MIN_TAKER_BUY_SELL_LONG", "1.02"))
    max_taker_buy_sell_short: float = float(os.getenv("MAX_TAKER_BUY_SELL_SHORT", "0.98"))

    # --- Sentiment Engine ---
    cryptopanic_token: str = os.getenv("CRYPTOPANIC_TOKEN", "")
    coinalyze_api_key: str = os.getenv("COINALYZE_API_KEY", "")
    coinalyze_cooldown_seconds: float = float(os.getenv("COINALYZE_COOLDOWN_SECONDS", "1.5"))
    # Max extra confidence points added by positive/negative macro sentiment.
    news_sentiment_weight: int = int(os.getenv("NEWS_SENTIMENT_WEIGHT", "10"))
    # Extra confidence points added when breaking Trump+Crypto positive news detected.
    news_trump_boost: float = float(os.getenv("NEWS_TRUMP_BOOST", "20.0"))

    # --- Coinalyze Cross-Platform Confirmation ---
    min_coinalyze_agg_oi_long: float = float(os.getenv("MIN_COINALYZE_AGG_OI_LONG", "0.10"))
    max_coinalyze_agg_oi_short: float = float(os.getenv("MAX_COINALYZE_AGG_OI_SHORT", "-0.05"))
    min_coinalyze_ls_long: float = float(os.getenv("MIN_COINALYZE_LS_LONG", "1.0"))
    max_coinalyze_ls_short: float = float(os.getenv("MAX_COINALYZE_LS_SHORT", "1.0"))
    max_predicted_funding_for_long: float = float(
        os.getenv("MAX_PREDICTED_FUNDING_FOR_LONG", "0.0005")
    )
    min_predicted_funding_for_short: float = float(
        os.getenv("MIN_PREDICTED_FUNDING_FOR_SHORT", "-0.0005")
    )
    coinalyze_oi_divergence_penalty: float = float(
        os.getenv("COINALYZE_OI_DIVERGENCE_PENALTY", "10")
    )
    single_source_penalty: float = float(os.getenv("SINGLE_SOURCE_PENALTY", "8"))
    coinalyze_liquidation_spike_1h_usd: float = float(
        os.getenv("COINALYZE_LIQUIDATION_SPIKE_1H_USD", "50000000")
    )
    coinalyze_short_liq_spike_1h_usd: float = float(
        os.getenv("COINALYZE_SHORT_LIQ_SPIKE_1H_USD", "15000000")
    )

    # --- Aggressive Momentum Mode ---
    fast_track_candle_move_pct: float = float(os.getenv("FAST_TRACK_CANDLE_MOVE_PCT", "1.2"))
    fast_track_volume_mult: float = float(os.getenv("FAST_TRACK_VOLUME_MULT", "1.8"))
    breakout_confluence_boost: float = float(os.getenv("BREAKOUT_CONFLUENCE_BOOST", "12"))

    # --- Quad-TF & Conviction Tuning ---
    quad_4h_boost: int = int(os.getenv("QUAD_4H_BOOST", "15"))
    triple_alignment_boost: int = int(os.getenv("TRIPLE_ALIGNMENT_BOOST", "20"))
    btc_sentinel_max_penalty: int = int(os.getenv("BTC_SENTINEL_MAX_PENALTY", "25"))
    cluster_proximity_boost: int = int(os.getenv("CLUSTER_PROXIMITY_BOOST", "20"))
    min_buy_conditions: int = int(os.getenv("MIN_BUY_CONDITIONS", "4"))
    min_sell_conditions: int = int(os.getenv("MIN_SELL_CONDITIONS", "4"))

    # --- Economic Event Caution Mode ---
    economic_event_lookahead_hours: int = int(os.getenv("ECONOMIC_EVENT_LOOKAHEAD_HOURS", "2"))
    caution_mode_position_scale: float = float(os.getenv("CAUTION_MODE_POSITION_SCALE", "0.5"))

    # --- Kelly Criterion Position Sizing ---
    kelly_enabled: bool = os.getenv("KELLY_ENABLED", "false").lower() == "true"
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    kelly_default_win_rate: float = float(os.getenv("KELLY_DEFAULT_WIN_RATE", "0.55"))
    kelly_default_avg_rr: float = float(os.getenv("KELLY_DEFAULT_AVG_RR", "2.5"))

    # --- Parallel Fetching ---
    max_concurrent_fetches: int = int(os.getenv("MAX_CONCURRENT_FETCHES", "5"))

    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "signals.db")

    backtest_pair: str = os.getenv("BACKTEST_PAIR", "BTCUSDT")
    backtest_bars: int = int(os.getenv("BACKTEST_BARS", "1200"))
    backtest_lookahead_bars: int = int(os.getenv("BACKTEST_LOOKAHEAD_BARS", "24"))
    backtest_all_pairs: bool = os.getenv("BACKTEST_ALL_PAIRS", "false").lower() == "true"

    optimize_bars: int = int(os.getenv("OPTIMIZE_BARS", "1200"))
    optimize_lookahead_bars: int = int(os.getenv("OPTIMIZE_LOOKAHEAD_BARS", "24"))
    optimize_top_n: int = int(os.getenv("OPTIMIZE_TOP_N", "10"))
    optimize_export_csv: str = os.getenv("OPTIMIZE_EXPORT_CSV", "backtests/optimization_top.csv")
    optimize_auto_apply_best: bool = (
        os.getenv("OPTIMIZE_AUTO_APPLY_BEST", "false").lower() == "true"
    )
    optimize_apply_env_path: str = os.getenv("OPTIMIZE_APPLY_ENV_PATH", ".env")

    def validate(self) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            raise ValueError("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        if not (0 < self.min_confidence <= 100):
            raise ValueError(f"MIN_CONFIDENCE must be between 1 and 100, got {self.min_confidence}")
        if not (0 < self.risk_per_trade < 0.5):
            raise ValueError(
                f"RISK_PER_TRADE must be between 0 and 0.5 (50%), got {self.risk_per_trade}"
            )
        if self.max_signals_per_day < 1:
            raise ValueError("MAX_SIGNALS_PER_DAY must be at least 1")
        if not (0 <= self.fear_greed_min_long <= 100):
            raise ValueError("FEAR_GREED_MIN_LONG must be between 0 and 100")
        if not (0 <= self.fear_greed_max_short <= 100):
            raise ValueError("FEAR_GREED_MAX_SHORT must be between 0 and 100")
        if not (0 <= self.extreme_fear_threshold <= 100):
            raise ValueError("EXTREME_FEAR_THRESHOLD must be between 0 and 100")
        if not (1 <= self.economic_event_lookahead_hours <= 24):
            raise ValueError("ECONOMIC_EVENT_LOOKAHEAD_HOURS must be between 1 and 24")
        if not (0 < self.caution_mode_position_scale <= 1.0):
            raise ValueError("CAUTION_MODE_POSITION_SCALE must be between 0 (exclusive) and 1")
        if self.pair_fetch_delay_seconds < 0:
            raise ValueError("PAIR_FETCH_DELAY_SECONDS must be >= 0")
        if self.coinalyze_oi_divergence_penalty < 0:
            raise ValueError("COINALYZE_OI_DIVERGENCE_PENALTY must be >= 0")
        if self.single_source_penalty < 0:
            raise ValueError("SINGLE_SOURCE_PENALTY must be >= 0")
        if self.coinalyze_cooldown_seconds < 0.1:
            raise ValueError("COINALYZE_COOLDOWN_SECONDS must be at least 0.1")
        if self.coinalyze_liquidation_spike_1h_usd < 0:
            raise ValueError("COINALYZE_LIQUIDATION_SPIKE_1H_USD must be >= 0")
        if self.coinalyze_short_liq_spike_1h_usd < 0:
            raise ValueError("COINALYZE_SHORT_LIQ_SPIKE_1H_USD must be >= 0")


# ── Scoring Weights (moved from strategy.py for configurability) ────────────
SCORING_WEIGHTS = {
    "taker_flow": 20,
    "aggregated_sentiment": 15,
    "trend": 15,
    "key_level": 12,
    "volume": 10,
    "oi": 8,
    "rsi": 8,
    "funding": 6,
    "top_trader": 6,
    "liquidations": 12,
    "volatility_breakout": 20,
    "supertrend": 8,
}

settings = Settings()
