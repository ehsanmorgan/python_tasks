import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

from backtester import BacktestResult, run_backtest_many
from config import Settings
from data_fetcher import BinanceFetcher


@dataclass
class OptimizationResult:
    params: Dict[str, float]
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_r_multiple: float
    expectancy_r: float
    avg_risk_reward: float
    max_drawdown_r: float
    max_drawdown_pct: float


def _aggregate_results(
    results: List[BacktestResult], params: Dict[str, float], risk_per_trade: float
) -> OptimizationResult:
    total_trades = sum(r.total_trades for r in results)
    wins = sum(r.wins for r in results)
    losses = sum(r.losses for r in results)
    net_r = sum(r.net_r_multiple for r in results)
    max_dd = max((r.max_drawdown_r for r in results), default=0.0)
    weighted_rr_sum = sum(r.avg_risk_reward * r.total_trades for r in results)

    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    expectancy = (net_r / total_trades) if total_trades else 0.0
    avg_risk_reward = (weighted_rr_sum / total_trades) if total_trades else 0.0
    max_drawdown_pct = max_dd * risk_per_trade * 100.0

    return OptimizationResult(
        params=params,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        net_r_multiple=net_r,
        expectancy_r=expectancy,
        avg_risk_reward=avg_risk_reward,
        max_drawdown_r=max_dd,
        max_drawdown_pct=max_drawdown_pct,
    )


def _param_grid() -> List[Dict[str, float]]:
    grid: List[Dict[str, float]] = []
    for min_confidence in [85.0, 88.0, 90.0]:
        for min_adx in [18.0, 20.0, 22.0]:
            for min_oi_change_pct in [0.1, 0.2, 0.35]:
                for volume_spike_factor in [1.5, 1.7]:
                    for min_rejection_wick_ratio in [0.30, 0.40]:
                        for max_extension_atr in [1.0, 1.2]:
                            grid.append(
                                {
                                    "min_confidence": min_confidence,
                                    "min_adx": min_adx,
                                    "min_oi_change_pct": min_oi_change_pct,
                                    "volume_spike_factor": volume_spike_factor,
                                    "min_rejection_wick_ratio": min_rejection_wick_ratio,
                                    "max_extension_atr": max_extension_atr,
                                }
                            )
    return grid


def _sort_key(item: OptimizationResult) -> tuple:
    return (
        item.expectancy_r,
        item.win_rate,
        item.net_r_multiple,
        -item.max_drawdown_pct,
        item.total_trades,
    )


def meets_apply_conditions(row: OptimizationResult) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if row.win_rate < 60.0:
        reasons.append(f"win_rate {row.win_rate:.2f}% < 60%")
    if row.total_trades < 50:
        reasons.append(f"total_trades {row.total_trades} < 50")
    if row.max_drawdown_pct > 20.0:
        reasons.append(f"max_drawdown {row.max_drawdown_pct:.2f}% > 20%")
    if row.avg_risk_reward < 2.0:
        reasons.append(f"avg_risk_reward {row.avg_risk_reward:.2f} < 2.0")
    return (len(reasons) == 0, reasons)


def _upsert_env_params(env_path: str, params: Dict[str, float]) -> str:
    path = Path(env_path)
    existing_lines: List[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    updates = {
        "MIN_CONFIDENCE": str(params["min_confidence"]),
        "MIN_ADX": str(params["min_adx"]),
        "MIN_OI_CHANGE_PCT": str(params["min_oi_change_pct"]),
        "VOLUME_SPIKE_FACTOR": str(params["volume_spike_factor"]),
        "MIN_REJECTION_WICK_RATIO": str(params["min_rejection_wick_ratio"]),
        "MAX_EXTENSION_ATR": str(params["max_extension_atr"]),
    }

    remaining = dict(updates)
    out_lines: List[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue

        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in remaining:
            out_lines.append(f"{key}={remaining.pop(key)}")
        else:
            out_lines.append(line)

    for key, value in remaining.items():
        out_lines.append(f"{key}={value}")

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return str(path)


def conditional_apply_best(
    ranked: List[OptimizationResult],
    env_path: str,
) -> Tuple[bool, str]:
    if not ranked:
        return False, "No ranked optimization results available"

    best = ranked[0]
    ok, reasons = meets_apply_conditions(best)
    if not ok:
        return False, "Best set not applied: " + "; ".join(reasons)

    target = _upsert_env_params(env_path, best.params)
    return True, f"Applied best set to {target}"


def _write_optimization_csv(path: str, rows: List[OptimizationResult]) -> str:
    csv_path = Path(path)
    if csv_path.parent and str(csv_path.parent) != ".":
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "total_trades",
                "wins",
                "losses",
                "win_rate",
                "net_r_multiple",
                "expectancy_r",
                "avg_risk_reward",
                "max_drawdown_r",
                "max_drawdown_pct",
                "min_confidence",
                "min_adx",
                "min_oi_change_pct",
                "volume_spike_factor",
                "min_rejection_wick_ratio",
                "max_extension_atr",
            ]
        )

        for i, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    i,
                    row.total_trades,
                    row.wins,
                    row.losses,
                    f"{row.win_rate:.2f}",
                    f"{row.net_r_multiple:.4f}",
                    f"{row.expectancy_r:.4f}",
                    f"{row.avg_risk_reward:.4f}",
                    f"{row.max_drawdown_r:.4f}",
                    f"{row.max_drawdown_pct:.4f}",
                    row.params["min_confidence"],
                    row.params["min_adx"],
                    row.params["min_oi_change_pct"],
                    row.params["volume_spike_factor"],
                    row.params["min_rejection_wick_ratio"],
                    row.params["max_extension_atr"],
                ]
            )

    return str(csv_path)


async def optimize_parameters(
    fetcher: BinanceFetcher,
    settings: Settings,
    pairs: List[str],
    bars: int,
    lookahead_bars: int,
    top_n: int,
    export_csv_path: str,
) -> List[OptimizationResult]:
    ranked: List[OptimizationResult] = []

    for params in _param_grid():
        tuned = replace(
            settings,
            min_confidence=float(params["min_confidence"]),
            min_adx=float(params["min_adx"]),
            min_oi_change_pct=float(params["min_oi_change_pct"]),
            volume_spike_factor=float(params["volume_spike_factor"]),
            min_rejection_wick_ratio=float(params["min_rejection_wick_ratio"]),
            max_extension_atr=float(params["max_extension_atr"]),
        )

        results = await run_backtest_many(
            fetcher=fetcher,
            settings=tuned,
            pairs=pairs,
            bars=bars,
            lookahead_bars=lookahead_bars,
        )
        summary = _aggregate_results(results, params, settings.risk_per_trade)

        # Ignore parameter sets that produce too few samples to be trustworthy.
        if summary.total_trades < 10:
            continue

        ranked.append(summary)

    ranked = sorted(ranked, key=_sort_key, reverse=True)
    top = ranked[:top_n]

    if export_csv_path:
        _write_optimization_csv(export_csv_path, top)

    return top
