import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import Settings
from data_fetcher import BinanceFetcher
from strategy import Signal, evaluate_pair


@dataclass
class BacktestResult:
    pair: str
    total_trades: int
    wins: int
    losses: int
    open_trades: int
    win_rate: float
    net_r_multiple: float
    avg_r_per_trade: float
    expectancy_r: float
    avg_risk_reward: float
    profit_factor: float
    max_drawdown_r: float
    trades_per_day: float
    csv_path: str = ""


@dataclass
class TradeRecord:
    pair: str
    day_key: str
    entry_time: str
    exit_time: str
    side: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    result_r: float
    outcome: str


def _utc_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _slice_higher_tf(klines: List[Dict], timestamp_ms: int, min_len: int) -> Optional[List[Dict]]:
    subset = [k for k in klines if k["open_time"] <= timestamp_ms]
    if len(subset) < min_len:
        return None
    return subset


def _simulate_outcome(signal: Signal, future_klines: List[Dict]) -> tuple[float, str, str]:
    # Conservative order: if TP and SL are touched in the same candle, count as loss.
    for k in future_klines:
        high = k["high"]
        low = k["low"]
        exit_time = datetime.fromtimestamp(k["open_time"] / 1000, tz=timezone.utc).isoformat()

        if signal.side == "BUY":
            if low <= signal.stop_loss:
                return -1.0, "LOSS", exit_time
            if high >= signal.take_profit:
                return signal.risk_reward, "WIN", exit_time
        else:
            if high >= signal.stop_loss:
                return -1.0, "LOSS", exit_time
            if low <= signal.take_profit:
                return signal.risk_reward, "WIN", exit_time

    if future_klines:
        last_time = datetime.fromtimestamp(
            future_klines[-1]["open_time"] / 1000, tz=timezone.utc
        ).isoformat()
    else:
        last_time = ""
    return 0.0, "OPEN", last_time


def _write_trades_csv(csv_path: str, trades: List[TradeRecord]) -> str:
    path = Path(csv_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair",
                "day_key",
                "entry_time",
                "exit_time",
                "side",
                "confidence",
                "entry",
                "stop_loss",
                "take_profit",
                "risk_reward",
                "result_r",
                "outcome",
            ]
        )
        for t in trades:
            writer.writerow(
                [
                    t.pair,
                    t.day_key,
                    t.entry_time,
                    t.exit_time,
                    t.side,
                    f"{t.confidence:.2f}",
                    f"{t.entry:.8f}",
                    f"{t.stop_loss:.8f}",
                    f"{t.take_profit:.8f}",
                    f"{t.risk_reward:.4f}",
                    f"{t.result_r:.4f}",
                    t.outcome,
                ]
            )

    return str(path)


async def run_backtest(
    fetcher: BinanceFetcher,
    settings: Settings,
    pair: str,
    bars: int,
    lookahead_bars: int,
    export_csv_path: Optional[str] = None,
) -> BacktestResult:
    primary = await fetcher.fetch_klines(pair, settings.kline_interval, bars)
    higher = await fetcher.fetch_klines(pair, settings.higher_tf_interval, bars)

    if len(primary) < 220 or len(higher) < 220:
        return BacktestResult(
            pair=pair,
            total_trades=0,
            wins=0,
            losses=0,
            open_trades=0,
            win_rate=0.0,
            net_r_multiple=0.0,
            avg_r_per_trade=0.0,
            expectancy_r=0.0,
            avg_risk_reward=0.0,
            profit_factor=0.0,
            max_drawdown_r=0.0,
            trades_per_day=0.0,
            csv_path="",
        )

    total = 0
    wins = 0
    losses = 0
    open_trades = 0
    net_r = 0.0
    sent_per_day: Dict[str, int] = {}
    pnl_curve: List[float] = []
    trade_records: List[TradeRecord] = []
    gross_win_r = 0.0
    gross_loss_r = 0.0
    sum_risk_reward = 0.0

    start_idx = 220
    end_idx = max(start_idx + 1, len(primary) - lookahead_bars)

    for i in range(start_idx, end_idx):
        now_kline = primary[i]
        day_key = _utc_day(now_kline["open_time"])
        if sent_per_day.get(day_key, 0) >= settings.max_signals_per_day:
            continue

        higher_slice = _slice_higher_tf(higher, now_kline["open_time"], 220)
        if not higher_slice:
            continue

        snapshot = {
            "symbol": pair,
            "klines": primary[: i + 1],
            "higher_tf_klines": higher_slice,
            "funding_rate": 0.0,
            "oi_change_pct": 0.1,
        }

        signal = evaluate_pair(snapshot, settings)
        if not signal:
            continue

        total += 1
        sent_per_day[day_key] = sent_per_day.get(day_key, 0) + 1

        future = primary[i + 1 : i + 1 + lookahead_bars]
        result_r, outcome, exit_time = _simulate_outcome(signal, future)
        net_r += result_r
        pnl_curve.append(net_r)

        entry_time = datetime.fromtimestamp(now_kline["open_time"] / 1000, tz=timezone.utc).isoformat()
        trade_records.append(
            TradeRecord(
                pair=pair,
                day_key=day_key,
                entry_time=entry_time,
                exit_time=exit_time,
                side=signal.side,
                confidence=signal.confidence,
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_reward=signal.risk_reward,
                result_r=result_r,
                outcome=outcome,
            )
        )
        sum_risk_reward += signal.risk_reward

        if result_r > 0:
            wins += 1
            gross_win_r += result_r
        elif result_r < 0:
            losses += 1
            gross_loss_r += abs(result_r)
        else:
            open_trades += 1

    win_rate = (wins / total * 100.0) if total else 0.0
    avg_r_per_trade = (net_r / total) if total else 0.0
    loss_rate = (losses / total) if total else 0.0
    avg_win_r = (gross_win_r / wins) if wins else 0.0
    avg_loss_r = (gross_loss_r / losses) if losses else 0.0
    expectancy_r = (win_rate / 100.0 * avg_win_r) - (loss_rate * avg_loss_r)
    avg_risk_reward = (sum_risk_reward / total) if total else 0.0
    profit_factor = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else (999.0 if gross_win_r > 0 else 0.0)

    peak = 0.0
    max_drawdown = 0.0
    for equity in pnl_curve:
        peak = max(peak, equity)
        drawdown = peak - equity
        max_drawdown = max(max_drawdown, drawdown)

    active_days = len(sent_per_day)
    trades_per_day = (total / active_days) if active_days else 0.0
    csv_path = _write_trades_csv(export_csv_path, trade_records) if export_csv_path else ""

    return BacktestResult(
        pair=pair,
        total_trades=total,
        wins=wins,
        losses=losses,
        open_trades=open_trades,
        win_rate=win_rate,
        net_r_multiple=net_r,
        avg_r_per_trade=avg_r_per_trade,
        expectancy_r=expectancy_r,
        avg_risk_reward=avg_risk_reward,
        profit_factor=profit_factor,
        max_drawdown_r=max_drawdown,
        trades_per_day=trades_per_day,
        csv_path=csv_path,
    )


async def run_backtest_many(
    fetcher: BinanceFetcher,
    settings: Settings,
    pairs: List[str],
    bars: int,
    lookahead_bars: int,
    export_csv_dir: Optional[str] = None,
    export_summary_csv: Optional[str] = None,
) -> List[BacktestResult]:
    results: List[BacktestResult] = []
    for pair in pairs:
        export_path = None
        if export_csv_dir:
            export_path = str(Path(export_csv_dir) / f"backtest_{pair}.csv")

        result = await run_backtest(
            fetcher,
            settings,
            pair,
            bars,
            lookahead_bars,
            export_csv_path=export_path,
        )
        results.append(result)

    if export_summary_csv:
        _write_summary_csv(export_summary_csv, results)

    return results


def _write_summary_csv(csv_path: str, results: List[BacktestResult]) -> str:
    path = Path(csv_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair",
                "total_trades",
                "wins",
                "losses",
                "open_trades",
                "win_rate",
                "net_r_multiple",
                "avg_r_per_trade",
                "expectancy_r",
                "avg_risk_reward",
                "profit_factor",
                "max_drawdown_r",
                "trades_per_day",
                "trade_csv_path",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r.pair,
                    r.total_trades,
                    r.wins,
                    r.losses,
                    r.open_trades,
                    f"{r.win_rate:.2f}",
                    f"{r.net_r_multiple:.4f}",
                    f"{r.avg_r_per_trade:.4f}",
                    f"{r.expectancy_r:.4f}",
                    f"{r.avg_risk_reward:.4f}",
                    f"{r.profit_factor:.4f}",
                    f"{r.max_drawdown_r:.4f}",
                    f"{r.trades_per_day:.4f}",
                    r.csv_path,
                ]
            )

    return str(path)
