import asyncio
import argparse
import logging
from typing import List, Optional

from backtester import run_backtest, run_backtest_many
from config import settings
from data_fetcher import BinanceFetcher
from economic_events import EconomicCalendarChecker, EconomicEventStatus
from optimizer import conditional_apply_best, optimize_parameters
from signal_store import SignalStore
from strategy import Signal, build_telegram_message, evaluate_pair
from telegram_notifier import TelegramNotifier


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("signal_bot")


async def scan_once(
    fetcher: BinanceFetcher,
    notifier: TelegramNotifier,
    store: SignalStore,
    econ_checker: Optional[EconomicCalendarChecker] = None,
) -> None:
    fetcher.start_cycle_cache()

    # Check for upcoming high-impact economic events once per cycle.
    econ_status = EconomicEventStatus(caution_mode=False)
    if econ_checker is not None:
        try:
            econ_status = await econ_checker.check_high_impact_today()
            if econ_status.caution_mode:
                LOGGER.warning(
                    "CAUTION MODE active: %s | Events: %s",
                    econ_status.reason,
                    ", ".join(econ_status.upcoming_events),
                )
        except Exception as exc:
            LOGGER.warning("Economic calendar check failed, continuing without caution mode: %s", exc)

    ordered_pairs = settings.scan_pairs
    if settings.prioritize_high_volume:
        try:
            limit = settings.high_volume_scan_limit or len(settings.scan_pairs)
            ordered_pairs = await fetcher.rank_symbols_by_24h_volume(settings.scan_pairs, limit)
            LOGGER.info("Volume-priority order: %s", ",".join(ordered_pairs))
        except Exception as exc:
            LOGGER.warning("24h volume ranking failed, using default order: %s", exc)

    # Fetch Coinglass liquidation heatmaps once per scan cycle (batch).
    await fetcher.fetch_coinglass_heatmaps(ordered_pairs)

    snapshots: List[dict | Exception | None] = []
    sem = asyncio.Semaphore(settings.max_concurrent_fetches)

    async def _fetch_one(pair: str) -> dict | Exception | None:
        async with sem:
            try:
                return await fetcher.fetch_pair_snapshot(
                    symbol=pair,
                    interval=settings.kline_interval,
                    limit=settings.klines_limit,
                    higher_tf_interval=settings.higher_tf_interval,
                    higher_tf_limit=settings.klines_limit,
                )
            except Exception as exc:
                return exc

    snapshots = list(await asyncio.gather(*[_fetch_one(p) for p in ordered_pairs]))

    # --- BTC Sentinel: find BTC snapshot to pass to all non-BTC pairs ---
    btc_snap = None
    snap_by_pair = {}
    for pair, snap in zip(ordered_pairs, snapshots):
        if isinstance(snap, dict):
            snap_by_pair[pair] = snap
            if pair == "BTCUSDT":
                btc_snap = snap

    scan_errors = 0
    candidates: List[Signal] = []
    for pair, snap in zip(ordered_pairs, snapshots):
        if isinstance(snap, Exception):
            LOGGER.warning("Snapshot error for %s: %s", pair, snap)
            scan_errors += 1
            continue
        if not snap:
            scan_errors += 1
            continue

        # Inject economic caution mode into snapshot so strategy can scale position size.
        snap["caution_mode"] = econ_status.caution_mode
        snap["caution_reason"] = econ_status.reason or "High-impact economic event soon"

        oi_value = float(snap.get("open_interest", 0.0))
        funding_value = float(snap.get("funding_rate", 0.0))
        if settings.require_non_zero_derivatives and (oi_value == 0.0 or funding_value == 0.0):
            LOGGER.warning(
                "Skipped %s due to invalid derivatives data (oi=%s funding=%s)",
                pair,
                oi_value,
                funding_value,
            )
            continue

        debug_msgs: List[str] = []
        signal = evaluate_pair(snap, settings, debug_msgs, btc_snapshot=btc_snap if pair != "BTCUSDT" else None)
        if signal:
            candidates.append(signal)

        # Send intelligent debug alerts to Telegram (score >= 55 only).
        for dbg_msg in debug_msgs:
            try:
                await notifier.send_signal(dbg_msg)
            except Exception as exc:
                LOGGER.warning("Debug notification failed for %s: %s", pair, exc)

    # Per-cycle scan summary — visible at INFO level so operators can see the bot is alive.
    coinalyze_mode = "ON" if any(
        isinstance(s, dict) and s.get("coinalyze_available") for s in snapshots
    ) else "OFF"
    LOGGER.info(
        "Cycle summary: pairs_scanned=%s errors=%s signals_found=%s caution=%s coinalyze=%s",
        len(ordered_pairs), scan_errors, len(candidates),
        econ_status.caution_mode, coinalyze_mode,
    )

    if not candidates:
        LOGGER.info(
            "No high-confidence setup this cycle. "
            "Check 'Signal Rejected' lines above for per-pair reasons."
        )
        return

    ranked = sorted(candidates, key=lambda s: s.confidence, reverse=True)
    max_cycle = max(1, min(settings.max_signals_per_cycle, 2))
    selected = ranked[:max_cycle]

    sent_count = 0
    for best in selected:
        if not store.can_send_today(settings.max_signals_per_day):
            LOGGER.info("Daily signal limit reached (%s)", settings.max_signals_per_day)
            break

        if store.is_duplicate(best):
            LOGGER.info("Duplicate setup skipped: %s %s", best.pair, best.side)
            continue

        try:
            message = build_telegram_message(best)
            await notifier.send_signal(message, confidence=best.confidence)
            store.save(best)
            sent_count += 1
            LOGGER.info(
                "Signal sent: %s %s conf=%.1f%% sent_today=%s",
                best.pair,
                best.side,
                best.confidence,
                store.count_today(),
            )
        except Exception:
            LOGGER.warning("Signal not saved — Telegram delivery failed for %s", best.pair)
            continue

    if sent_count == 0:
        LOGGER.info("No signal passed delivery checks this cycle")


async def run_loop() -> None:
    settings.validate()

    fetcher = BinanceFetcher(
        timeout_seconds=10,
        coinalyze_api_key=settings.coinalyze_api_key,
        coinalyze_cooldown_seconds=settings.coinalyze_cooldown_seconds,
    )
    notifier = TelegramNotifier(settings.telegram_token, settings.telegram_chat_id)
    store = SignalStore(settings.sqlite_db_path)
    econ_checker = EconomicCalendarChecker(
        lookahead_hours=settings.economic_event_lookahead_hours
    )

    LOGGER.info(
        "Starting bot: pairs=%s interval=%ss min_conf=%.1f max/day=%s",
        ",".join(settings.scan_pairs),
        settings.scan_interval_seconds,
        settings.min_confidence,
        settings.max_signals_per_day,
    )

    try:
        await notifier.send_startup_notification()
    except Exception:
        LOGGER.warning("Startup notification failed")

    while True:
        try:
            await scan_once(fetcher, notifier, store, econ_checker)
        except Exception as exc:
            LOGGER.exception("Scan loop error: %s", exc)

        await asyncio.sleep(settings.scan_interval_seconds)


async def run_backtest_mode(
    pair: str,
    all_pairs: bool,
    export_csv: str,
    export_csv_dir: str,
    export_summary_csv: str,
) -> None:
    fetcher = BinanceFetcher(
        timeout_seconds=10,
        coinalyze_api_key=settings.coinalyze_api_key,
        coinalyze_cooldown_seconds=settings.coinalyze_cooldown_seconds,
    )
    if all_pairs:
        results = await run_backtest_many(
            fetcher=fetcher,
            settings=settings,
            pairs=settings.scan_pairs,
            bars=settings.backtest_bars,
            lookahead_bars=settings.backtest_lookahead_bars,
            export_csv_dir=export_csv_dir,
            export_summary_csv=export_summary_csv,
        )
        for result in results:
            LOGGER.info(
                "Backtest %s | trades=%s wins=%s losses=%s open=%s win_rate=%.2f%% netR=%.2f avgR=%.3f expR=%.3f PF=%.2f maxDD_R=%.2f trades/day=%.2f csv=%s",
                result.pair,
                result.total_trades,
                result.wins,
                result.losses,
                result.open_trades,
                result.win_rate,
                result.net_r_multiple,
                result.avg_r_per_trade,
                result.expectancy_r,
                result.profit_factor,
                result.max_drawdown_r,
                result.trades_per_day,
                result.csv_path or "-",
            )
        return

    result = await run_backtest(
        fetcher=fetcher,
        settings=settings,
        pair=pair,
        bars=settings.backtest_bars,
        lookahead_bars=settings.backtest_lookahead_bars,
        export_csv_path=export_csv,
    )

    LOGGER.info(
        "Backtest %s | trades=%s wins=%s losses=%s open=%s win_rate=%.2f%% netR=%.2f avgR=%.3f expR=%.3f PF=%.2f maxDD_R=%.2f trades/day=%.2f csv=%s",
        result.pair,
        result.total_trades,
        result.wins,
        result.losses,
        result.open_trades,
        result.win_rate,
        result.net_r_multiple,
        result.avg_r_per_trade,
        result.expectancy_r,
        result.profit_factor,
        result.max_drawdown_r,
        result.trades_per_day,
        result.csv_path or "-",
    )


async def run_optimize_mode(top_n: int, export_csv: str, auto_apply_best: bool, apply_env_path: str) -> None:
    fetcher = BinanceFetcher(
        timeout_seconds=10,
        coinalyze_api_key=settings.coinalyze_api_key,
        coinalyze_cooldown_seconds=settings.coinalyze_cooldown_seconds,
    )
    ranked = await optimize_parameters(
        fetcher=fetcher,
        settings=settings,
        pairs=settings.scan_pairs,
        bars=settings.optimize_bars,
        lookahead_bars=settings.optimize_lookahead_bars,
        top_n=top_n,
        export_csv_path=export_csv,
    )

    if not ranked:
        LOGGER.info("Optimization finished with no valid parameter sets")
        return

    LOGGER.info("Optimization top=%s csv=%s", len(ranked), export_csv or "-")
    for idx, row in enumerate(ranked, start=1):
        LOGGER.info(
            "Rank %s | trades=%s win_rate=%.2f%% netR=%.2f expR=%.3f maxDD_R=%.2f params=%s",
            idx,
            row.total_trades,
            row.win_rate,
            row.net_r_multiple,
            row.expectancy_r,
            row.max_drawdown_r,
            row.params,
        )

    if auto_apply_best:
        applied, message = conditional_apply_best(ranked, apply_env_path)
        if applied:
            LOGGER.info("Auto-apply status: %s", message)
        else:
            LOGGER.info("Auto-apply skipped: %s", message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professional crypto signal bot")
    parser.add_argument("--mode", choices=["live", "backtest", "optimize"], default="live")
    parser.add_argument("--pair", default=settings.backtest_pair)
    parser.add_argument("--all-pairs", action="store_true", default=settings.backtest_all_pairs)
    parser.add_argument("--export-csv", default="")
    parser.add_argument("--export-csv-dir", default="")
    parser.add_argument("--export-summary-csv", default="")
    parser.add_argument("--opt-top-n", type=int, default=settings.optimize_top_n)
    parser.add_argument("--opt-export-csv", default=settings.optimize_export_csv)
    parser.add_argument(
        "--opt-auto-apply-best",
        action="store_true",
        default=settings.optimize_auto_apply_best,
    )
    parser.add_argument("--opt-apply-env-path", default=settings.optimize_apply_env_path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.mode == "backtest":
            asyncio.run(
                run_backtest_mode(
                    args.pair,
                    args.all_pairs,
                    args.export_csv,
                    args.export_csv_dir,
                    args.export_summary_csv,
                )
            )
        elif args.mode == "optimize":
            asyncio.run(
                run_optimize_mode(
                    args.opt_top_n,
                    args.opt_export_csv,
                    args.opt_auto_apply_best,
                    args.opt_apply_env_path,
                )
            )
        else:
            asyncio.run(run_loop())
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user")
