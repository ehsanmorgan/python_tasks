import asyncio
import logging

from telegram import Bot
from telegram.error import NetworkError, RetryAfter, TimedOut


LOGGER = logging.getLogger(__name__)

_MAX_RETRIES = 3


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id

    async def _send_html(self, text: str) -> None:
        """Deliver a message with HTML parse mode and automatic retry on transient errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                return
            except RetryAfter as exc:
                wait = float(exc.retry_after) + 1.0
                LOGGER.warning(
                    "Telegram flood-control: waiting %.1fs (attempt %s/%s)",
                    wait, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
            except (TimedOut, NetworkError) as exc:
                wait = 2.0 ** attempt
                LOGGER.warning(
                    "Telegram transient error: %s — retry in %.1fs (attempt %s/%s)",
                    exc, wait, attempt + 1, _MAX_RETRIES,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(wait)
                else:
                    LOGGER.error(
                        "Telegram delivery failed after %s retries: %s", _MAX_RETRIES, exc
                    )
                    raise

    async def send_signal(self, message: str, confidence: float | None = None) -> None:
        # message is already HTML-formatted by build_telegram_message; send it directly.
        await self._send_html(message)

    async def send_startup_notification(self) -> None:
        startup_message = (
            "🎯 <b>Quad-TF Precision System Active</b>\n\n"
            "Timeframes: <b>4H → 1H → 15M → 5M</b>\n"
            "Pairs: <b>BTC, ETH, SOL, XRP, SUI, DOGE, BCH, TAO, ZEC, AVAX, LINK, FARTCOIN, 1000PEPE, HYPE</b>\n\n"
            "🅰️ Tier A (≥85): Full signal + Full position\n"
            "🅱️ Tier B (70-84): Full signal + 60% position\n"
            "🔍 Tier C (60-69): Debug alert only\n\n"
            "🛡️ BTC Sentinel: -25 if BTC 1H opposes\n"
            "⚡ 5M Surgical Entry: EMA9 pullback + RSI gate\n"
            "📐 Auto Break-Even at 50% of TP distance"
        )
        await self._send_html(startup_message)








