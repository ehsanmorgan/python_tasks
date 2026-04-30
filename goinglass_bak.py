import os
import asyncio
import requests
from telegram.ext import ApplicationBuilder
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
COINGLASS_API_KEY = os.getenv('COINGLASS_API_KEY')
if not TELEGRAM_TOKEN or not CHAT_ID or not COINGLASS_API_KEY:
    raise SystemExit('Environment variables TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, and COINGLASS_API_KEY must be set')
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
# Coin list
TOP_40_COINS = [
    'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'SUI', 'RAVE', 'HYPE', 'ZEC'
]

COINGLASS_BASE_URL = 'https://open-api-v4.coinglass.com'
async def scrape_liquidation_heatmaps():
    """
    Get Liquidation Heatmaps data from Coinglass API.
    Returns dict of coin: {'high_density_zones': [prices], 'clusters': [prices]}
    """
    liquidation_data = {}
    for coin in TOP_40_COINS:
        try:
            # Use Coinglass API to get liquidation heatmap data
            url = f'{COINGLASS_BASE_URL}/public/v2/liquidation_heatmap'
            params = {'symbol': f'{coin}USDT'}
            headers = {'coinglassSecret': COINGLASS_API_KEY}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                heatmap_data = data.get('data', {})
                high_density_zones = heatmap_data.get('high_density_zones', [])
                clusters = heatmap_data.get('clusters', [])
                liquidation_data[coin] = {
                    'high_density_zones': high_density_zones,
                    'clusters': clusters
                }
            else:
                # Fallback with placeholder data
                liquidation_data[coin] = {
                    'high_density_zones': [50000, 51000],
                    'clusters': [48000, 49000]
                }
        except Exception as e:
            print(f"Error fetching liquidation data for {coin}: {e}")
            liquidation_data[coin] = {
                'high_density_zones': [50000, 51000],
                'clusters': [48000, 49000]
            }
    return liquidation_data
async def scrape_whale_movements():
    """
    Scrape Whale movements data.
    Placeholder implementation.
    """
    # Similar to above, scrape whale data
    whale_data = {}
    for coin in TOP_40_COINS:
        whale_data[coin] = {'large_positions': []}  # placeholder
    return whale_data
async def scrape_netflow():
    """
    Scrape Netflow data (inflows/outflows from exchanges).
    Returns dict of coin: netflow_value (positive = outflows, negative = inflows)
    """
    # Placeholder - in real implementation, scrape from Coinglass
    netflow_data = {}
    for coin in TOP_40_COINS:
        netflow_data[coin] = np.random.uniform(-1000000, 1000000)  # random for demo
    return netflow_data
def get_api_data(coin):
    """
    Get standard metrics from Coinglass API: OI, Funding Rate, Long/Short Ratio
    """
    headers = {'coinglassSecret': COINGLASS_API_KEY}  # Assuming this is the header
    # Open Interest
    oi_url = f'{COINGLASS_BASE_URL}/public/v2/open_interest'
    oi_params = {'symbol': f'{coin}USDT', 'interval': '1h', 'limit': 1}
    oi_response = requests.get(oi_url, headers=headers, params=oi_params)
    oi_data = oi_response.json() if oi_response.status_code == 200 else {}
    # Funding Rate
    fr_url = f'{COINGLASS_BASE_URL}/public/v2/funding_rate'
    fr_params = {'symbol': f'{coin}USDT', 'interval': '1h', 'limit': 1}
    fr_response = requests.get(fr_url, headers=headers, params=fr_params)
    fr_data = fr_response.json() if fr_response.status_code == 200 else {}
    # Long/Short Ratio
    ls_url = f'{COINGLASS_BASE_URL}/public/v2/long_short_ratio'
    ls_params = {'symbol': f'{coin}USDT', 'interval': '1h', 'limit': 1}
    ls_response = requests.get(ls_url, headers=headers, params=ls_params)
    ls_data = ls_response.json() if ls_response.status_code == 200 else {}
    return {
        'oi': oi_data.get('data', [{}])[0] if oi_data.get('data') else {},
        'funding_rate': fr_data.get('data', [{}])[0] if fr_data.get('data') else {},
        'ls_ratio': ls_data.get('data', [{}])[0] if ls_data.get('data') else {}
    }
def get_price_data(coin):
    """
    Get recent price data for ATR calculation from Binance API.
    Returns list of dicts with 'high', 'low', 'close', 'volume'.
    """
    symbol = f'{coin}USDT'
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': '1h',
        'limit': 200  # More data for EMA 200 and support/resistance
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        prices = []
        for kline in data:
            # Kline format: [open_time, open, high, low, close, volume, ...]
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            volume = float(kline[5])
            prices.append({'high': high, 'low': low, 'close': close, 'volume': volume})
        return prices
    except requests.RequestException as e:
        print(f"Error fetching price data for {coin}: {e}")
        # Fallback to sample data if API fails
        base_price = 50000 if coin == 'BTC' else 3000
        prices = []
        for i in range(200):
            high = base_price * (1 + np.random.uniform(-0.05, 0.05))
            low = base_price * (1 + np.random.uniform(-0.05, 0.05))
            close = base_price * (1 + np.random.uniform(-0.05, 0.05))
            volume = np.random.uniform(100000, 1000000)
            prices.append({'high': high, 'low': low, 'close': close, 'volume': volume})
            base_price = close
        return prices
def calculate_atr(prices):
    """
    Calculate Average True Range from price data.
    """
    df = pd.DataFrame(prices)
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    return atr_indicator.average_true_range().iloc[-1]
def calculate_support_resistance(prices):
    """
    Calculate simple support and resistance levels from the last 14 candles.
    """
    recent_prices = prices[-14:]
    highs = [p['high'] for p in recent_prices]
    lows = [p['low'] for p in recent_prices]
    support = min(lows)
    resistance = max(highs)
    return support, resistance
def calculate_rsi(prices):
    """
    Calculate RSI from price data.
    """
    df = pd.DataFrame(prices)
    rsi_indicator = RSIIndicator(close=df['close'])
    return rsi_indicator.rsi().iloc[-1]
def calculate_ema_200(prices):
    """
    Calculate EMA 200 from price data.
    """
    df = pd.DataFrame(prices)
    ema_indicator = EMAIndicator(close=df['close'], window=200)
    return ema_indicator.ema_indicator().iloc[-1]
def get_market_context():
    """
    Get market context: USDT Dominance or TOTAL3 (Market Cap) from CoinGlass or Binance.
    """
    try:
        # Try CoinGlass API for market data
        url = f'{COINGLASS_BASE_URL}/public/v2/market_stats'
        headers = {'coinglassSecret': COINGLASS_API_KEY}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Assuming data has 'usdt_dominance' or similar
            usdt_dominance = data.get('data', {}).get('usdt_dominance', 0)
            return usdt_dominance
        else:
            # Fallback to Binance for BTC dominance or similar
            # For simplicity, return a placeholder
            return 0.05  # 5% as example
    except Exception as e:
        print(f"Error fetching market context: {e}")
        return 0.05  # Default
def analyze_signals(coin, liquidation_data, netflow, api_data, prices):
    """
    Analyze data for trading signals based on enhanced criteria.
    Returns list of signals with strength score.
    """
    signals = []
    current_price = prices[-1]['close']
    atr = calculate_atr(prices)
    support, resistance = calculate_support_resistance(prices)
    rsi = calculate_rsi(prices)
    ema_200 = calculate_ema_200(prices)
    market_context = get_market_context()
    # Volume check: current volume above the 20-period average
    current_volume = prices[-1]['volume']
    avg_volume = np.mean([p['volume'] for p in prices[-20:]])
    high_volume = current_volume > avg_volume
    oi_value = api_data['oi'].get('openInterest', 0)
    funding_rate = api_data['funding_rate'].get('fundingRate', 0)
    ls_ratio = api_data['ls_ratio'].get('longShortRatio', 1)
    high_density_zones = liquidation_data.get('high_density_zones', [])
    clusters = liquidation_data.get('clusters', [])
    # Long Signal
    approaching_high_density = any(abs(current_price - zone) / current_price < 0.02 for zone in high_density_zones)
    positive_netflow = netflow > 0
    oi_rising = oi_value > 0
    neutral_funding = funding_rate <= 0.01
    above_support = current_price > support
    rsi_ok_long = rsi < 70
    above_ema = current_price > ema_200
    market_bullish = market_context < 0.1  # Example: low USDT dominance means bullish
    long_score = sum([
        approaching_high_density,
        positive_netflow,
        oi_rising,
        neutral_funding,
        high_volume,
        above_support,
        rsi_ok_long,
        above_ema,
        market_bullish
    ])
    if long_score >= 7:  # Strong
        strength = 'Strong'
    elif long_score >= 5:  # Medium
        strength = 'Medium'
    else:
        strength = 'Weak'
    if strength in ['Strong', 'Medium']:
        entry = current_price
        tp = entry + 2 * atr
        sl = entry - atr
        signals.append({
            'direction': 'LONG',
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'strength': strength,
            'reason': f'OI: {oi_value:.2f}, Netflow: {netflow:.2f}, Funding: {funding_rate:.4f}, RSI: {rsi:.2f}, EMA200: {ema_200:.2f}'
        })
    # Short Signal
    approaching_cluster = any(abs(current_price - cluster) / current_price < 0.02 for cluster in clusters)
    negative_netflow = netflow < 0
    high_ls_ratio = ls_ratio > 1.2
    below_resistance = current_price < resistance
    rsi_ok_short = rsi > 30
    below_ema = current_price < ema_200
    market_bearish = market_context > 0.1
    short_score = sum([
        approaching_cluster, negative_netflow, high_ls_ratio,
        high_volume, below_resistance, rsi_ok_short, below_ema, market_bearish
    ])
    if short_score >= 7:
        strength = 'Strong'
    elif short_score >= 5:
        strength = 'Medium'
    else:
        strength = 'Weak'
    if strength in ['Strong', 'Medium']:
        entry = current_price
        tp = entry - 2 * atr
        sl = entry + atr
        signals.append({
            'direction': 'SHORT',
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'strength': strength,
            'reason': f'LS Ratio: {ls_ratio:.2f}, Netflow: {netflow:.2f}, RSI: {rsi:.2f}, EMA200: {ema_200:.2f}'
        })
    return signals
async def send_telegram_alert(coin, signal):
    """
    Send alert to Telegram using the configured application bot.
    """
    message = (
        f"🚀 SIGNAL ALERT [{signal['strength']}]\n"
        f"Coin: {coin} | Direction: {signal['direction']}\n"
        f"Entry: {signal['entry']:.2f} | TP: {signal['tp']:.2f} | SL: {signal['sl']:.2f}\n"
        f"Reasoning: {signal['reason']}"
    )
    print(f"Sending Telegram alert for {coin}: {signal['direction']} ({signal['strength']})")
    try:
        result = await app.bot.send_message(chat_id=CHAT_ID, text=message)
        print(f"Telegram alert sent, message id: {result.message_id}")
    except Exception as e:
        print(f"Error sending Telegram alert for {coin}: {e}")
async def main():
    """
    Main monitoring loop.
    """
    # Initialize Telegram application and verify bot credentials.
    try:
        await app.initialize()
        await app.start()
        bot_info = await app.bot.get_me()
        print(f"Telegram bot initialized: @{bot_info.username} (id={bot_info.id})")
        await app.bot.send_message(chat_id=CHAT_ID, text="🚀 Trading Bot Started! Monitoring Top 40 cryptocurrencies for signals.")
        print("Startup Telegram message sent successfully")
    except Exception as e:
        print(f"Error initializing Telegram bot or sending startup message: {e}")
        return
    try:
        while True:
            try:
                # Scrape advanced data
                liquidation_data = await scrape_liquidation_heatmaps()
                whale_data = await scrape_whale_movements()
                netflow_data = await scrape_netflow()
                for coin in TOP_40_COINS:
                    # Get API data
                    api_data = await asyncio.to_thread(get_api_data, coin)
                    # Get price data for ATR and other indicators
                    prices = await asyncio.to_thread(get_price_data, coin)
                    if not prices:
                        print(f"{coin}: no price data, skipping")
                        continue
                    # Analyze signals
                    signals = analyze_signals(coin, liquidation_data[coin], netflow_data[coin], api_data, prices)
                    if signals:
                        print(f"{coin}: {len(signals)} signal(s) found")
                    else:
                        print(f"{coin}: no signal")
                    # Send alerts
                    for signal in signals:
                        await send_telegram_alert(coin, signal)
                # Wait before next check (e.g., 1 hour)
                await asyncio.sleep(3600)
            except Exception as e:
                print(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    finally:
        await app.stop()
        await app.shutdown()
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البوت.")

    
    
    
