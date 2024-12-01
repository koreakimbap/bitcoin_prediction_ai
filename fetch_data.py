# fetch_data.py
import ccxt
import pandas as pd
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD

def fetch_futures_data_with_indicators(symbol='BTC/USDT', timeframe='15m', limit=1000, total_candles=182000):
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })

    all_data = []
    since = 0  # Start fetching data from the beginning (can be adjusted to a specific date)

    while len(all_data) < total_candles:
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Update the starting point for the next fetch
            time.sleep(1)  # Prevent hitting the rate limit
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)  # Retry after a short delay

    # Convert fetched data to a DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]  # Reorder columns

    # Add RSI (Relative Strength Index)
    rsi = RSIIndicator(close=df['close'], window=14)
    df['RSI'] = rsi.rsi()

    # Add MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    return df
