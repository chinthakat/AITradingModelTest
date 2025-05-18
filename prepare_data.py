import yfinance as yf
import ta
import pandas as pd
from binance.client import Client
import datetime
import time
import os
import shutil

def get_btc_data():
    df = yf.download('BTC-USD',
                    period='60d',  # Yahoo only allows 15m data for up to 60 days
                    interval='15m')
    # Fix: Ensure columns are properly named and handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    # If 'Close' not in columns, try to find it or print columns for debug
    if 'Close' not in df.columns:
        print(f"DEBUG: Columns in df: {df.columns}")
        # Try common alternatives
        for col in df.columns:
            if 'close' in col.lower():
                df['Close'] = df[col]
                break
        else:
            raise KeyError("'Close' column not found in downloaded data.")
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    return df.dropna()

def get_binance_1m_data(symbol="BTCUSDT", days=60, csv_path="binance_btc_1m.csv"): 
    client = Client()  # No API key required for public data

    # Binance allows only ~1,000 candles per request; 1 min * 1000 = ~16.6 hours
    limit = 1000
    interval = Client.KLINE_INTERVAL_1MINUTE
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days)

    data = []
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    print("Fetching data from Binance...")

    while start_ts < end_ts:
        candles = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            limit=limit
        )
        if not candles:
            break
        data.extend(candles)
        last_open_time = candles[-1][0]
        start_ts = last_open_time + 60_000  # move to next minute
        time.sleep(0.5)  # respect API rate limits

    print(f"Retrieved {len(data)} candles.")
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    # Add required features for RL environment
    df['Close'] = df['close']  # Capital C for compatibility
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df = df.dropna()
    # Archive existing file if it exists
    if os.path.exists(csv_path):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"{csv_path.rstrip('.csv')}_archive_{now}.csv"
        shutil.move(csv_path, archive_path)
        print(f"Archived old CSV to {archive_path}")
    df.to_csv(csv_path)
    print(f"Saved new data to {csv_path}")
    return df
