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

def get_binance_1m_data(symbol="BTCUSDT", months=12, csv_path="data/binance_btc_1m.csv"): 
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    client = Client()  # No API key required for public data
    limit = 1000
    interval = Client.KLINE_INTERVAL_1MINUTE
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=months*30)
    data = []
    chunk_days = 60
    chunk_start = start_time
    # Archive existing file if it exists (should not happen, but for safety)
    if os.path.exists(csv_path):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"{csv_path.rstrip('.csv')}_archive_{now}.csv"
        shutil.move(csv_path, archive_path)
        print(f"Archived old CSV to {archive_path}")
    first_write = True
    while chunk_start < end_time:
        chunk_end = min(chunk_start + datetime.timedelta(days=chunk_days), end_time)
        start_ts = int(chunk_start.timestamp() * 1000)
        end_ts = int(chunk_end.timestamp() * 1000)
        print(f"Fetching data from {chunk_start} to {chunk_end}...")
        chunk_data = []
        while start_ts < end_ts:
            candles = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=limit
            )
            if not candles:
                break
            chunk_data.extend(candles)
            last_open_time = candles[-1][0]
            start_ts = last_open_time + 60_000  # move to next minute
            time.sleep(0.5)  # respect API rate limits
        if chunk_data:
            df_chunk = pd.DataFrame(chunk_data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ])
            df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="ms")
            df_chunk.set_index("timestamp", inplace=True)
            df_chunk = df_chunk[["open", "high", "low", "close", "volume"]].astype(float)
            # Add required features for RL environment
            df_chunk['Close'] = df_chunk['close']  # Capital C for compatibility
            df_chunk['SMA_10'] = df_chunk['Close'].rolling(window=10).mean()
            df_chunk['SMA_50'] = df_chunk['Close'].rolling(window=50).mean()
            df_chunk['RSI'] = ta.momentum.RSIIndicator(close=df_chunk['Close'], window=14).rsi()
            macd = ta.trend.MACD(close=df_chunk['Close'])
            df_chunk['MACD'] = macd.macd()
            df_chunk['MACD_Signal'] = macd.macd_signal()
            df_chunk = df_chunk.dropna()
            df_chunk.to_csv(csv_path, mode='w' if first_write else 'a', header=first_write)
            first_write = False
            print(f"Saved chunk to {csv_path} (rows: {len(df_chunk)})")
        chunk_start = chunk_end
    # Read the full file back in
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"Saved all data to {csv_path}")
    return df
