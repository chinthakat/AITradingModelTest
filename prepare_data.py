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

def get_binance_1m_data(
    symbol="BTCUSDT",
    train_from_date=None,  # format: "YYYYMMDD"
    train_to_date=None,    # format: "YYYYMMDD"
    val_from_date=None,    # format: "YYYYMMDD"
    val_to_date=None,      # format: "YYYYMMDD"
    csv_path="data/binance_btc_1m.csv",
    val_csv_path="data/binance_btc_1m_val.csv"
):
    def fetch_binance_data(start_time, end_time):
        client = Client()
        limit = 1000
        interval = Client.KLINE_INTERVAL_1MINUTE
        chunk_days = 60
        chunk_start = start_time
        all_chunks = []
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
                start_ts = last_open_time + 60_000
                time.sleep(0.5)
            if chunk_data:
                df_chunk = pd.DataFrame(chunk_data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
                ])
                df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="ms")
                df_chunk.set_index("timestamp", inplace=True)
                df_chunk = df_chunk[["open", "high", "low", "close", "volume"]].astype(float)
                df_chunk['Close'] = df_chunk['close']
                df_chunk['SMA_10'] = df_chunk['Close'].rolling(window=10).mean()
                df_chunk['SMA_50'] = df_chunk['Close'].rolling(window=50).mean()
                df_chunk['RSI'] = ta.momentum.RSIIndicator(close=df_chunk['Close'], window=14).rsi()
                macd = ta.trend.MACD(close=df_chunk['Close'])
                df_chunk['MACD'] = macd.macd()
                df_chunk['MACD_Signal'] = macd.macd_signal()
                df_chunk['MACD_Hist'] = macd.macd_diff()
                bb = ta.volatility.BollingerBands(close=df_chunk['Close'], window=20, window_dev=2)
                df_chunk['BB_MID'] = bb.bollinger_mavg()
                df_chunk['BB_HIGH'] = bb.bollinger_hband()
                df_chunk['BB_LOW'] = bb.bollinger_lband()
                df_chunk = df_chunk.dropna()
                all_chunks.append(df_chunk)
                print(f"Fetched and processed chunk ({len(df_chunk)} rows)")
            chunk_start = chunk_end
        if all_chunks:
            all_data = pd.concat(all_chunks)
            all_data = all_data.sort_index()
            return all_data
        else:
            return pd.DataFrame()

    # Remove old files if they exist
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(val_csv_path):
        os.remove(val_csv_path)

    # Parse dates
    def parse_date(date_str, default):
        return datetime.datetime.strptime(date_str, "%Y%m%d") if date_str else default

    train_df = pd.DataFrame()
    if train_from_date and train_to_date:
        train_start = parse_date(train_from_date, datetime.datetime.now() - datetime.timedelta(days=365))
        train_end = parse_date(train_to_date, datetime.datetime.now() - datetime.timedelta(days=60))
        train_df = fetch_binance_data(train_start, train_end)
        train_df.to_csv(csv_path)
        print(f"Saved training data to {csv_path} ({len(train_df)} rows)")
    else:
        print("Training date range not provided, skipping training data fetch.")

    val_start = parse_date(val_from_date, datetime.datetime.now() - datetime.timedelta(days=60))
    val_end = parse_date(val_to_date, datetime.datetime.now())
    val_df = fetch_binance_data(val_start, val_end)
    val_df.to_csv(val_csv_path)
    print(f"Saved validation data to {val_csv_path} ({len(val_df)} rows)")

    return train_df, val_df
