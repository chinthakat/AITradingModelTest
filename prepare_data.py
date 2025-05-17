import yfinance as yf
import ta
import pandas as pd

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
