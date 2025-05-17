import yfinance as yf
import pandas as pd
import ta

def get_btc_data():
    df = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df = df.dropna()
    return df
