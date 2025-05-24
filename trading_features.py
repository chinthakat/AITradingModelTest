import numpy as np
import pandas as pd
import ta

def add_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean() if hasattr(df, 'Close') else 0.0
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean() if hasattr(df, 'Close') else 0.0
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_MID'] = bb.bollinger_mavg()
    df['BB_HIGH'] = bb.bollinger_hband()
    df['BB_LOW'] = bb.bollinger_lband()
    return df

def get_scalar(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    if isinstance(val, np.ndarray):
        return float(val.item())
    return float(val) if pd.notnull(val) else 0.0

def build_observation(df, current_step, balance, btc_held, last_trade_profits):
    last_trade_profits = list(last_trade_profits)[-5:]
    while len(last_trade_profits) < 5:
        last_trade_profits.insert(0, 0.0)

    row = df.iloc[current_step]
    sma_10 = get_scalar(row['SMA_10'])
    sma_50 = get_scalar(row['SMA_50'])
    close = get_scalar(row['close'])

    obs = np.array([
        get_scalar(row['open']),
        get_scalar(row['high']),
        get_scalar(row['low']),
        close,
        sma_10,
        sma_50,
        sma_10 - sma_50,
        close - sma_10,
        close - sma_50,
        get_scalar(row['volume']),
        get_scalar(row['BB_MID']),
        get_scalar(row['BB_HIGH']),
        get_scalar(row['BB_LOW']),
        get_scalar(row['MACD']),
        get_scalar(row['MACD_Signal']),
        get_scalar(row['MACD_Hist']),
        get_scalar(row['RSI']),
        float(balance),
        float(btc_held),
        float(current_step),
        *last_trade_profits,
    ], dtype=np.float32)

    return obs