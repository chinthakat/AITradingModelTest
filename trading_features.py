import numpy as np
import pandas as pd
import ta

def add_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    return df

def get_scalar(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    if isinstance(val, np.ndarray):
        return float(val.item())
    return float(val) if pd.notnull(val) else 0.0

def build_observation(df, current_step, balance, btc_held, last_trade_profits):
    # Ensure last_trade_profits is always length 5
    last_trade_profits = list(last_trade_profits)[-5:]
    while len(last_trade_profits) < 5:
        last_trade_profits.insert(0, 0.0)

    row = df.iloc[current_step]
    sma_10_vals = [
        get_scalar(df.iloc[max(0, current_step - i)]['SMA_10'])
        for i in range(5)
    ]
    sma_50_vals = [
        get_scalar(df.iloc[max(0, current_step - i)]['SMA_50'])
        for i in range(5)
    ]
    rsi_vals = [
        get_scalar(df.iloc[max(0, current_step - i)]['RSI'])
        for i in range(5)
    ]
    macd_vals = [
        get_scalar(df.iloc[max(0, current_step - i)]['MACD'])
        for i in range(5)
    ]
    macd_signal_vals = [
        get_scalar(df.iloc[max(0, current_step - i)]['MACD_Signal'])
        for i in range(5)
    ]
    obs = np.array([
        get_scalar(row['Close']),
        get_scalar(row['SMA_10']),
        get_scalar(row['SMA_50']),
        get_scalar(row['RSI']),
        get_scalar(row['MACD']),
        get_scalar(row['MACD_Signal']),
        float(balance),
        float(btc_held),
        float(current_step),
        *last_trade_profits,
        *sma_10_vals,
        *sma_50_vals,
        *rsi_vals,
        *macd_vals,
        *macd_signal_vals
    ], dtype=np.float32)
    # Ensure the observation is exactly 39 elements
    assert obs.shape == (39,), f"Observation shape is {obs.shape}, expected (39,)"
    return obs