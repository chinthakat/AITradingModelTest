import time
import pandas as pd
from binance.client import Client
from stable_baselines3 import PPO
from trading_features import add_indicators, build_observation
from collections import deque

def get_binance_client():
    return Client()

client = get_binance_client()

# --- Get previous 2 hours (120 minutes) of 1m candles for indicator warm-up ---
candles = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=120)
rows = []
for candle in candles:
    ts = pd.to_datetime(candle[0], unit='ms')
    price = float(candle[4])
    row = {
        "open": float(candle[1]),
        "high": float(candle[2]),
        "low": float(candle[3]),
        "close": price,
        "volume": float(candle[5]),
        "Close": price
    }
    rows.append((ts, row))
df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])

# Calculate indicators and drop NaN rows
df = add_indicators(df)
df = df.dropna()
print(f"Initial DataFrame after indicator calculation: {df.shape}")
print(df.tail(3))

balance = 10_000
btc_held = 0
last_trade_profits = deque([0.0] * 5, maxlen=5)

model = PPO.load("ppo_btc_trend")

while True:
    try:
        # 1. Get the latest 1m candle
        candles = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)
        if not candles:
            print("No candle data received.")
            time.sleep(60)
            continue

        candle = candles[0]
        ts = pd.to_datetime(candle[0], unit='ms')
        price = float(candle[4])
        new_row = {
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": price,
            "volume": float(candle[5]),
            "Close": price
        }
        # Log all data retrieved from Binance
        print(f"Retrieved candle from Binance: {new_row} at {ts}")

        # 2. Append new row to DataFrame
        df.loc[ts] = new_row
        df = df.tail(120)  # Keep last 2 hours for efficiency

        # 3. Update indicators and drop NaN rows
        df = add_indicators(df)

        current_step = len(df) - 1

        # Only use the latest row if it has no NaNs in required columns
        if df.iloc[current_step].isnull().any():
            print("Waiting for enough data to compute indicators (latest row has NaNs)...")
            time.sleep(60)
            continue

        # 4. Build observation using the latest valid row
        obs = build_observation(df, current_step, balance, btc_held, last_trade_profits)

        # 5. Model prediction
        action, _ = model.predict(obs)

        # 6. Execute action logic and trace details
        action_type = "HOLD"
        trade_amount = 0
        profit_loss = 0
        prev_balance = balance
        prev_btc_held = btc_held
        prev_net_worth = balance + btc_held * price

        # Example action mapping (adjust to your environment's action space)
        # 0: Sell all, 1: Hold, 2+: Buy fractions
        if action == 0:
            if btc_held > 0:
                action_type = "SELL"
                trade_amount = btc_held
                balance += btc_held * price
                profit_loss = (price - df.iloc[current_step-1]['Close']) * btc_held
                btc_held = 0
                last_trade_profits.append(profit_loss)
            else:
                action_type = "SELL (no BTC)"
        elif action >= 1:
            # Example: buy 1% of balance for action==1, 5% for action==2, etc.
            buy_fractions = [0, 0.01, 0.05, 0.10, 0.25, 0.50]
            buy_fraction = buy_fractions[action] if action < len(buy_fractions) else 0.0
            max_buy = balance * buy_fraction
            btc_to_buy = max_buy / price if price > 0 else 0
            min_trade_btc = 0.00001
            if btc_to_buy >= min_trade_btc and max_buy > 0:
                action_type = f"BUY {buy_fraction*100:.1f}%"
                trade_amount = btc_to_buy
                balance -= max_buy
                btc_held += btc_to_buy
                profit_loss = 0  # Not realized yet
                last_trade_profits.append(profit_loss)
            else:
                action_type = "BUY (insufficient funds)"

        net_worth = balance + btc_held * price
        print(
            f"Time: {ts}, Action: {action} ({action_type}), "
            f"Trade Amount: {trade_amount:.6f} BTC, "
            f"Price: {price:.2f}, "
            f"Balance: {balance:.2f}, BTC Held: {btc_held:.6f}, "
            f"Net Worth: {net_worth:.2f}, "
            f"Profit/Loss: {profit_loss:.2f}"
        )

        # 7. Wait for the next minute
        time.sleep(60)

    except Exception as e:
        print(f"Error occurred: {e}. Reconnecting to Binance in 10 seconds...")
        time.sleep(10)

        client = get_binance_client()