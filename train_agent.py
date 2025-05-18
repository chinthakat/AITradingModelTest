from stable_baselines3 import PPO
from btc_trading_env import BTCTradingEnv
from prepare_data import get_binance_1m_data
import pandas as pd
import os
import datetime

# Archive existing trading_log.txt if it exists
log_path = "trading_log.txt"
if os.path.exists(log_path):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"trading_log_{now}.txt"
    os.rename(log_path, archive_path)
    print(f"Archived old log file to {archive_path}")

# Use Binance 1-minute data for training
csv_path = "data/binance_btc_1m.csv"
try:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"Loaded data from {csv_path}")
except FileNotFoundError:
    df = get_binance_1m_data(symbol="BTCUSDT", months=12, csv_path=csv_path)
    df.to_csv(csv_path)
    print(f"Downloaded and saved data to {csv_path}")

env = BTCTradingEnv(df)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=150_000)
model.save("ppo_btc_trend")
