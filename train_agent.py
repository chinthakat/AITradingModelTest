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

# Use Binance 1-minute data for training and validation
csv_path = "data/binance_btc_1m.csv"
val_csv_path = "data/binance_btc_1m_val.csv"

if os.path.exists(csv_path) and os.path.exists(val_csv_path):
    train_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    val_df = pd.read_csv(val_csv_path, index_col=0, parse_dates=True)
    print(f"Loaded training data from {csv_path}")
    print(f"Loaded validation data from {val_csv_path}")
else:
    train_df, val_df = get_binance_1m_data(
        symbol="BTCUSDT",
        train_from_date="20230501", train_to_date="20240501",
        val_from_date="20240501", val_to_date="20240701",
        csv_path=csv_path,
        val_csv_path=val_csv_path
    )

# Training environment
train_env = BTCTradingEnv(train_df)
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=1_500_000)
model.save("ppo_btc_trend")

# Validation environment (optional: evaluate after training)
val_env = BTCTradingEnv(val_df)
obs, info = val_env.reset()
for _ in range(len(val_df)):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = val_env.step(action)
    if val_env.current_step % 10000 == 0:
        val_env.render()
    if terminated or truncated:
        break
