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

from trading_features import add_indicators

if os.path.exists(csv_path) and os.path.exists(val_csv_path):
    train_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    val_df = pd.read_csv(val_csv_path, index_col=0, parse_dates=True)
    print(f"Loaded training data from {csv_path}")
    print(f"Loaded validation data from {val_csv_path}")
    train_df = add_indicators(train_df)
    val_df = add_indicators(val_df)
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    assert not train_df.isnull().values.any(), "train_df still contains NaNs!"
    assert not val_df.isnull().values.any(), "val_df still contains NaNs!"
else:
    result = get_binance_1m_data(
        symbol="BTCUSDT",
        train_from_date="20250101", train_to_date="20250301",
        val_from_date="20250301", val_to_date="20250515",
        csv_path=csv_path,
        val_csv_path=val_csv_path
    )
    if result is None or len(result) != 2:
        raise RuntimeError("get_binance_1m_data did not return (train_df, val_df)")
    train_df, val_df = result
    train_df = add_indicators(train_df)
    val_df = add_indicators(val_df)
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    assert not train_df.isnull().values.any(), "train_df still contains NaNs!"
    assert not val_df.isnull().values.any(), "val_df still contains NaNs!"

# Training environment
train_env = BTCTradingEnv(train_df)
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=100_000)
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
