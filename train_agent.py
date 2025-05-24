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

# Load and preprocess data ONCE
if os.path.exists(csv_path) and os.path.exists(val_csv_path):
    train_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    val_df = pd.read_csv(val_csv_path, index_col=0, parse_dates=True)
    train_df = add_indicators(train_df)
    val_df = add_indicators(val_df)
    train_df = train_df.dropna()
    val_df = val_df.dropna()
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

# Print observation shape for debugging
env_test = BTCTradingEnv(train_df)
obs, _ = env_test.reset()
print("Observation shape at reset:", obs.shape)
del env_test

# Create the environment and model ONCE
train_env = BTCTradingEnv(train_df)
model = PPO("MlpPolicy", train_env, verbose=1)

# Continue learning across multiple runs, always starting from the beginning of the data
for run in range(20):
    print(f"=== Training run {run+1} ===")
    # Reset the environment to the beginning of the data for each run
    train_env = BTCTradingEnv(train_df)
    train_env.reset()
    model.set_env(train_env)
    model.learn(total_timesteps=800_000, reset_num_timesteps=False)
    model.save(f"ppo_btc_trend_run{run+1}")

    # Optionally evaluate on validation set
    val_env = BTCTradingEnv(val_df)
    obs, info = val_env.reset()
    for _ in range(len(val_df)):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = val_env.step(action)
        log_line = (
            f"Step: {val_env.current_step} | "
            f"Original Net Worth: {info.get('original_net_worth', 0):.2f} | "
            f"Net Worth: {info.get('net_worth', 0):.2f} | "
            f"Total Trades: {info.get('total_trades', 0)} | "
            f"Buy Trades: {info.get('buy_trades', 0)} | "
            f"Sell Trades: {info.get('sell_trades', 0)} | "
            f"Profit: {info.get('net_worth', 0) - info.get('original_net_worth', 0):.2f} | "
            f"Profitable Trades: {info.get('profitable_trades', 0)} | "
            f"Loss Trades: {info.get('loss_trades', 0)}"
        )
        print(log_line)
        with open("trading_log.txt", "a") as f:
            f.write(log_line + "\n")
        val_env.render()
        if terminated or truncated:
            break
