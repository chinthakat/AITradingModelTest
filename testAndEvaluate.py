from stable_baselines3 import PPO
from btc_trading_env import BTCTradingEnv
from prepare_data import get_binance_1m_data
import pandas as pd
import os

# Load data from CSV file or fetch if not present
csv_path = "data/binance_btc_1m_val.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
else:
    # You can adjust the date range as needed
    _, df = get_binance_1m_data(
        symbol="BTCUSDT",
        train_from_date=None, train_to_date=None,
        val_from_date="20230201", val_to_date="20230401",
        csv_path="data/binance_btc_1m_val.csv",
        val_csv_path=csv_path
    )

env = BTCTradingEnv(df)

# Load the trained model
model = PPO.load("ppo_btc_trend", env=env)

obs, info = env.reset()
for _ in range(len(df)):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    # Only render/log every 10000 steps
    if env.current_step % 5000 == 0:
        env.render()
    if terminated or truncated:
        break
