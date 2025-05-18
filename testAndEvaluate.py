from stable_baselines3 import PPO
from btc_trading_env import BTCTradingEnv
import pandas as pd

# Load data from CSV file
csv_path = "data/binance_btc_1m.csv"
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
env = BTCTradingEnv(df)

# Load the trained model
model = PPO.load("ppo_btc_trend", env=env)

obs, info = env.reset()
for _ in range(len(df)):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    # Only render/log every 10000 steps
    if env.current_step % 10000 == 0:
        env.render()
    if terminated or truncated:
        break
