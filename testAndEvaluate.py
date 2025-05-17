from stable_baselines3 import PPO
from btc_trading_env import BTCTradingEnv
from prepare_data import get_btc_data

# Prepare data and environment

df = get_btc_data()
env = BTCTradingEnv(df)

# Load the trained model
model = PPO.load("ppo_btc_trend", env=env)

obs, info = env.reset()
for _ in range(len(df)):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        break
