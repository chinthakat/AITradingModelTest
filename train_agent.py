from stable_baselines3 import PPO
from btc_trading_env import BTCTradingEnv
from prepare_data import get_btc_data

df = get_btc_data()
env = BTCTradingEnv(df)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("ppo_btc_trend")
