import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BTCTradingEnv(gym.Env):
    def __init__(self, df):
        super(BTCTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 10_000
        self.btc_held = 0
        self.current_step = 0
        self.max_net_worth = self.balance
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        def get_scalar(val):
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            if isinstance(val, np.ndarray):
                return float(val.item())
            return float(val) if pd.notnull(val) else 0.0

        return np.array([
            get_scalar(row['Close']),
            get_scalar(row['SMA_10']),
            get_scalar(row['SMA_50']),
            get_scalar(row['RSI']),
            get_scalar(row['MACD']),
            get_scalar(row['MACD_Signal']),
            float(self.balance),
            float(self.btc_held),
            float(self.current_step)
        ], dtype=np.float32)

    def step(self, action):
        done = self.current_step >= self.n_steps - 1
        row = self.df.iloc[self.current_step]
        price = row['Close']

        # Trade execution
        if action == 0:  # Sell
            self.balance += self.btc_held * price
            self.btc_held = 0
        elif action == 2:  # Buy
            btc_bought = self.balance / price
            self.btc_held += btc_bought
            self.balance = 0

        self.current_step += 1
        next_obs = np.zeros(self.observation_space.shape, dtype=np.float32) if self.current_step >= self.n_steps else self._get_obs()

        # Net worth and drawdown
        net_worth = self.balance + self.btc_held * price
        drawdown = (self.max_net_worth - net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        self.max_net_worth = max(self.max_net_worth, net_worth)

        # Reward function
        profit = net_worth - 10_000
        reward = profit * 0.001 - drawdown * 10  # small profit incentive, big drawdown penalty

        terminated = done
        truncated = False
        info = {
            "net_worth": net_worth,
            "drawdown": drawdown,
        }

        return next_obs, reward, terminated, truncated, info

    def render(self):
        net_worth = self.balance + self.btc_held * self.df.iloc[self.current_step - 1]['Close']
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, BTC: {self.btc_held:.6f}, Net Worth: {net_worth:.2f}")
