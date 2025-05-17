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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 1_000  # USD
        self.btc_held = 0
        self.current_step = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        # Use .iloc[0] if value is a Series, to avoid FutureWarning and TypeError
        def get_scalar(val):
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            if isinstance(val, np.ndarray):
                return float(val.item())
            return float(val) if pd.notnull(val) else 0.0
        close = get_scalar(row['Close'])
        sma_10 = get_scalar(row['SMA_10'])
        sma_50 = get_scalar(row['SMA_50'])
        return np.array([
            close,
            sma_10,
            sma_50,
            float(self.balance),
            float(self.btc_held),
            float(self.current_step)
        ], dtype=np.float32)

    def step(self, action):
        done = self.current_step >= self.n_steps - 1
        row = self.df.iloc[self.current_step]
        price = row['Close']
        
        # Simple strategy: act on action
        if action == 0:  # Sell
            self.balance += self.btc_held * price
            self.btc_held = 0
        elif action == 2:  # Buy
            btc_bought = self.balance / price
            self.btc_held += btc_bought
            self.balance = 0

        self.current_step += 1
        if self.current_step >= self.n_steps:
            # If out of bounds, return a dummy observation (zeros)
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._get_obs()

        # Reward = net worth change
        net_worth = self.balance + self.btc_held * price
        reward = net_worth - 1_000

        terminated = done
        truncated = False
        info = {}

        return next_obs, reward, terminated, truncated, info

    def render(self):
        # Ensure balance and btc_held are always floats for formatting
        balance = float(self.balance)
        btc_held = float(self.btc_held)
        print(f"Step: {self.current_step}, Balance: {balance:.2f}, BTC: {btc_held:.6f}")
