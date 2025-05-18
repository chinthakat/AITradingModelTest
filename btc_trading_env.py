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

    def step(self, action, log_file="trading_log.txt"):
        done = self.current_step >= self.n_steps - 1
        row = self.df.iloc[self.current_step]
        price = row['Close']
        trade_trace = ""
        # Trade execution
        if action == 0:  # Sell
            sell_value = self.btc_held * price
            profit_loss = sell_value - 0  # If you want to track buy price, you can store it and use here
            trade_trace = f"SELL: Sold {self.btc_held:.6f} BTC at {price:.2f} for {sell_value:.2f} USD. Profit/Loss: {sell_value - 0:.2f} USD\n"
            self.balance += sell_value
            self.btc_held = 0
        elif action == 2:  # Buy
            btc_bought = self.balance / price
            trade_trace = f"BUY: Bought {btc_bought:.6f} BTC at {price:.2f} for {self.balance:.2f} USD\n"
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
        # Trace log for buy/sell
        if trade_trace:
            print(trade_trace, end="")
            with open(log_file, "a") as f:
                f.write(trade_trace)
        return next_obs, reward, terminated, truncated, info

    def render(self, log_file="trading_log.txt"):
        import datetime
        net_worth = self.balance + self.btc_held * self.df.iloc[self.current_step - 1]['Close']
        # Calculate profit/loss for this step
        if self.current_step > 1:
            prev_net_worth = self.balance + self.btc_held * self.df.iloc[self.current_step - 2]['Close']
            step_profit = net_worth - prev_net_worth
        else:
            step_profit = 0.0
        log_line = (
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"BTC: {self.btc_held:.6f}, Net Worth: {net_worth:.2f}, "
            f"Step Profit/Loss: {step_profit:.2f}\n"
        )
        print(log_line, end="")
        with open(log_file, "a") as f:
            f.write(log_line)

