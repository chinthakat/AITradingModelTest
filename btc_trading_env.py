import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from trading_features import build_observation
MAX_BUY_FRACTION = 0.05  # 5% of balance

class BTCTradingEnv(gym.Env):
    def __init__(self, df):
        super(BTCTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        self.action_space = spaces.Discrete(7)
        # 9 original + 5 last trade profits + 5 SMA_10 + 5 SMA_50 = 24
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)
        self.total_trades = 0
        self.profitable_trades = 0
        self.loss_trades = 0
        self.last_trade_profits = deque([0.0]*5, maxlen=5)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 10_000
        self.btc_held = 0
        self.current_step = 0
        self.max_net_worth = self.balance
        self.total_trades = 0
        self.profitable_trades = 0
        self.loss_trades = 0
        self.last_trade_profits = deque([0.0]*5, maxlen=5)
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return build_observation(
            self.df,
            self.current_step,
            self.balance,
            self.btc_held,
            self.last_trade_profits
        )

    def step(self, action, log_file="trading_log.txt"):
        # Ensure action is an int (Stable-Baselines3 may pass a numpy array)
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        done = self.current_step >= self.n_steps - 1
        row = self.df.iloc[self.current_step]
        price = row['Close']
        trade_trace = ""
        large_trade_penalty = 0  # Default value for all actions
        # Track if a trade was made
        trade_made = False
        trade_profit = None  # Track profit/loss for this trade
        # Trade execution
        if action == 0:  # Sell All
            if self.btc_held > 0:
                sell_value = self.btc_held * price
                profit_loss = sell_value - 0  # If you want to track buy price, you can store it and use here
                trade_trace = f"SELL: Sold {self.btc_held:.6f} BTC at {price:.2f} for {sell_value:.2f} USD. Profit/Loss: {sell_value - 0:.2f} USD\n"
                self.balance += sell_value
                self.btc_held = 0
                large_trade_penalty = 0
                trade_made = True
                trade_profit = profit_loss
            else:
                trade_trace = f"SELL: Tried to sell with 0 BTC at {price:.2f} -- strongly discouraged\n"
                large_trade_penalty = -5  # Strongly discourage selling with no BTC
        elif action >= 1:  # Buy actions (action 1+)
            # Define buy fractions: 0.5% to 10% (0.005 to 0.10, step 0.005), then 11% to 30% (0.11 to 0.30, step 0.01)
            buy_fractions = [0] + [round(x * 0.005, 3) for x in range(1, 21)] + [round(0.10 + x * 0.01, 3) for x in range(1, 21)]
            # action 1 = 0.5%, 2 = 1%, ..., 20 = 10%, 21 = 11%, ..., 40 = 30%
            buy_fraction = float(buy_fractions[action]) if action < len(buy_fractions) else 0.0
            max_buy = float(self.balance) * buy_fraction
            min_trade_btc = 0.00001  # Minimum trade size
            btc_bought = max_buy / float(price) if price > 0 else 0.0
            if max_buy > 0 and self.balance >= max_buy and buy_fraction > 0 and btc_bought >= min_trade_btc:
                self.btc_held += btc_bought
                self.balance -= max_buy
                trade_made = True
                # Penalty/Reward logic
                if buy_fraction <= 0.025:
                    large_trade_penalty = 5  # Strongly encourage very small trades
                elif buy_fraction <= 0.05:
                    large_trade_penalty = 3  # Encourage small trades
                elif buy_fraction <= 0.10:
                    large_trade_penalty = 1  # Mildly encourage up to 10%
                elif buy_fraction <= 0.30:
                    large_trade_penalty = -3  # Discourage 11-30%
                else:
                    large_trade_penalty = -10  # Strongly discourage >30%
                trade_trace = f"BUY {buy_fraction*100:.1f}%: Bought {btc_bought:.6f} BTC at {price:.2f} for {max_buy:.2f} USD\n"
                trade_profit = 0  # For buys, profit is not realized yet
            else:
                btc_bought = 0
                large_trade_penalty = -2  # can't even afford min trade
                trade_trace = f"BUY {buy_fraction*100:.1f}%: Insufficient balance for min trade at {price:.2f} -- discouraged\n"
        # Track trade stats
        if trade_made:
            self.total_trades += 1
            if trade_profit is not None:
                if trade_profit > 0:
                    self.profitable_trades += 1
                elif trade_profit < 0:
                    self.loss_trades += 1
        # After trade_profit is set (after trade_made)
        if trade_made and trade_profit is not None:
            self.last_trade_profits.append(trade_profit)
        self.current_step += 1
        next_obs = np.zeros(self.observation_space.shape, dtype=np.float32) if self.current_step >= self.n_steps else self._get_obs()
        # Net worth and drawdown
        net_worth = self.balance + self.btc_held * price
        drawdown = (self.max_net_worth - net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        self.max_net_worth = max(self.max_net_worth, net_worth)
        # Reward: growth - drawdown - large_trade_penalty + trade bonus - inactivity penalty
        profit = net_worth - 10_000
        trade_bonus = 0.5 if trade_made else 0
        inactivity_penalty = -0.2 if not trade_made else 0
        reward = profit * 0.002 - drawdown * 5 + large_trade_penalty + trade_bonus + inactivity_penalty
        terminated = done
        truncated = False
        info = {
            "net_worth": net_worth,
            "drawdown": drawdown,
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "loss_trades": self.loss_trades,
        }
        # Trace log for buy/sell
        if trade_trace and self.total_trades % 10000 == 0:
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
            f"Step Profit/Loss: {step_profit:.2f}, "
            f"Total Trades: {self.total_trades}, Profitable: {self.profitable_trades}, Loss: {self.loss_trades}\n"
        )
        print(log_line, end="")
        with open(log_file, "a") as f:
            f.write(log_line)

