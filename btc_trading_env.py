import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from trading_features import build_observation

class BTCTradingEnv(gym.Env):
    def __init__(self, df):
        super(BTCTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25 + 10*7 + 10*10,), dtype=np.float32)  # Update shape as needed

        # --- Add deques for new features ---
        self.sma_10_vals = deque([0.0]*10, maxlen=10)
        self.sma_50_vals = deque([0.0]*10, maxlen=10)
        self.ema_10_vals = deque([0.0]*10, maxlen=10)
        self.ema_50_vals = deque([0.0]*10, maxlen=10)
        self.sma_diff_vals = deque([0.0]*10, maxlen=10)
        self.close_sma10_vals = deque([0.0]*10, maxlen=10)
        self.close_sma50_vals = deque([0.0]*10, maxlen=10)
        # --- Initialize bb_features ---
        self.bb_features = {k: deque([0.0]*10, maxlen=10) for k in [
            "BB_MID", "BB_HIGH", "BB_LOW",
            "BB_MID_close", "BB_HIGH_close", "BB_LOW_close",
            "BB_MID_open", "BB_HIGH_open", "BB_LOW_open",
            "BB_norm_pos"
        ]}
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
        # --- Reset new feature deques ---
        self.sma_10_vals = deque([0.0]*10, maxlen=10)
        self.sma_50_vals = deque([0.0]*10, maxlen=10)
        self.ema_10_vals = deque([0.0]*10, maxlen=10)
        self.ema_50_vals = deque([0.0]*10, maxlen=10)
        self.sma_diff_vals = deque([0.0]*10, maxlen=10)
        self.close_sma10_vals = deque([0.0]*10, maxlen=10)
        self.close_sma50_vals = deque([0.0]*10, maxlen=10)
        # --- Reset bb_features ---
        self.bb_features = {k: deque([0.0]*10, maxlen=10) for k in [
            "BB_MID", "BB_HIGH", "BB_LOW",
            "BB_MID_close", "BB_HIGH_close", "BB_LOW_close",
            "BB_MID_open", "BB_HIGH_open", "BB_LOW_open",
            "BB_norm_pos"
        ]}
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        base_obs = build_observation(
            self.df,
            self.current_step,
            self.balance,
            self.btc_held,
            self.last_trade_profits
        )
        # --- Add new features to observation ---
        feature_obs = []
        feature_obs.extend(list(self.sma_10_vals))
        feature_obs.extend(list(self.sma_50_vals))
        feature_obs.extend(list(self.ema_10_vals))
        feature_obs.extend(list(self.ema_50_vals))
        feature_obs.extend(list(self.sma_diff_vals))
        feature_obs.extend(list(self.close_sma10_vals))
        feature_obs.extend(list(self.close_sma50_vals))
        # ...existing bb_features code...
        bb_obs = []
        for key in [
            "BB_MID", "BB_HIGH", "BB_LOW",
            "BB_MID_close", "BB_HIGH_close", "BB_LOW_close",
            "BB_MID_open", "BB_HIGH_open", "BB_LOW_open",
            "BB_norm_pos"
        ]:
            bb_obs.extend(list(self.bb_features[key]))
        obs = np.concatenate([base_obs, np.array(feature_obs, dtype=np.float32), np.array(bb_obs, dtype=np.float32)])
        return obs

    def step(self, action, log_file="trading_log.txt"):
        # Ensure action is an int (Stable-Baselines3 may pass a numpy array)
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        done = self.current_step >= self.n_steps - 1
        row = self.df.iloc[self.current_step]
        open_ = row['open'] if 'open' in row else row['Open']
        close = row['close'] if 'close' in row else row['Close']
        price = row['close'] if 'close' in row else row['Close']  # <-- FIX: define price here
        bb_mid = row['BB_MID']
        bb_high = row['BB_HIGH']
        bb_low = row['BB_LOW']
        bb_mid_close = bb_mid - close
        bb_high_close = bb_high - close
        bb_low_close = bb_low - close
        bb_mid_open = bb_mid - open_
        bb_high_open = bb_high - open_
        bb_low_open = bb_low - open_
        bb_norm_pos = (close - bb_mid) / (bb_high - bb_low + 1e-8)

        self.bb_features["BB_MID"].append(bb_mid)
        self.bb_features["BB_HIGH"].append(bb_high)
        self.bb_features["BB_LOW"].append(bb_low)
        self.bb_features["BB_MID_close"].append(bb_mid_close)
        self.bb_features["BB_HIGH_close"].append(bb_high_close)
        self.bb_features["BB_LOW_close"].append(bb_low_close)
        self.bb_features["BB_MID_open"].append(bb_mid_open)
        self.bb_features["BB_HIGH_open"].append(bb_high_open)
        self.bb_features["BB_LOW_open"].append(bb_low_open)
        self.bb_features["BB_norm_pos"].append(bb_norm_pos)

        # --- Compute new features ---
        sma_10 = row['SMA_10']
        sma_50 = row['SMA_50']
        ema_10 = row['EMA_10']
        ema_50 = row['EMA_50']
        sma_diff = sma_10 - sma_50
        close_sma10 = row['close'] - sma_10
        close_sma50 = row['close'] - sma_50
        # --- Append to deques ---
        self.sma_10_vals.append(sma_10)
        self.sma_50_vals.append(sma_50)
        self.ema_10_vals.append(ema_10)
        self.ema_50_vals.append(ema_50)
        self.sma_diff_vals.append(sma_diff)
        self.close_sma10_vals.append(close_sma10)
        self.close_sma50_vals.append(close_sma50)

        trade_trace = ""
        large_trade_penalty = 0  # Default value for all actions
        # Track if a trade was made
        trade_made = False
        trade_profit = None  # Track profit/loss for this trade
        # Trade execution
        if action == 0:  # Sell All
            if self.btc_held > 0:
                sell_value = self.btc_held * price
                fee = sell_value * 0.001  # 0.1% fee
                net_sell_value = sell_value - fee
                profit_loss = net_sell_value - 0  # If you want to track buy price, you can store it and use here
                trade_trace = f"SELL: Sold {self.btc_held:.6f} BTC at {price:.2f} for {net_sell_value:.2f} USD (fee: {fee:.2f}). Profit/Loss: {net_sell_value - 0:.2f} USD\n"
                self.balance += net_sell_value
                self.btc_held = 0
                large_trade_penalty = 0
                trade_made = True
                trade_profit = profit_loss
            else:
                trade_trace = f"SELL: Tried to sell with 0 BTC at {price:.2f} -- strongly discouraged\n"
                large_trade_penalty = -5  # Strongly discourage selling with no BTC
        elif action >= 1:  # Buy actions (action 1+)
            buy_fraction = min(action * 0.01, 1.0)
            max_buy = float(self.balance) * buy_fraction
            min_trade_btc = 0.00001  # Minimum trade size
            # Apply 0.1% fee to the amount spent
            total_cost = max_buy * 1.001
            btc_bought = max_buy / float(price) if price > 0 else 0.0
            if total_cost > 0 and self.balance >= total_cost and buy_fraction > 0 and btc_bought >= min_trade_btc:
                self.btc_held += btc_bought
                self.balance -= total_cost
                trade_made = True
                # Penalty/Reward logic: lower buy_fraction = better reward, >0.3 = extremely discouraged
                if buy_fraction <= 0.05:
                    large_trade_penalty = 2  # Encourage small trades
                elif buy_fraction <= 0.10:
                    large_trade_penalty = 1  # Mildly encourage up to 10%
                elif buy_fraction <= 0.30:
                    large_trade_penalty = -2  # Discourage 11-30%
                else:
                    large_trade_penalty = -10  # Extremely discourage >30%
                trade_trace = f"BUY {buy_fraction*100:.1f}%: Bought {btc_bought:.6f} BTC at {price:.2f} for {max_buy:.2f} USD (fee: {total_cost - max_buy:.2f}, total: {total_cost:.2f})\n"
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

