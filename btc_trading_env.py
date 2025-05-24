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
        # Fix: set the correct shape to match your actual observation vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(206,), dtype=np.float32)

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
        self.steps_since_last_trade = 100  # Large initial value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 10_000      # Reset balance for every run
        self.btc_held = 0.05       # Reset BTC held for every run
        self.current_step = 0
        self.original_net_worth = self.balance + self.btc_held * (
            self.df.iloc[0]['close'] if 'close' in self.df.columns else self.df.iloc[0]['Close']
        )
        self.max_net_worth = self.original_net_worth
        self.total_trades = 0
        self.profitable_trades = 0
        self.loss_trades = 0
        self.buy_trades = 0
        self.sell_trades = 0
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
        self.steps_since_last_trade = 100
        self.prev_net_worth = self.original_net_worth
        return self._get_obs(), {}

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
        # Calculate current profit (net_worth - prev_net_worth)
        net_worth = self.balance + self.btc_held * (
            self.df.iloc[self.current_step]['close'] if 'close' in self.df.columns else self.df.iloc[self.current_step]['Close']
        )
        current_profit = net_worth - self.prev_net_worth
        obs = np.concatenate([
            base_obs,
            np.array([current_profit], dtype=np.float32),  # Add current profit as an important observation
            np.array(feature_obs, dtype=np.float32),
            np.array(bb_obs, dtype=np.float32)
        ])
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
        if action == 0:  # Hold
            trade_trace = f"HOLD: No action taken at {price:.2f}\n"
            large_trade_penalty = 0.5

        elif action == 1:  # Sell 10% of BTC held
            sell_fraction = 0.1
            btc_to_sell = self.btc_held * sell_fraction
            min_trade_btc = 0.00001
            if btc_to_sell > min_trade_btc and self.btc_held >= btc_to_sell:
                sell_value = btc_to_sell * price
                fee = sell_value * 0.001  # 0.1% fee
                net_sell_value = sell_value - fee
                self.btc_held -= btc_to_sell
                self.balance += net_sell_value
                trade_made = True
                self.sell_trades += 1
                large_trade_penalty = 2
                trade_trace = f"SELL 10%: Sold {btc_to_sell:.6f} BTC at {price:.2f} for {net_sell_value:.2f} USD (fee: {fee:.2f})\n"
                trade_profit = net_sell_value - 0
            else:
                trade_trace = f"SELL 10%: Tried to sell with {self.btc_held:.6f} BTC at {price:.2f} -- discouraged\n"
                large_trade_penalty = -5

        elif action == 2:  # Sell 25% of BTC held
            sell_fraction = 0.25
            btc_to_sell = self.btc_held * sell_fraction
            min_trade_btc = 0.00001
            if btc_to_sell > min_trade_btc and self.btc_held >= btc_to_sell:
                sell_value = btc_to_sell * price
                fee = sell_value * 0.001
                net_sell_value = sell_value - fee
                self.btc_held -= btc_to_sell
                self.balance += net_sell_value
                trade_made = True
                self.sell_trades += 1
                large_trade_penalty = 1.5
                trade_trace = f"SELL 25%: Sold {btc_to_sell:.6f} BTC at {price:.2f} for {net_sell_value:.2f} USD (fee: {fee:.2f})\n"
                trade_profit = net_sell_value - 0
            else:
                trade_trace = f"SELL 25%: Tried to sell with {self.btc_held:.6f} BTC at {price:.2f} -- discouraged\n"
                large_trade_penalty = -5

        elif action == 3:  # Sell 50% of BTC held
            sell_fraction = 0.5
            btc_to_sell = self.btc_held * sell_fraction
            min_trade_btc = 0.00001
            if btc_to_sell > min_trade_btc and self.btc_held >= btc_to_sell:
                sell_value = btc_to_sell * price
                fee = sell_value * 0.001
                net_sell_value = sell_value - fee
                self.btc_held -= btc_to_sell
                self.balance += net_sell_value
                trade_made = True
                self.sell_trades += 1
                large_trade_penalty = 1
                trade_trace = f"SELL 50%: Sold {btc_to_sell:.6f} BTC at {price:.2f} for {net_sell_value:.2f} USD (fee: {fee:.2f})\n"
                trade_profit = net_sell_value - 0
            else:
                trade_trace = f"SELL 50%: Tried to sell with {self.btc_held:.6f} BTC at {price:.2f} -- discouraged\n"
                large_trade_penalty = -5

        elif action == 4:  # Sell 75% of BTC held
            sell_fraction = 0.75
            btc_to_sell = self.btc_held * sell_fraction
            min_trade_btc = 0.00001
            if btc_to_sell > min_trade_btc and self.btc_held >= btc_to_sell:
                sell_value = btc_to_sell * price
                fee = sell_value * 0.001
                net_sell_value = sell_value - fee
                self.btc_held -= btc_to_sell
                self.balance += net_sell_value
                trade_made = True
                self.sell_trades += 1
                large_trade_penalty = 0.75
                trade_trace = f"SELL 75%: Sold {btc_to_sell:.6f} BTC at {price:.2f} for {net_sell_value:.2f} USD (fee: {fee:.2f})\n"
                trade_profit = net_sell_value - 0
            else:
                trade_trace = f"SELL 75%: Tried to sell with {self.btc_held:.6f} BTC at {price:.2f} -- discouraged\n"
                large_trade_penalty = -0.5

        elif action == 5:  # Sell 100% of BTC held
            sell_fraction = 1.0
            btc_to_sell = self.btc_held * sell_fraction
            min_trade_btc = 0.00001
            if btc_to_sell > min_trade_btc and self.btc_held >= btc_to_sell:
                sell_value = btc_to_sell * price
                fee = sell_value * 0.001
                net_sell_value = sell_value - fee
                self.btc_held -= btc_to_sell
                self.balance += net_sell_value
                trade_made = True
                self.sell_trades += 1
                large_trade_penalty = -1
                trade_trace = f"SELL 100%: Sold {btc_to_sell:.6f} BTC at {price:.2f} for {net_sell_value:.2f} USD (fee: {fee:.2f})\n"
                trade_profit = net_sell_value - 0
            else:
                trade_trace = f"SELL 100%: Tried to sell with {self.btc_held:.6f} BTC at {price:.2f} -- discouraged\n"
                large_trade_penalty = 0

        # BUY logic remains unchanged
        elif action >= 6:
            buy_fraction = min((action - 5) * 0.01, 1.0)
            max_buy = float(self.balance) * buy_fraction
            min_trade_btc = 0.00001  # Minimum trade size
            # Apply 0.1% fee to the amount spent
            total_cost = max_buy * 1.001
            btc_bought = max_buy / float(price) if price > 0 else 0.0
            if total_cost > 0 and self.balance >= total_cost and buy_fraction > 0 and btc_bought >= min_trade_btc:
                self.btc_held += btc_bought
                self.balance -= total_cost
                trade_made = True
                self.buy_trades += 1
                # Penalty/Reward logic: lower buy_fraction = better reward, >0.3 = extremely discouraged
                if buy_fraction <= 0.05:
                    large_trade_penalty = 2  # Encourage small trades
                elif buy_fraction <= 0.10:
                    large_trade_penalty = 1  # Mildly encourage up to 10%
                elif buy_fraction <= 0.30:
                    large_trade_penalty = -0.5  # Discourage 11-30%
                else:
                    large_trade_penalty = -1  # Extremely discourage >30%
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

        net_worth = self.balance + self.btc_held * price
        drawdown = (self.max_net_worth - net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        self.max_net_worth = max(self.max_net_worth, net_worth)
        profit = net_worth - self.original_net_worth

        # Calculate current profit (net_worth - prev_net_worth)
        current_profit = net_worth - self.prev_net_worth 
        # --- Save current net worth as previous for next step ---
        self.prev_net_worth = net_worth
        # After trade_profit is set (after trade_made)
        if trade_made and current_profit is not None:
            self.last_trade_profits.append(current_profit)
        self.current_step += 1
        next_obs = np.zeros(self.observation_space.shape, dtype=np.float32) if self.current_step >= self.n_steps else self._get_obs()
        # Net worth and drawdown
  

      

        # Use original net worth for profit calculation
        trade_bonus = 0.5 if (current_profit > 0 and trade_made) else 0
        inactivity_penalty = -0.2 if not trade_made else 0
        # Add penalty for trading too frequently
        trade_frequency_penalty = 0
        if  self.steps_since_last_trade > 100:
            trade_frequency_penalty = -(1 + self.steps_since_last_trade*0.01)

        if current_profit < 0:
            large_trade_penalty = 0
        # --- Updated reward: positive if profit, negative if loss (per trade) ---
        trade_profit_reward = 0
        
        
        if current_profit is not None:
            if current_profit > 0:
                trade_profit_reward = 4  # Positive reward proportional to profit
            elif current_profit < 0:
                trade_profit_reward = -4  # Negative reward proportional to loss
        
        current_profit_reward = current_profit * 0.01
        if current_profit < 0:
            current_profit_reward = current_profit * 0.5
        reward = (
            profit * 0.002
            - drawdown * 5
            + large_trade_penalty
            + trade_bonus
            + inactivity_penalty
            + trade_frequency_penalty
            + trade_profit_reward
            +current_profit_reward  # <-- Add this to the reward
        )
        terminated = done

        truncated = False
        info = {
            "net_worth": net_worth,
            "drawdown": drawdown,
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "loss_trades": self.loss_trades,
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "net_worth_at_step": net_worth,
            "total_buy_trades": self.buy_trades,
            "total_sell_trades": self.sell_trades,
            "original_net_worth": self.original_net_worth,  # Optionally add for logging
            "current_profit": current_profit,  # <-- Add current profit to info
        }

        # Print info dict to terminal and log file every 100 steps
        if self.current_step % 2500 == 0 or terminated:
            info_line = (
                f"[INFO] Step: {self.current_step} | "
                f"Original Net Worth: {info.get('original_net_worth', 0):.2f} | "
                f"Net Worth: {info['net_worth']:.2f} | "
                f"Total Trades: {info['total_trades']} | "
                f"Buy Trades: {info['buy_trades']} | "
                f"Sell Trades: {info['sell_trades']} | "
                f"Profit: {info['net_worth'] - info.get('original_net_worth', 0):.2f} | "
                f"Current Profit: {info['current_profit']:.2f} | "  # <-- Log current profit
                f"Profitable Trades: {info['profitable_trades']} | "
                f"Loss Trades: {info['loss_trades']}\n"
            )
            print(info_line, end="")
            with open(log_file, "a") as f:
                f.write(info_line)


        # Trace log for buy/sell with reward details
        if trade_made and self.total_trades % 5000 == 0:
            trade_type = "BUY" if action >= 6 else "SELL"
            trade_amount = 0
            trade_fraction = 0
            if trade_type == "BUY":
                trade_amount = btc_bought if 'btc_bought' in locals() else 0
                trade_fraction = buy_fraction
            else:
                trade_amount = btc_to_sell if 'btc_to_sell' in locals() else 0
                trade_fraction = sell_fraction
            reward_details = (
                f"\n[TRADE] Step: {self.current_step} | Type: {trade_type}\n"
                f"  Amount: {trade_amount:.6f} BTC\n"
                f"  USD/AUD Amount: {trade_amount * price:.2f}\n"
                f"  Trade %: {trade_fraction}%\n"
                f"  Net Worth: {net_worth:.2f}\n"
                f"  Current Profit: {current_profit:.2f}\n"
                f"  Reward: {reward:.4f}\n"
                f"  Breakdown:\n"
                f"    profit*0.002           = {profit*0.002:.4f}\n"
                f"    -drawdown*5            = {-drawdown*5:.4f}\n"
                f"    large_trade_penalty    = {large_trade_penalty:.4f}\n"
                f"    trade_bonus            = {trade_bonus:.4f}\n"
                f"    inactivity_penalty     = {inactivity_penalty:.4f}\n"
                f"    trade_frequency_penalty= {trade_frequency_penalty:.4f}\n"
                f"    trade_profit_reward    = {trade_profit_reward:.4f}\n"
                f"    current_profit_reward  = {current_profit_reward:.4f}\n"
            )
            print(reward_details, end="")
            with open(log_file, "a") as f:
                f.write(reward_details)
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

