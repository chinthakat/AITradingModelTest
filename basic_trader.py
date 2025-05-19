import time
from trading_features import add_indicators, build_observation
from stable_baselines3 import PPO
from binance_trader import BinanceTrader
from collections import deque
import os

class BasicTrader:
    def __init__(self, api_key, api_secret, model_path, testnet=True):
        self.trader = BinanceTrader(api_key, api_secret, testnet=testnet)
        self.model = PPO.load(model_path)
        self.last_trade_profits = deque([0.0] * 5, maxlen=5)
        self.df = self.trader.get_ohlcv_dataframe(limit=120)
        self.df = add_indicators(self.df)
        self.df = self.df.dropna()
        self.btc_balance, self.usdt_balance = self.trader.get_balances()
        self.buy_count = 0
        self.sell_count = 0
        print(f"Initial balances: BTC={self.btc_balance}, USDT={self.usdt_balance}")

    def run(self, interval_sec=60):
        while True:
            try:
                # Get the latest candle and update DataFrame
                new_df = self.trader.get_ohlcv_dataframe(limit=120)
                new_df = add_indicators(new_df)
                new_df = new_df.dropna()
                if len(new_df) == 0:
                    print("Waiting for enough data to compute indicators...")
                    time.sleep(interval_sec)
                    continue
                self.df = new_df
                current_step = len(self.df) - 1

                # Build observation
                obs = build_observation(
                    self.df, current_step,
                    self.usdt_balance, self.btc_balance, self.last_trade_profits
                )

                # Model prediction
                action, _ = self.model.predict(obs)

                # Action mapping (adjust as needed)
                action_type = "HOLD"
                trade_amount = 0
                profit_loss = 0
                price = float(self.df.iloc[current_step]['Close'])
                prev_net_worth = self.usdt_balance + self.btc_balance * price

                if action == 0:
                    # SELL all BTC
                    if self.btc_balance > 0.00001:
                        action_type = "SELL"
                        prev_btc = self.btc_balance
                        prev_usdt = self.usdt_balance
                        order = self.trader.sell(self.btc_balance)
                        if order is not None:
                            self.sell_count += 1
                        time.sleep(2)
                        self.btc_balance, self.usdt_balance = self.trader.reconcile()
                        profit_loss = (price - float(self.df.iloc[current_step-1]['Close'])) * prev_btc
                        self.last_trade_profits.append(profit_loss)
                        trade_amount = prev_btc
                elif action >= 1:
                    # BUY with a fraction of USDT, as a multiple of 0.01
                    buy_fraction = action * 0.01
                    if buy_fraction > 1.0:
                        buy_fraction = 1.0
                    max_buy = self.usdt_balance * buy_fraction
                    price = float(self.df.iloc[current_step]['Close'])
                    btc_to_buy = max_buy / price if price > 0 else 0
                    # Binance min qty and step size for BTCUSDT is 0.0001
                    btc_to_buy = int(btc_to_buy / 0.0001) * 0.0001
                    if btc_to_buy >= 0.0001:
                        action_type = f"BUY {buy_fraction*100:.1f}%"
                        prev_btc = self.btc_balance
                        prev_usdt = self.usdt_balance
                        order = self.trader.buy(btc_to_buy * price)  # Pass USDT amount to .buy()
                        if order is not None:
                            self.buy_count += 1
                        time.sleep(2)
                        self.btc_balance, self.usdt_balance = self.trader.reconcile()
                        profit_loss = 0  # Not realized yet
                        self.last_trade_profits.append(profit_loss)
                        trade_amount = btc_to_buy
                    else:
                        action_type = "BUY (below min lot size)"

                net_worth = self.usdt_balance + self.btc_balance * price
                print(
                    f"Action: {action} ({action_type}), "
                    f"Trade Amount: {trade_amount:.6f} BTC, "
                    f"Price: {price:.2f}, "
                    f"USDT: {self.usdt_balance:.2f}, BTC: {self.btc_balance:.6f}, "
                    f"Net Worth: {net_worth:.2f}, "
                    f"Profit/Loss: {profit_loss:.2f}"
                )
                print(
                    f"Reconciled balances - USDT: {self.usdt_balance:.2f}, BTC: {self.btc_balance:.6f}, "
                    f"Prev Net Worth: {prev_net_worth:.2f}, Current Net Worth: {net_worth:.2f}, "
                    f"Total BUY orders: {self.buy_count}, Total SELL orders: {self.sell_count}"
                )

                time.sleep(interval_sec)
            except Exception as e:
                print(f"Error: {e}. Retrying in 10 seconds...")
                time.sleep(10)

if __name__ == "__main__":
    # You can set your API keys as environment variables for safety
    API_KEY = 'jpXLj5rSS0WDVPmvKxanETlbQwEdkz2L5ZzZaBmUqOgM8Zq0IxktFK4r67K3jA0A' #os.getenv("BINANCE_TESTNET_API_KEY", "YOUR_TESTNET_API_KEY")
    API_SECRET = 'YoIZwMg1m32TZCRFLrqyqvfjmd7KkIyC2YXmQUa02Ae9Qqsj36xjxnXOzfWvxSM3' #.getenv("BINANCE_TESTNET_API_SECRET", "YOUR_TESTNET_API_SECRET")
    MODEL_PATH = "ppo_btc_trend"

    trader = BasicTrader(API_KEY, API_SECRET, MODEL_PATH, testnet=True)
    trader.run(interval_sec=60)