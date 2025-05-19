from binance.client import Client
from binance.enums import *
import time
import pandas as pd
import math

class BinanceTrader:
    def __init__(self, api_key, api_secret, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = self._connect()
        self.symbol = "BTCUSDT"
        self.asset = "BTC"
        self.quote = "USDT"
        print(f"[TRACE] BinanceTrader initialized (testnet={self.testnet})")

    def _connect(self):
        print("[TRACE] Connecting to Binance...")
        client = Client(self.api_key, self.api_secret)
        if self.testnet:
            client.API_URL = 'https://testnet.binance.vision/api'
            print("[TRACE] Using Binance TESTNET endpoint.")
        else:
            print("[TRACE] Using Binance MAINNET endpoint.")
        return client

    def get_balances(self):
        print("[TRACE] Fetching account balances...")
        info = self.client.get_account()
        balances = {b['asset']: float(b['free']) for b in info['balances']}
        btc = balances.get(self.asset, 0.0)
        usdt = balances.get(self.quote, 0.0)
        print(f"[TRACE] Balances - BTC: {btc}, USDT: {usdt}")
        return btc, usdt

    def get_price(self):
        print("[TRACE] Fetching current price...")
        ticker = self.client.get_symbol_ticker(symbol=self.symbol)
        price = float(ticker['price'])
        print(f"[TRACE] Current {self.symbol} price: {price}")
        return price

    def buy(self, usdt_amount):
        price = self.get_price()
        qty = usdt_amount / price
        # Round DOWN to nearest 0.0001
        qty = math.floor(qty * 10000) / 10000
        if qty < 0.0001:
            print(f"[TRACE] BUY order qty {qty} below minimum lot size. Order not placed.")
            return None
        print(f"[TRACE] Placing BUY order: {qty} BTC (~{usdt_amount} USDT at {price})")
        order = self.client.create_order(
            symbol=self.symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=qty
        )
        print(f"[TRACE] BUY order response: {order}")
        return order

    def sell(self, btc_amount):
        # Round DOWN to nearest 0.0001
        qty = math.floor(btc_amount * 10000) / 10000
        if qty < 0.0001:
            print(f"[TRACE] SELL order qty {qty} below minimum lot size. Order not placed.")
            return None
        print(f"[TRACE] Placing SELL order: {qty} BTC")
        order = self.client.create_order(
            symbol=self.symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=qty
        )
        print(f"[TRACE] SELL order response: {order}")
        return order

    def reconcile(self):
        print("[TRACE] Reconciling balances...")
        btc, usdt = self.get_balances()
        print(f"[TRACE] Reconciled balances - BTC: {btc}, USDT: {usdt}")
        return btc, usdt

    def wait_for_balance_update(self, prev_btc, prev_usdt, timeout=30):
        print("[TRACE] Waiting for balance update...")
        for _ in range(timeout):
            btc, usdt = self.get_balances()
            if btc != prev_btc or usdt != prev_usdt:
                print("[TRACE] Balance updated.")
                return btc, usdt
            time.sleep(1)
        print("[TRACE] Balance did not update within timeout.")
        return btc, usdt

    def get_ohlcv_dataframe(self, interval=Client.KLINE_INTERVAL_1MINUTE, limit=120):
        print(f"[TRACE] Fetching OHLCV data: interval={interval}, limit={limit}")
        candles = self.client.get_klines(symbol=self.symbol, interval=interval, limit=limit)
        rows = []
        for candle in candles:
            ts = pd.to_datetime(candle[0], unit='ms')
            price = float(candle[4])
            row = {
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": price,
                "volume": float(candle[5]),
                "Close": price
            }
            rows.append((ts, row))
        df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
        print(f"[TRACE] OHLCV DataFrame shape: {df.shape}")
        return df

    def get_realtime_market_status(self):
        print("[TRACE] Fetching real-time market status...")
        ticker = self.client.get_ticker(symbol=self.symbol)
        order_book = self.client.get_order_book(symbol=self.symbol, limit=5)
        trades = self.client.get_recent_trades(symbol=self.symbol, limit=10)

        status = {
            "last_price": float(ticker["lastPrice"]),
            "bid_price": float(order_book["bids"][0][0]),
            "bid_qty": float(order_book["bids"][0][1]),
            "ask_price": float(order_book["asks"][0][0]),
            "ask_qty": float(order_book["asks"][0][1]),
            "volume_24h": float(ticker["volume"]),
            "recent_trades": [
                {
                    "price": float(trade["price"]),
                    "qty": float(trade["qty"]),
                    "is_buyer_maker": trade["isBuyerMaker"],
                    "time": trade["time"]
                }
                for trade in trades
            ]
        }
        print(f"[TRACE] Market status: {status}")
        return status