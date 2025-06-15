# matic_bot.py

import pandas as pd
import numpy as np
import time
import os
import yaml
from binance.client import Client
from binance.enums import *
from bot_base import load_config, log_trade, telegram_alert, retry_binance_call
from state_utils import load_state, save_state
from strategiesLive import matic_strategy

def run_matic_bot(logger, stop_event):
    # Update config path to point to correct location
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    
    def load_config():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    config = load_config()
    client = Client(config['binance']['api_key'], config['binance']['secret_key'])

    symbol = "MATICUSDT"
    state = load_state(symbol)
    in_position = state.get("in_position", False)
    entry_price = state.get("entry_price", 0)
    last_trade_time = state.get("last_trade_time", 0)
    COOLDOWN_SECONDS = config.get('matic_bot', {}).get('cooldown', 180)

    win_count = 0
    total_trades = 0

    logger.info("🟢 MATIC bot started.")

    def get_klines(interval):
        klines = retry_binance_call(lambda: client.get_klines(symbol=symbol, interval=interval, limit=100))
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df

    while not stop_event.is_set():
        try:
            config = load_config()
            quantity = config.get('matic_bot', {}).get('quantity', 30)
            tp_pct = config.get('matic_bot', {}).get('tp_pct', 0.045)
            sl_pct = config.get('matic_bot', {}).get('sl_pct', 0.025)
            interval = config.get('matic_bot', {}).get('interval', '1h')
            mode = config.get('trading', {}).get('mode', 'test')

            df = get_klines(interval)
            buy_signal, sell_signal, price = matic_strategy(df)

            now = time.time()
            if buy_signal and not in_position and now - last_trade_time > COOLDOWN_SECONDS:
                if mode == "live":
                    retry_binance_call(lambda: client.order_market_buy(symbol=symbol, quantity=quantity))
                    logger.info(f"🟢 LIVE BUY ORDER PLACED at {price}")
                else:
                    retry_binance_call(lambda: client.create_test_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity))
                    logger.info(f"🧪 TEST BUY ORDER at {price}")
                in_position = True
                entry_price = price
                log_trade(symbol, 'buy', price)

                save_state(symbol, {
                    "in_position": in_position,
                    "entry_price": entry_price,
                    "last_trade_time": time.time(),
                    "win_count": win_count,
                    "total_trades": total_trades
                })

                telegram_alert(f"🟢 *{symbol} BUY* at `${price}`")

            elif sell_signal and in_position and time.time() - last_trade_time > COOLDOWN_SECONDS:
                if mode == "live":
                    retry_binance_call(lambda: client.order_market_sell(symbol=symbol, quantity=quantity))
                    logger.info(f"🔴 LIVE SELL ORDER PLACED at {price}")
                else:
                    retry_binance_call(lambda: client.create_test_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity))
                    logger.info(f"🧪 TEST SELL ORDER at {price}")
                in_position = False
                pnl = price - entry_price
                win = pnl > 0
                win_count += 1 if win else 0
                total_trades += 1
                log_trade(symbol, 'sell', price, pnl)
                logger.info(f"📊 PnL: {round(pnl, 3)} | Win Rate: {win_count}/{total_trades}")

                save_state(symbol, {
                    "in_position": in_position,
                    "entry_price": entry_price,
                    "last_trade_time": time.time(),
                    "win_count": win_count,
                    "total_trades": total_trades
                })

                telegram_alert(
                    f"🔴 *{symbol} SELL* at `${price}`\n"
                    f"PnL: `${round(pnl, 3)}`\n"
                    f"Win Rate: `{win_count}/{total_trades}`"
                )

            else:
                logger.info("No action taken.")

        except Exception as e:
            import traceback
            logger.error(f"❌ Exception: {type(e).__name__} - {str(e)}")
            logger.error(traceback.format_exc())


        for _ in range(3600):
            if stop_event.is_set():
                break
            time.sleep(1)

    logger.info("🛑 MATIC bot stopped.")
