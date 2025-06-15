# eth_bot.py

import pandas as pd
import numpy as np
import time
import os
import yaml
from binance.client import Client
from binance.enums import *
from bot_base import load_config, log_trade, telegram_alert, retry_binance_call
from state_utils import load_state, save_state
from strategiesLive import eth_strategy

def run_eth_bot(logger, stop_event):
    # Update config path to point to correct location
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    
    def load_config():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    config = load_config()
    client = Client(config['binance']['api_key'], config['binance']['secret_key'])

    symbol = "ETHUSDT"
    interval = Client.KLINE_INTERVAL_15MINUTE

    state = load_state(symbol)
    in_position = state.get("in_position", False)
    entry_price = state.get("entry_price", 0)
    last_trade_time = state.get("last_trade_time", 0)
    win_count = state.get("win_count", 0)
    total_trades = state.get("total_trades", 0)
    COOLDOWN_SECONDS = config.get('eth_bot', {}).get('cooldown', 180)

    logger.info("🟢 ETH bot started.")

    def fetch_data():
        klines = retry_binance_call(lambda: client.get_klines(symbol=symbol, interval=interval, limit=100))
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df

    while not stop_event.is_set():
        try:
            config = load_config()
            quantity = config.get('eth_bot', {}).get('quantity', 0.05)
            tp_pct = config.get('eth_bot', {}).get('tp_pct', 0.04)
            sl_pct = config.get('eth_bot', {}).get('sl_pct', 0.03)
            mode = config.get('trading', {}).get('mode', 'test')

            df = fetch_data()
            entry_condition, exit_condition, price_now = eth_strategy(df)

            now = time.time()
            if not in_position and entry_condition and now - last_trade_time > COOLDOWN_SECONDS:
                if mode == "live":
                    retry_binance_call(lambda: client.order_market_buy(symbol=symbol, quantity=quantity))
                else:
                    retry_binance_call(lambda: client.create_test_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity))
                in_position = True
                entry_price = price_now
                log_trade(symbol, 'buy', entry_price)
                logger.info(f"🟩 BUY at {entry_price}")

                telegram_alert(f"🟢 *{symbol} BUY* at `${entry_price}`")
                
                save_state(symbol, {
                    "in_position": True,
                    "entry_price": entry_price,
                    "last_trade_time": time.time(),
                    "win_count": win_count,
                    "total_trades": total_trades
                })

            elif in_position:
                tp = entry_price * (1 + tp_pct)
                sl = entry_price * (1 - sl_pct)

                if (exit_condition or price_now >= tp or price_now <= sl) and time.time() - last_trade_time > COOLDOWN_SECONDS:
                    if mode == "live":
                        retry_binance_call(lambda: client.order_market_sell(symbol=symbol, quantity=quantity))
                    else:
                        retry_binance_call(lambda: client.create_test_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity))
                    in_position = False
                    pnl = price_now - entry_price
                    win = pnl > 0
                    win_count += 1 if win else 0
                    total_trades += 1
                    log_trade(symbol, 'sell', price_now, pnl)
                    logger.info(f"🟥 SELL at {price_now} | PnL: {round(pnl, 3)} | Win Rate: {win_count}/{total_trades}")

                    telegram_alert(
                        f"🔴 *{symbol} SELL* at `${price_now}`\n"
                        f"PnL: `${round(pnl, 3)}`\n"
                        f"Win Rate: `{win_count}/{total_trades}`"
                    )
                    
                    save_state(symbol, {
                        "in_position": False,
                        "entry_price": 0,
                        "last_trade_time": time.time(),
                        "win_count": win_count,
                        "total_trades": total_trades
                    })
                else:
                    logger.info("🕐 Holding position...")

        except Exception as e:
            import traceback
            logger.error(f"❌ Exception: {type(e).__name__} - {str(e)}")
            logger.error(traceback.format_exc())

        # Exit early if stop requested
        for _ in range(60 * 15):
            if stop_event.is_set():
                break
            time.sleep(1)

    logger.info("🛑 ETH bot stopped.")
