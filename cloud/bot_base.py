# bot_base.py
import requests
import yaml
import os
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))
TRADE_LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs', 'trade_logs.txt'))

# Add path to liveBackup folder
LIVE_BACKUP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'liveBackup'))
os.makedirs(LIVE_BACKUP_DIR, exist_ok=True)

# Ensure logs directory exists
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

# Supported coins
SUPPORTED_COINS = ['ETH', 'LINK', 'DOGE', 'MATIC']

# Load or reload config
def load_config():
    """Load the configuration file"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Config loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        # Return a minimal default config
        return {
            "trading": {"mode": "test"},
            "alerts": {"telegram": {"enabled": False}}
        }

# Setup logger for each bot
def get_logger(name):
    """Set up a logger for a specific bot"""
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    bot_logger = logging.getLogger(name)
    bot_logger.setLevel(logging.INFO)
    
    # Check if handler already exists
    if not bot_logger.handlers:
        handler = logging.FileHandler(os.path.join(log_dir, f'{name}_live.log'))
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        bot_logger.addHandler(handler)
        
    return bot_logger

# Log trade with timestamp and update CSV files
def log_trade(symbol, action, price, pnl=None):
    """
    Log a trade to both text logs and CSV files
    
    Args:
        symbol (str): Trading pair symbol (e.g., "ETHUSDT")
        action (str): Trade action ("buy" or "sell")
        price (float): Trade price
        pnl (float, optional): Profit/loss for sell trades
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write to traditional log file
        with open(TRADE_LOG_PATH, 'a') as f:
            pnl_str = f" | PnL: {round(pnl, 4)}" if pnl is not None else ""
            f.write(f"{timestamp} | {symbol} | {action.upper()} | Price: {price}{pnl_str}\n")
            
        # Log to CSV files in liveBackup
        coin = symbol.replace('USDT', '')
        
        # Skip if not one of our supported coins
        if coin not in SUPPORTED_COINS:
            logger.warning(f"Unsupported coin: {coin}, not logging to CSV")
            return
            
        trades_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_after_tariff_trades.csv")
        metrics_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_after_tariff_metrics.csv")
        
        # Create dataframe if file doesn't exist
        if not os.path.exists(trades_file):
            trades_df = pd.DataFrame(columns=['timestamp', 'type', 'price', 'profit'])
        else:
            trades_df = pd.read_csv(trades_file)
        
        # Add new trade
        new_trade = {
            'timestamp': timestamp,
            'type': action.upper(),
            'price': price,
            'profit': pnl if pnl is not None else None
        }
        trades_df = pd.concat([trades_df, pd.DataFrame([new_trade])], ignore_index=True)
        trades_df.to_csv(trades_file, index=False)
        
        # Update metrics if this is a SELL (completed trade)
        if action.upper() == 'SELL' and pnl is not None:
            if not os.path.exists(metrics_file):
                metrics_df = pd.DataFrame(columns=['coin', 'final_capital', 'total_trades', 'win_rate', 'total_profit'])
                metrics_df.loc[0] = {
                    'coin': coin,
                    'final_capital': 1000.0 + pnl,  # Assuming starting capital of 1000
                    'total_trades': 1,
                    'win_rate': 1.0 if pnl > 0 else 0.0,
                    'total_profit': pnl
                }
            else:
                metrics_df = pd.read_csv(metrics_file)
                
                # Get current values
                current_capital = metrics_df.loc[0, 'final_capital']
                current_trades = metrics_df.loc[0, 'total_trades']
                current_win_rate = metrics_df.loc[0, 'win_rate']
                current_profit = metrics_df.loc[0, 'total_profit']
                
                # Update metrics
                new_capital = current_capital + pnl
                new_trades = current_trades + 1
                
                # Calculate new win rate
                win_trades = current_win_rate * current_trades
                win_trades += 1 if pnl > 0 else 0
                new_win_rate = win_trades / new_trades
                
                new_profit = current_profit + pnl
                
                # Update dataframe
                metrics_df.loc[0] = {
                    'coin': coin,
                    'final_capital': new_capital,
                    'total_trades': new_trades,
                    'win_rate': new_win_rate,
                    'total_profit': new_profit
                }
            
            metrics_df.to_csv(metrics_file, index=False)
            
        logger.info(f"Trade logged: {symbol} {action.upper()} at {price}" + (f" with PnL: {pnl}" if pnl is not None else ""))
        
    except Exception as e:
        logger.error(f"Error logging trade: {str(e)}")
        # Ensure we at least log to console if there's an error
        print(f"[TRADE LOG ERROR] {symbol} {action.upper()} at {price}" + (f" with PnL: {pnl}" if pnl is not None else ""))

def generate_trade_chart(symbol, days=30):
    """
    Generate a trade performance chart for a specific coin
    
    Args:
        symbol (str): Coin symbol (e.g., "ETH")
        days (int): Number of days to include
        
    Returns:
        bytes: PNG image data as bytes
    """
    try:
        coin = symbol.replace('USDT', '')
        trades_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_after_tariff_trades.csv")
        
        if not os.path.exists(trades_file):
            logger.warning(f"No trade data found for {coin}")
            return None
            
        trades_df = pd.read_csv(trades_file)
        
        if trades_df.empty:
            logger.warning(f"Empty trade data for {coin}")
            return None
            
        # Convert timestamp to datetime
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Filter by date range
        start_date = datetime.now() - pd.Timedelta(days=days)
        filtered_df = trades_df[trades_df['timestamp'] >= start_date]
        
        if filtered_df.empty:
            logger.warning(f"No recent trades for {coin} in the last {days} days")
            return None
        
        # Extract sell trades with profit
        sell_trades = filtered_df[(filtered_df['type'] == 'SELL') & (~filtered_df['profit'].isna())]
        
        if sell_trades.empty:
            logger.warning(f"No sell trades with profit data for {coin}")
            return None
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Calculate cumulative profit
        sell_trades['cumulative_profit'] = sell_trades['profit'].cumsum()
        
        # Plot profit line
        plt.plot(sell_trades['timestamp'], sell_trades['cumulative_profit'], 'b-', linewidth=2)
        plt.fill_between(sell_trades['timestamp'], 0, sell_trades['cumulative_profit'], alpha=0.3, color='blue')
        
        # Format the chart
        plt.title(f'{coin} Profit/Loss Over Time (Last {days} Days)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Profit ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return None

def telegram_alert(message):
    """Send an alert message to Telegram"""
    try:
        config = load_config()
        alert_cfg = config.get("alerts", {}).get("telegram", {})
        if not alert_cfg.get("enabled", False):
            return

        token = alert_cfg.get("token")
        chat_id = alert_cfg.get("chat_id")
        
        if not token or not chat_id:
            logger.warning("Telegram token or chat_id missing")
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        logger.info(f"Telegram alert sent: {message[:50]}...")
        
    except Exception as e:
        logger.error(f"[TELEGRAM ALERT ERROR] {e}")
        print(f"[TELEGRAM ALERT ERROR] {e}")

def retry_binance_call(fn, retries=3, delay=2):
    """Retry a Binance API call with exponential backoff"""
    last_exception = None
    
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            wait_time = delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Binance API call failed (attempt {attempt+1}/{retries}): {str(e)}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    logger.error(f"Binance API call failed after {retries} attempts: {str(last_exception)}")
    raise last_exception

def get_coin_status(coin_symbol):
    """Get trading status for a specific coin"""
    try:
        coin = coin_symbol.upper().replace('USDT', '')
        
        # Get state file path (from state_utils.py logic)
        state_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'state'))
        state_path = os.path.join(state_dir, f"{coin.lower()}_state.json")
        
        if not os.path.exists(state_path):
            return {
                "coin": coin,
                "in_position": False,
                "current_price": None,
                "entry_price": 0,
                "last_trade_time": None,
                "win_rate": 0,
                "total_trades": 0,
                "state_file": "Not found"
            }
            
        # Read state file
        import json
        with open(state_path, 'r') as f:
            state = json.load(f)
            
        # Get metrics from CSV
        metrics_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_after_tariff_metrics.csv")
        win_rate = 0
        total_profit = 0
        
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            if not metrics_df.empty:
                win_rate = metrics_df.loc[0, 'win_rate'] * 100
                total_profit = metrics_df.loc[0, 'total_profit']
        
        # Format last trade time
        last_trade_time = datetime.fromtimestamp(state.get("last_trade_time", 0)).strftime('%Y-%m-%d %H:%M:%S') if state.get("last_trade_time", 0) > 0 else "Never"
        
        return {
            "coin": coin,
            "in_position": state.get("in_position", False),
            "entry_price": state.get("entry_price", 0),
            "last_trade_time": last_trade_time,
            "win_rate": win_rate,
            "total_trades": state.get("total_trades", 0),
            "total_profit": total_profit,
            "state_file": "Found"
        }
        
    except Exception as e:
        logger.error(f"Error getting coin status: {str(e)}")
        return {
            "coin": coin_symbol.upper().replace('USDT', ''),
            "error": str(e)
        }

def get_system_status():
    """Get overall system status"""
    try:
        # Import here to avoid circular imports
        from live_trading_manager import bot_threads, stop_events
        
        status = {
            "bots": {},
            "system": {
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "config_file": os.path.exists(CONFIG_PATH)
            }
        }
        
        # Check each bot's status
        for coin in ['eth', 'link', 'doge', 'matic']:
            is_running = coin in bot_threads and bot_threads[coin].is_alive()
            stop_requested = coin in stop_events and stop_events[coin].is_set()
            
            status["bots"][coin] = {
                "running": is_running,
                "stop_requested": stop_requested,
                **get_coin_status(coin)
            }
            
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {"error": str(e)}

# Initialize on import
CONFIG = load_config()