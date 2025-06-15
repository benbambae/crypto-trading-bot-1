# telegram_bot.py

import os
import yaml
import requests
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
import traceback
from collections import deque
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown
from telegram.error import BadRequest
from io import BytesIO
import reddit_sentiment

# Update config path to point to correct location
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize global variables
CONFIG = {}
last_tx_ids = deque(maxlen=500)
BOT_TOKEN = None
ALLOWED_CHAT_ID = None
whale_enabled = True
WH_API_KEY = None
MIN_VALUE = 1000000
# Track if we're showing before or after tariff data
show_before_tariff = False  # Default to after tariff view

# Load configuration
try:
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
    BOT_TOKEN = CONFIG.get('alerts', {}).get('telegram', {}).get('token')
    ALLOWED_CHAT_ID = str(CONFIG.get('alerts', {}).get('telegram', {}).get('chat_id'))
    whale_enabled = CONFIG.get("whaleAlert", {}).get("enabled", True)
    WH_API_KEY = CONFIG.get("whaleAlert", {}).get("api_key")
    MIN_VALUE = CONFIG.get("whaleAlert", {}).get("min_value_usd", 10000)
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

# Add path to liveBackup folder
LIVE_BACKUP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'liveBackup'))
os.makedirs(LIVE_BACKUP_DIR, exist_ok=True)

# Define supported coins
SUPPORTED_COINS = ['ETH', 'LINK', 'DOGE', 'ARB']


def auth(func):
    """Restrict use to the configured chat ID."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != ALLOWED_CHAT_ID:
            # Use safe_reply (works for both commands & buttons)
            await safe_reply(update, "⛔ Unauthorized.")
            return
        return await func(update, context)
    return wrapper


@auth
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command to display welcome message and main menu"""
    main_menu_keyboard = [
            [InlineKeyboardButton("📊 Metrics", callback_data="metrics_menu"),
            InlineKeyboardButton("🧾 Logs", callback_data="logs_menu")],
            [InlineKeyboardButton("🐋 Whale Alerts", callback_data="whale_menu"),
            InlineKeyboardButton("📝 Sentiment", callback_data="sentiment_menu")],  # Add this line
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu"),
            InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
    
    
    markup = InlineKeyboardMarkup(main_menu_keyboard)
    
    welcome_text = (
        "🤖 *Trading Bot Dashboard*\n\n"
        "Welcome to your crypto trading bot management interface\\.\n"
        "Please select an option from the menu below:"
    )
    
    await safe_reply(update,
        welcome_text,
        reply_markup=markup,
        parse_mode="Markdown"
    )



@auth
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display help information with button navigation"""
    help_text = (
        "*📋 Available Commands:*\n\n"
        "/start \\- Open the main menu\n"
        "/help \\- Show this help message\n"
        "/status \\- Check bot system status\n"
        "/metrics \\[COIN\\] \\- Show trading metrics\n"
        "/logs \\[COIN\\] \\- Display recent trade logs\n"
        "/sentiment \\[COIN\\] \\- Analyze Reddit sentiment\n"
        "/config \\- Show current configuration\n"
        "/whale on\\|off \\- Toggle whale alerts\n"
        "/restart \\[COIN\\] \\- Restart a specific bot\n"
        "/stop \\[COIN\\] \\- Stop a specific bot\n"
        "/reload \\- Reload configuration\n\n"
        "You can also use the interactive menu with /start"
    )
    
    buttons = [[InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]]
    markup = InlineKeyboardMarkup(buttons)
    
    await safe_reply(update,
        help_text,
        reply_markup=markup,
        parse_mode="Markdown"
    )

@auth
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot status with more details"""
    global whale_enabled
    try:
        # Get status of all bots
        from live_trading_manager import bot_threads
        
        status_text = ["🖥️ *Bot System Status*\n"]
        
        # Check if each bot is running
        for coin in ['eth', 'link', 'doge', 'arb']:
            is_running = coin in bot_threads and bot_threads[coin].is_alive()
            status_emoji = "✅" if is_running else "❌"
            status_text.append(f"{status_emoji} {coin.upper()}: {'Running' if is_running else 'Stopped'}")
        
        # Add overall system info
        status_text.append("\n🔄 System Info:")
        status_text.append(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        status_text.append(f"🐋 Whale Alerts: {'Enabled' if whale_enabled else 'Disabled'}")
        
        # Check if config file exists
        status_text.append(f"⚙️ Config File: {'Found' if os.path.exists(config_path) else 'Missing'}")
        
        buttons = [[InlineKeyboardButton("🔄 Refresh Status", callback_data="refresh_status"),
                    InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
        markup = InlineKeyboardMarkup(buttons)
        
        status_text_formatted = "\n".join(status_text).replace(".", "\\.")
        
        await safe_reply(update,
            status_text_formatted,
            reply_markup=markup,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in status_cmd: {e}")
        error_msg = f"⚠️ Error checking status: {str(e)}".replace(".", "\\.")
        await safe_reply(update, error_msg, parse_mode="Markdown")

@auth
async def config_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show configuration with redacted sensitive info"""
    try:
        # Make a copy of config to redact sensitive information
        safe_config = dict(CONFIG)
        
        # Redact API keys and sensitive info
        if 'binance' in safe_config:
            if 'api_key' in safe_config['binance']:
                safe_config['binance']['api_key'] = '****' + safe_config['binance']['api_key'][-4:] if safe_config['binance']['api_key'] else '****'
            if 'secret_key' in safe_config['binance']:
                safe_config['binance']['secret_key'] = '****************'
        
        if 'whaleAlert' in safe_config and 'api_key' in safe_config['whaleAlert']:
            safe_config['whaleAlert']['api_key'] = '****' + safe_config['whaleAlert']['api_key'][-4:] if safe_config['whaleAlert']['api_key'] else '****'
        
        if 'alerts' in safe_config and 'telegram' in safe_config['alerts'] and 'token' in safe_config['alerts']['telegram']:
            safe_config['alerts']['telegram']['token'] = '****' + safe_config['alerts']['telegram']['token'][-4:] if safe_config['alerts']['telegram']['token'] else '****'
        
        config_preview = yaml.dump(safe_config, default_flow_style=False)
        
        # Split into chunks if too long
        if len(config_preview) > 4000:
            chunks = [config_preview[i:i+4000] for i in range(0, len(config_preview), 4000)]
            for i, chunk in enumerate(chunks):
                chunk_text = f"🛠️ Config (Part {i+1}/{len(chunks)}):\n{chunk}"
                chunk_text = chunk_text.replace(".", "\\.").replace("-", "\\-").replace("+", "\\+")
                await safe_reply(update,
                    chunk_text,
                    parse_mode="Markdown"
                )
        else:
            config_text = f"🛠️ Current Config:\n{config_preview}"
            config_text = config_text.replace(".", "\\.").replace("-", "\\-").replace("+", "\\+")
            await safe_reply(update,
                config_text,
                parse_mode="Markdown"
            )
            
        # Add navigation buttons
        buttons = [[InlineKeyboardButton("🔄 Reload Config", callback_data="reload_config"),
                    InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
        markup = InlineKeyboardMarkup(buttons)
        
        await safe_reply(update,
            "Use buttons below to reload config or return to menu:",
            reply_markup=markup
        )
    except Exception as e:
        logger.error(f"Error in config_cmd: {e}")
        await safe_reply(update, f"⚠️ Error displaying config: {str(e)}")

@auth
async def metrics_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display trading metrics for all coins or a specific coin"""
    try:
        coin = context.args[0].upper() if context.args else None
        await send_metrics(update, context, coin)
    except Exception as e:
        logger.error(f"Error in metrics_cmd: {e}")
        error_msg = f"⚠️ Error fetching metrics: {str(e)}".replace(".", "\\.")
        await safe_reply(update, error_msg, parse_mode="Markdown")
        
async def send_metrics(update: Update, context: ContextTypes.DEFAULT_TYPE, coin=None, is_callback=False):
    """Helper function to send metrics with proper formatting and buttons"""
    try:
        global show_before_tariff
        tariff_type = "before" if show_before_tariff else "after"
        tariff_label = "Before" if show_before_tariff else "After"
        
        message = []
        if coin:
            # Get specific coin metrics
            metrics_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_metrics.csv")
            trades_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_trades.csv")
            
            if not os.path.exists(metrics_file) or not os.path.exists(trades_file):
                buttons = [
                    [InlineKeyboardButton("🔙 Back to Metrics", callback_data="metrics_menu")]
                ]
                markup = InlineKeyboardMarkup(buttons)
                
                # Use different methods based on callback or command
                if is_callback:
                    query = update.callback_query
                    await query.message.edit_text(
                        f"❌ No metrics found for {coin} ({tariff_label} Tariff)",
                        reply_markup=markup
                    )
                else:
                    await safe_reply(update,
                        f"❌ No metrics found for {coin} ({tariff_label} Tariff)",
                        reply_markup=markup
                    )
                return
                
            metrics_df = pd.read_csv(metrics_file)
            trades_df = pd.read_csv(trades_file)
            
            total_trades = metrics_df['total_trades'].iloc[0]
            win_rate = metrics_df['win_rate'].iloc[0] * 100
            total_profit = metrics_df['total_profit'].iloc[0]
            final_capital = metrics_df['final_capital'].iloc[0]
            
            # Add initial capital if available, otherwise estimate it
            if 'initial_capital' in metrics_df.columns:
                initial_capital = metrics_df['initial_capital'].iloc[0]
            else:
                # Fallback calculation if initial_capital isn't stored
                initial_capital = final_capital - total_profit
            
            message.append(f"📊 *{coin} Trading Metrics \\({tariff_label} Tariff\\)*\n")
            message.append(f"💼 Initial Capital: *${initial_capital:.2f}*")
            message.append(f"💰 Final Capital: *${final_capital:.2f}*")
            message.append(f"📈 Total Profit: *${total_profit:.2f}*")
            message.append(f"🔄 Total Trades: *{total_trades}*")
            message.append(f"✅ Win Rate: *{win_rate:.2f}%*")
            
            # Calculate ROI
            if initial_capital > 0:
                roi = (total_profit / initial_capital) * 100
                message.append(f"📊 ROI: *{roi:.2f}%*")
            
            # Calculate average profit per trade
            if total_trades > 0:
                avg_profit = total_profit / total_trades
                message.append(f"📊 Avg Profit/Trade: *${avg_profit:.2f}*")
            
            # Add recent trades
            if not trades_df.empty:
                recent_trades = trades_df.tail(5)
                message.append("\n*Recent Trades:*")
                for _, row in recent_trades.iterrows():
                    trade_type = row['type']
                    price = row['price']
                    profit = row['profit'] if 'profit' in row and not pd.isna(row['profit']) else 0
                    timestamp = row['timestamp']
                    emoji = "🟢" if trade_type == "BUY" else "🔴"
                    profit_str = f" \\(PnL: ${profit:.2f}\\)" if trade_type == "SELL" else ""
                    message.append(f"{emoji} {timestamp} \\- {trade_type} at ${price:.2f}{profit_str}")
            
            # Add navigation buttons with tariff toggle
            tariff_toggle_label = "Switch to Before Tariff" if not show_before_tariff else "Switch to After Tariff"
            
            buttons = [
                [InlineKeyboardButton("🧾 Show Logs", callback_data=f"logs_{coin}")],
                [InlineKeyboardButton(f"🔄 {tariff_toggle_label}", callback_data="toggle_tariff")],
                [InlineKeyboardButton("🔙 All Coins", callback_data="metrics_menu")]
            ]
            markup = InlineKeyboardMarkup(buttons)
            
            formatted_message = "\n".join(message)
            
            # Use different methods based on callback or command
            if is_callback:
                query = update.callback_query
                await query.message.edit_text(
                    formatted_message,
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
            else:
                await safe_reply(update,
                    formatted_message,
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
            
        else:
            # Get all coins metrics - PORTFOLIO VIEW
            message.append(f"📊 *Overall Portfolio Metrics ({tariff_label} Tariff)*\n")
            
            # Initialize variables
            total_initial_capital = 0
            total_final_capital = 0 
            total_profit = 0
            all_metrics = []
            
            for coin in SUPPORTED_COINS:
                metrics_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_metrics.csv")
                if os.path.exists(metrics_file):
                    metrics_df = pd.read_csv(metrics_file)
                    win_rate = metrics_df['win_rate'].iloc[0] * 100
                    coin_profit = metrics_df['total_profit'].iloc[0]
                    final_capital = metrics_df['final_capital'].iloc[0]
                    total_trades = metrics_df['total_trades'].iloc[0]
                    
                    # Get initial capital if available, otherwise calculate it
                    if 'initial_capital' in metrics_df.columns:
                        initial_capital = metrics_df['initial_capital'].iloc[0]
                    else:
                        # Fallback calculation if initial_capital isn't stored
                        initial_capital = final_capital - coin_profit
                    
                    # Update totals
                    total_initial_capital += initial_capital
                    total_final_capital += final_capital
                    # We'll calculate true profit at the end
                    
                    all_metrics.append({
                        'coin': coin,
                        'win_rate': win_rate,
                        'profit': coin_profit,
                        'capital': final_capital,
                        'initial_capital': initial_capital,
                        'trades': total_trades
                    })
            
            # Calculate true total profit to avoid double counting
            total_profit = total_final_capital - total_initial_capital
            
            # Sort by profit
            all_metrics.sort(key=lambda x: x['profit'], reverse=True)
            
            for metric in all_metrics:
                profit_emoji = "🟢" if metric['profit'] > 0 else "🔴"
                # Calculate ROI for each coin
                coin_roi = (metric['profit'] / metric['initial_capital']) * 100 if metric['initial_capital'] > 0 else 0
                message.append(f"{profit_emoji} *{metric['coin']}*: Win Rate {metric['win_rate']:.1f}%, Profit ${metric['profit']:.2f} \\(ROI: {coin_roi:.1f}%\\)")
            
            # Portfolio summary with both initial and final capital
            message.append(f"\n💼 *Initial Portfolio Capital: ${total_initial_capital:.2f}*")
            message.append(f"💰 *Current Portfolio Value: ${total_final_capital:.2f}*")
            message.append(f"📈 *Total Portfolio Profit: ${total_profit:.2f}*")
            
            # Calculate and display overall portfolio ROI
            if total_initial_capital > 0:
                portfolio_roi = (total_profit / total_initial_capital) * 100
                message.append(f"📊 *Portfolio ROI: {portfolio_roi:.2f}%*")
            
            # Add tariff toggle button
            tariff_toggle_label = "Switch to Before Tariff" if not show_before_tariff else "Switch to After Tariff"
            
            # Create buttons for each coin
            buttons = []
            for coin in SUPPORTED_COINS:
                buttons.append([InlineKeyboardButton(f"{coin} Details", callback_data=f"metrics_{coin}")])
            
            buttons.append([InlineKeyboardButton(f"🔄 {tariff_toggle_label}", callback_data="toggle_tariff")])
            buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
            
            markup = InlineKeyboardMarkup(buttons)
            
            formatted_message = "\n".join(message)
            
            # Use different methods based on callback or command
            if is_callback:
                query = update.callback_query
                await query.message.edit_text(
                    formatted_message,
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
            else:
                await safe_reply(update,
                    formatted_message,
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
            
    except Exception as e:
        logger.error(f"Error in send_metrics: {str(e)}\n{traceback.format_exc()}")
        error_message = f"⚠️ Error displaying metrics: {str(e)}"
        
        if is_callback:
            query = update.callback_query
            await query.message.reply_text(error_message)
        else:
            await safe_reply(update, error_message)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses from inline keyboards"""
    global whale_enabled
    global show_before_tariff
    query = update.callback_query
    await query.answer()  # Acknowledge the button press
    
    callback_data = query.data
    
    try:
        # Main menu navigation
        if callback_data == "main_menu":
            main_menu_keyboard = [
                [InlineKeyboardButton("📊 Metrics", callback_data="metrics_menu"),
                 InlineKeyboardButton("🧾 Logs", callback_data="logs_menu")],
                [InlineKeyboardButton("🐋 Whale Alerts", callback_data="whale_menu"),
                 InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu")],
                [InlineKeyboardButton("❓ Help", callback_data="help")]
            ]
            
            markup = InlineKeyboardMarkup(main_menu_keyboard)
            
            await safe_edit(
                query.message,
                "🤖 *Trading Bot Dashboard*\n\nPlease select an option from the menu below:",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Help section
        elif callback_data == "help":
            help_text = (
                "*📋 Available Commands:*\n\n"
                "/start \\- Open the main menu\n"
                "/help \\- Show this help message\n"
                "/status \\- Check bot system status\n"
                "/metrics \\[COIN\\] \\- Show trading metrics\n"
                "/logs \\[COIN\\] \\- Display recent trade logs\n"
                "/config \\- Show current configuration\n"
                "/whale on\\|off \\- Toggle whale alerts\n"
                "/restart \\[COIN\\] \\- Restart a specific bot\n"
                "/stop \\[COIN\\] \\- Stop a specific bot\n"
                "/reload \\- Reload configuration\n\n"
                "You can also use the interactive menu with /start"
            )
            
            buttons = [[InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]]
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                help_text,
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Metrics menu
        elif callback_data == "metrics_menu":
            # Show metrics selection menu
            buttons = []
            for coin in SUPPORTED_COINS:
                buttons.append([InlineKeyboardButton(f"{coin} Metrics", callback_data=f"metrics_{coin}")])
            
            buttons.append([InlineKeyboardButton("📊 Overall Portfolio", callback_data="metrics_overall")])
            
            # Add tariff toggle button
            tariff_toggle_label = (
                "Switch to Before Tariff" if not show_before_tariff else "Switch to After Tariff"
            )
            buttons.append(
                [InlineKeyboardButton(f"🔄 {tariff_toggle_label}", callback_data="toggle_tariff")]
            )
            
            buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
            
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "📊 *Metrics Dashboard*\n\nSelect a coin to view detailed metrics:",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Logs menu
        elif callback_data == "logs_menu":
            # Show logs selection menu
            buttons = []
            for coin in SUPPORTED_COINS:
                buttons.append([InlineKeyboardButton(f"{coin} Logs", callback_data=f"logs_{coin}")])
            
            # Add tariff toggle button
            tariff_toggle_label = (
                "Switch to Before Tariff" if not show_before_tariff else "Switch to After Tariff"
            )
            buttons.append(
                [InlineKeyboardButton(f"🔄 {tariff_toggle_label}", callback_data="toggle_tariff")]
            )
                        
            buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
            
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "🧾 *Trading Logs*\n\nSelect a coin to view its recent trades:",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Handle tariff toggle
        elif callback_data == "toggle_tariff":
            show_before_tariff = not show_before_tariff
            
            tariff_type = "Before" if show_before_tariff else "After"
            
            # Get the original callback data to determine which view to return to
            original_view = None
            message_text = query.message.text or query.message.caption or ""
            
            if "Metrics Dashboard" in message_text:
                original_view = "metrics_menu"
            elif "Trading Logs" in message_text:
                original_view = "logs_menu"
            elif message_text.startswith("📊 *") and "_Metrics" in message_text:
                # For individual coin metrics view
                coin = message_text.split()[1].replace("*", "")
                original_view = f"metrics_{coin}"
            elif message_text.startswith("🧾 *Last") and "trades" in message_text:
                # For logs view
                coin = message_text.split()[3]
                original_view = f"logs_{coin}"
            else:
                # Default to main menu
                original_view = "main_menu"
            
            # Send notification about toggle
            await query.message.reply_text(
                f"📊 Switched to *{tariff_type} Tariff* view",
                parse_mode="Markdown"
            )
            
            # Call the appropriate handler based on where we were
            if original_view == "metrics_menu":
                # Show metrics selection menu
                buttons = []
                for coin in SUPPORTED_COINS:
                    buttons.append([InlineKeyboardButton(f"{coin} Metrics", callback_data=f"metrics_{coin}")])
                
                buttons.append([InlineKeyboardButton("📊 Overall Portfolio", callback_data="metrics_overall")])
                
                # Add tariff toggle button
                tariff_status = "Before Tariff" if not show_before_tariff else "After Tariff"
                buttons.append([InlineKeyboardButton(f"🔄 Switch to {tariff_status}", callback_data="toggle_tariff")])
                
                buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
                
                markup = InlineKeyboardMarkup(buttons)
                
                await safe_edit(
                    query.message,
                    "📊 *Metrics Dashboard*\n\nSelect a coin to view detailed metrics:",
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
            elif original_view.startswith("metrics_"):
                # Handle metrics for specific coin or overall
                coin = original_view.split("_")[1] if original_view != "metrics_overall" else None
                await send_metrics(update, context, coin, is_callback=True)
            elif original_view == "logs_menu":
                # Show logs selection menu
                buttons = []
                for coin in SUPPORTED_COINS:
                    buttons.append([InlineKeyboardButton(f"{coin} Logs", callback_data=f"logs_{coin}")])
                
                # Add tariff toggle button
                tariff_status = "Before Tariff" if not show_before_tariff else "After Tariff"
                buttons.append([InlineKeyboardButton(f"🔄 Switch to {tariff_status}", callback_data="toggle_tariff")])
                
                buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
                
                markup = InlineKeyboardMarkup(buttons)
                
                await safe_edit(
                    query.message,
                    "🧾 *Trading Logs*\n\nSelect a coin to view its recent trades:",
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
            elif original_view.startswith("logs_"):
                # Handle logs for specific coin
                coin = original_view.split("_")[1]
                await send_logs_callback(update, context, coin)
            else:
                # Safe fallback to main menu without recursion
                await safe_edit(
                    query.message,
                    "🤖 *Trading Bot Dashboard*\n\nPlease select an option from the menu below:",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("📊 Metrics", callback_data="metrics_menu"),
                        InlineKeyboardButton("🧾 Logs", callback_data="logs_menu")],
                        [InlineKeyboardButton("🐋 Whale Alerts", callback_data="whale_menu"),
                        InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu")],
                        [InlineKeyboardButton("❓ Help", callback_data="help")]
                    ]),
                    parse_mode="Markdown"
                )

                # Sentiment Analysis Menu
        elif callback_data == "sentiment_menu":
            # Show sentiment analysis selection menu
            buttons = []
            for coin in SUPPORTED_COINS:
                buttons.append([InlineKeyboardButton(f"{coin} Sentiment", callback_data=f"sentiment_{coin}")])
            
            buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "📝 *Reddit Sentiment Analysis*\n\nSelect a coin to view market sentiment from Reddit:",
                reply_markup=markup,
                parse_mode="Markdown"
            )

        # Handle sentiment for specific coin and timeframe
        elif callback_data.startswith("sentiment_"):
            parts = callback_data.split("_")
            coin = parts[1] if len(parts) > 1 else None
            
            # Check if we have a timeframe specified
            timeframe = parts[2] if len(parts) > 2 else "week"
            force_refresh = timeframe == "refresh"
            
            if force_refresh:
                timeframe = "week"  # Default when refreshing
            
            if coin:
                # We need to delete the original message first
                await query.message.delete()
                
                # Then send a new sentiment analysis
                await send_sentiment_analysis(
                    update, 
                    context, 
                    coin, 
                    timeframe=timeframe, 
                    is_callback=True
                )
            else:
                await send_sentiment_menu(update, context)    
        # Whale alerts menu
        elif callback_data == "whale_menu":
            # Show whale alert status and controls
            status = "ON" if whale_enabled else "OFF"
            
            buttons = [
                [InlineKeyboardButton("🟢 Turn ON", callback_data="whale_on"),
                 InlineKeyboardButton("🔴 Turn OFF", callback_data="whale_off")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ]
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                f"🐋 *Whale Alerts*\n\nCurrent Status: *{status}*\n\nWhale alerts notify you of large transactions for your tracked coins\\.",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Settings menu
        elif callback_data == "settings_menu":
            # Show settings menu
            buttons = [
                [InlineKeyboardButton("⚙️ View Config", callback_data="view_config"),
                 InlineKeyboardButton("🔄 Reload Config", callback_data="reload_config")],
                [InlineKeyboardButton("🟢 Start Bots", callback_data="start_bots_menu"),
                 InlineKeyboardButton("🛑 Stop Bots", callback_data="stop_bots_menu")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ]
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "⚙️ *Settings*\n\nManage your trading bot configuration and operation:",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Start bots menu
        elif callback_data == "start_bots_menu":
            buttons = []
            for coin in ['eth', 'link', 'doge', 'arb']:
                buttons.append([InlineKeyboardButton(f"Start {coin.upper()}", callback_data=f"restart_{coin}")])
            
            buttons.append([InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")])
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "🟢 *Start Bots*\n\nSelect a bot to start or restart:",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Stop bots menu
        elif callback_data == "stop_bots_menu":
            buttons = []
            for coin in ['eth', 'link', 'doge', 'arb']:
                buttons.append([InlineKeyboardButton(f"Stop {coin.upper()}", callback_data=f"stop_{coin}")])
            
            buttons.append([InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")])
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "🛑 *Stop Bots*\n\nSelect a bot to stop:",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # View config
        elif callback_data == "view_config":
            # Make a copy of config to redact sensitive information
            safe_config = dict(CONFIG)
            
            # Redact API keys and sensitive info
            if 'binance' in safe_config:
                if 'api_key' in safe_config['binance']:
                    safe_config['binance']['api_key'] = '****' + safe_config['binance']['api_key'][-4:] if safe_config['binance']['api_key'] else '****'
                if 'secret_key' in safe_config['binance']:
                    safe_config['binance']['secret_key'] = '****************'
            
            if 'whaleAlert' in safe_config and 'api_key' in safe_config['whaleAlert']:
                safe_config['whaleAlert']['api_key'] = '****' + safe_config['whaleAlert']['api_key'][-4:] if safe_config['whaleAlert']['api_key'] else '****'
            
            if 'alerts' in safe_config and 'telegram' in safe_config['alerts'] and 'token' in safe_config['alerts']['telegram']:
                safe_config['alerts']['telegram']['token'] = '****' + safe_config['alerts']['telegram']['token'][-4:] if safe_config['alerts']['telegram']['token'] else '****'
            
            config_preview = yaml.dump(safe_config, default_flow_style=False)
            
            # Split into chunks if too long
            if len(config_preview) > 4000:
                await query.message.reply_text("Config is too large to display in a single message. Use /config command instead.")
                return
            
            # Escape characters for Markdown
            config_preview = config_preview.replace(".", "\\.").replace("-", "\\-").replace("+", "\\+")
                
            buttons = [
                [InlineKeyboardButton("🔄 Reload Config", callback_data="reload_config")],
                [InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")]
            ]
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                f"🛠️ *Current Config:*\n```\n{config_preview}\n```",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        # Reload config
        elif callback_data == "reload_config":
            try:
                # CONFIG should be declared as global at script start
                with open(config_path, 'r') as f:
                    CONFIG.clear()  # Clear existing config
                    CONFIG.update(yaml.safe_load(f))  # Update with new values
                
                buttons = [[InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")]]
                markup = InlineKeyboardMarkup(buttons)
                
                await safe_edit(
                    query.message,
                    "🔁 Config reloaded successfully\\.",
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Error reloading config: {e}")
                error_msg = f"⚠️ Error reloading config: {str(e)}".replace(".", "\\.")
                await safe_edit(
                    query.message,
                    error_msg,
                    parse_mode="Markdown"
                )
        # Handle metrics for specific coin
        elif callback_data.startswith("metrics_"):
            coin = callback_data.split("_")[1]
            
            if coin == "overall":
                # Send overall metrics
                await send_metrics(update, context, None, is_callback=True)
            else:
                # Send coin-specific metrics
                await send_metrics(update, context, coin, is_callback=True)
                
        # Handle logs for specific coin
        elif callback_data.startswith("logs_"):
            coin = callback_data.split("_")[1]
            await send_logs_callback(update, context, coin)
                
        # Handle whale alerts toggle
        elif callback_data == "whale_on":
            whale_enabled = True
            
            buttons = [
                [InlineKeyboardButton("🟢 Turn ON", callback_data="whale_on"),
                 InlineKeyboardButton("🔴 Turn OFF", callback_data="whale_off")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ]
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "🐋 *Whale Alerts*\n\nCurrent Status: *ON*\n\nWhale alerts notify you of large transactions for your tracked coins\\.",
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        elif callback_data == "whale_off":
            whale_enabled = False
            
            buttons = [
                [InlineKeyboardButton("🟢 Turn ON", callback_data="whale_on"),
                 InlineKeyboardButton("🔴 Turn OFF", callback_data="whale_off")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ]
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_edit(
                query.message,
                "🐋 *Whale Alerts*\n\nCurrent Status: *OFF*\n\nWhale alerts notify you of large transactions for your tracked coins\\.",
                reply_markup=markup,
                parse_mode="Markdown"
            )
                
        # Handle bot restart
        elif callback_data.startswith("restart_"):
            coin = callback_data.split("_")[1]
            
            try:
                from live_trading_manager import start_bot
                msg = start_bot(coin)
                
                buttons = [[InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")]]
                markup = InlineKeyboardMarkup(buttons)
                
                await safe_edit(
                    query.message,
                    f"♻️ {msg}",
                    reply_markup=markup
                )
            except Exception as e:
                logger.error(f"Error in restart button handler: {e}")
                await query.message.edit_text(f"⚠️ Restart error: {str(e)}")
                
        # Handle bot stop
        elif callback_data.startswith("stop_"):
            coin = callback_data.split("_")[1]
            
            try:
                from live_trading_manager import stop_bot
                msg = stop_bot(coin)
                
                buttons = [[InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")]]
                markup = InlineKeyboardMarkup(buttons)
                
                await safe_edit(
                    query.message,
                    f"🛑 {msg}",
                    reply_markup=markup
                )
            except Exception as e:
                logger.error(f"Error in stop button handler: {e}")
                await query.message.edit_text(f"⚠️ Stop error: {str(e)}")
                
        # Refresh status
        elif callback_data == "refresh_status":
            from live_trading_manager import bot_threads
            
            status_text = ["🖥️ *Bot System Status*\n"]
            
            # Check if each bot is running
            for coin in ['eth', 'link', 'doge', 'arb']:
                is_running = coin in bot_threads and bot_threads[coin].is_alive()
                status_emoji = "✅" if is_running else "❌"
                status_text.append(f"{status_emoji} {coin.upper()}: {'Running' if is_running else 'Stopped'}")
            
            # Add overall system info
            status_text.append("\n🔄 System Info:")
            status_text.append(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            status_text.append(f"🐋 Whale Alerts: {'Enabled' if whale_enabled else 'Disabled'}")
            
            # Check if config file exists
            status_text.append(f"⚙️ Config File: {'Found' if os.path.exists(config_path) else 'Missing'}")
            
            buttons = [[InlineKeyboardButton("🔄 Refresh Status", callback_data="refresh_status"),
                        InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
            markup = InlineKeyboardMarkup(buttons)
            
            status_text_formatted = "\n".join(status_text).replace(".", "\\.")
            
            await safe_edit(
                query.message,
                status_text_formatted,
                reply_markup=markup,
                parse_mode="Markdown"
            )
            
        else:
            logger.warning(f"Unknown callback data: {callback_data}")
            
    except Exception as e:
        logger.error(f"Error in button_handler: {str(e)}\n{traceback.format_exc()}")
        await query.message.reply_text(f"⚠️ Error processing button: {str(e)}")


async def send_logs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, coin):
    """Send formatted logs for a specific coin when triggered by callback"""
    try:
        global show_before_tariff
        tariff_type = "before" if show_before_tariff else "after"
        tariff_label = "Before" if show_before_tariff else "After"
        
        # Try to get logs from trades file
        trades_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_trades.csv")
        
        if not os.path.exists(trades_file):
            buttons = [[InlineKeyboardButton("🔙 Back to Logs Menu", callback_data="logs_menu")]]
            markup = InlineKeyboardMarkup(buttons)
            
            if update.callback_query:
                await update.callback_query.message.edit_text(
                    f"❌ No trade logs found for {coin} ({tariff_label} Tariff)",
                    reply_markup=markup
                )
            else:
                await safe_reply(update,
                    f"❌ No trade logs found for {coin} ({tariff_label} Tariff)",
                    reply_markup=markup
                )
            return
            
        # Read the trades file
        trades_df = pd.read_csv(trades_file)
        
        # Format the last 10 trades
        if trades_df.empty:
            if update.callback_query:
                await update.callback_query.message.edit_text(f"No trades found for {coin}")
            else:
                await safe_reply(update, f"No trades found for {coin}")
            return
            
        recent_trades = trades_df.tail(10)
        message = [f"🧾 *Last 10 {coin} trades \\({tariff_label} Tariff\\):*\n"]
        
        for _, row in recent_trades.iterrows():
            timestamp = row['timestamp']
            trade_type = row['type']
            price = row['price']
            profit = row['profit'] if 'profit' in row and not pd.isna(row['profit']) else None
            
            emoji = "🟢" if trade_type == "BUY" else "🔴"
            profit_str = f" \\| PnL: *${profit:.2f}*" if profit is not None else ""
            
            message.append(f"{emoji} {timestamp} \\| {trade_type} at *${price:.2f}*{profit_str}")
        
        # Add navigation buttons with tariff toggle
        tariff_toggle_label = "Switch to Before Tariff" if not show_before_tariff else "Switch to After Tariff"
        
        buttons = [
            [InlineKeyboardButton("📊 Metrics", callback_data=f"metrics_{coin}")],
            [InlineKeyboardButton(f"🔄 {tariff_toggle_label}", callback_data="toggle_tariff")],
            [InlineKeyboardButton("🔙 All Logs", callback_data="logs_menu")]
        ]
        markup = InlineKeyboardMarkup(buttons)
        
        formatted_message = "\n".join(message)
        
        if update.callback_query:
            await update.callback_query.message.edit_text(
                formatted_message,
                reply_markup=markup,
                parse_mode="Markdown"
            )
        else:
            await safe_reply(update,
                formatted_message,
                reply_markup=markup,
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error in send_logs_callback: {e}")
        if update.callback_query:
            await update.callback_query.message.reply_text(f"⚠️ Error displaying logs: {str(e)}")
        else:
            await safe_reply(update, f"⚠️ Error displaying logs: {str(e)}")

@auth
async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show recent trade logs for a specific coin"""
    try:
        coin = context.args[0].upper() if context.args else None
        if not coin:
            # Show logs menu if no coin specified
            buttons = []
            for coin in SUPPORTED_COINS:
                buttons.append([InlineKeyboardButton(f"{coin} Logs", callback_data=f"logs_{coin}")])
            
            buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
            markup = InlineKeyboardMarkup(buttons)
            
            await safe_reply(update,
                "Please select a coin to view logs:",
                reply_markup=markup
            )
            return
            
        await send_logs_callback(update, context, coin)
    except Exception as e:
        logger.error(f"Error in logs_cmd: {e}")
        error_msg = f"⚠️ Error reading logs: {str(e)}".replace(".", "\\.")
        await safe_reply(update, error_msg, parse_mode="Markdown")

@auth
async def restart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Restart a specific trading bot"""
    coin = context.args[0].lower() if context.args else None
    
    if not coin:
        # Show restart menu
        buttons = []
        for coin in ['eth', 'link', 'doge', 'arb']:
            buttons.append([InlineKeyboardButton(f"Restart {coin.upper()}", callback_data=f"restart_{coin}")])
        
        buttons.append([InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")])
        markup = InlineKeyboardMarkup(buttons)
        
        await safe_reply(update,
            "Select a bot to restart:",
            reply_markup=markup
        )
        return
    
    if coin not in ['eth', 'link', 'doge', 'arb']:
        await safe_reply(update,
            "⚠️ Usage: /restart eth|link|doge|arb")
        return
        
    try:
        from live_trading_manager import start_bot
        msg = start_bot(coin)
        
        buttons = [[InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")]]
        markup = InlineKeyboardMarkup(buttons)
        
        msg_formatted = f"♻️ {msg}".replace(".", "\\.")
        
        await safe_reply(update,
            msg_formatted,
            reply_markup=markup,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in restart_cmd: {e}")
        error_msg = f"⚠️ Restart error: {e}".replace(".", "\\.")
        await safe_reply(update, error_msg, parse_mode="Markdown")

@auth
async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop a specific trading bot"""
    coin = context.args[0].lower() if context.args else None
    
    if not coin:
        # Show stop menu
        buttons = []
        for coin in ['eth', 'link', 'doge', 'arb']:
            buttons.append([InlineKeyboardButton(f"Stop {coin.upper()}", callback_data=f"stop_{coin}")])
        
        buttons.append([InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")])
        markup = InlineKeyboardMarkup(buttons)
        
        await safe_reply(update,
            "Select a bot to stop:",
            reply_markup=markup
        )
        return
    
    if coin not in ['eth', 'link', 'doge', 'arb']:
        await safe_reply(update,
            "⚠️ Usage: /stop eth|link|doge|arb")
        return
        
    try:
        from live_trading_manager import stop_bot
        msg = stop_bot(coin)
        
        buttons = [[InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")]]
        markup = InlineKeyboardMarkup(buttons)
        
        msg_formatted = f"🛑 {msg}".replace(".", "\\.")
        
        await safe_reply(update,
            msg_formatted,
            reply_markup=markup,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in stop_cmd: {e}")
        error_msg = f"⚠️ Stop error: {e}".replace(".", "\\.")
        await safe_reply(update, error_msg, parse_mode="Markdown")

@auth
async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reload the configuration"""
    try:
        # CONFIG should be declared as global at script start
        with open(config_path, 'r') as f:
            CONFIG.clear()  # Clear existing config
            CONFIG.update(yaml.safe_load(f))  # Update with new values
            
        buttons = [[InlineKeyboardButton("🔙 Back to Settings", callback_data="settings_menu")]]
        markup = InlineKeyboardMarkup(buttons)
        
        await safe_reply(update,
            "🔁 Config reloaded successfully\\.",
            reply_markup=markup,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in reload_cmd: {e}")
        error_msg = f"⚠️ Reload failed: {e}".replace(".", "\\.")
        await safe_reply(update, error_msg, parse_mode="Markdown")

async def whale_alert_loop(app):
    """Background task to fetch and send whale alerts"""
    global whale_enabled
    while True:
        try:
            if whale_enabled:
                url = f"https://api.whale-alert.io/v1/transactions?api_key={WH_API_KEY}&min_value={MIN_VALUE}&limit=20"
                resp = requests.get(url)
                data = resp.json()
                txs = data.get("transactions", [])
                for tx in txs:
                    tx_id = tx.get("id")
                    if tx_id in last_tx_ids:
                        continue
                    last_tx_ids.append(tx_id)

                    symbol = tx.get("symbol", "").upper()
                    if symbol not in {"ETH", "LINK", "ARB", "DOGE"}:
                        continue

                    value = tx.get("amount_usd", 0)
                    from_label = tx.get("from", {}).get("owner", "Unknown")
                    to_label = tx.get("to", {}).get("owner", "Unknown")
                    ts = datetime.fromtimestamp(tx.get("timestamp")).strftime('%Y-%m-%d %H:%M:%S')

                    msg = format_whale_alert(symbol, value, from_label, to_label, ts)

                    
                    # Add buttons to show more info
                    buttons = [
                        [InlineKeyboardButton(f"📊 {symbol} Metrics", callback_data=f"metrics_{symbol}")]
                    ]
                    markup = InlineKeyboardMarkup(buttons)

                    await app.bot.send_message(
                        chat_id=ALLOWED_CHAT_ID,
                        text=msg,
                        reply_markup=markup,
                        parse_mode="Markdown"
                    )
        except Exception as e:
            logger.error(f"[WhaleAlert Error] {e}")
        await asyncio.sleep(90)

def format_whale_alert(symbol, value, from_label, to_label, timestamp_str):
    """Return a Markdown-safe formatted whale alert message."""
    symbol = symbol.upper().replace('.', '\\.').replace('(', '\\(').replace(')', '\\)')
    from_label = (from_label or "Unknown").replace('.', '\\.').replace('-', '\\-').replace('+', '\\+').replace('(', '\\(').replace(')', '\\)')
    to_label = (to_label or "Unknown").replace('.', '\\.').replace('-', '\\-').replace('+', '\\+').replace('(', '\\(').replace(')', '\\)')
    timestamp_str = timestamp_str.replace('.', '\\.').replace('(', '\\(').replace(')', '\\)')

    return (
        f"🐋 *Whale Alert*\n\n"
        f"*Token:* {symbol}\n"
        f"*Value:* ${round(value):,}\n"
        f"*From:* {from_label}\n"
        f"*To:* {to_label}\n"
        f"*Time:* {timestamp_str}"
    )

def generate_daily_summary():
    """Generate a summary of trading activity for the day"""
    try:
        global show_before_tariff
        tariff_type = "before" if show_before_tariff else "after"
        tariff_label = "Before" if show_before_tariff else "After"
        
        total_initial_capital = 0
        total_final_capital = 0
        total_pnl = 0
        total_trades = 0
        summary = [f"📊 *Daily Trading Summary ({tariff_label} Tariff)*\n"]
        
        for coin in SUPPORTED_COINS:
            trades_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_trades.csv")
            metrics_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_metrics.csv")
            
            if not os.path.exists(trades_file) or not os.path.exists(metrics_file):
                continue
                
            # Get today's trades
            trades_df = pd.read_csv(trades_file)
            metrics_df = pd.read_csv(metrics_file)
            
            if trades_df.empty:
                continue
                
            # Get initial capital if available, otherwise calculate it
            if 'initial_capital' in metrics_df.columns:
                initial_capital = metrics_df['initial_capital'].iloc[0]
            else:
                # Fallback calculation if initial_capital isn't stored
                initial_capital = metrics_df['final_capital'].iloc[0] - metrics_df['total_profit'].iloc[0]
                
            final_capital = metrics_df['final_capital'].iloc[0]
            
            # Update portfolio totals
            total_initial_capital += initial_capital
            total_final_capital += final_capital
                
            # Convert timestamp to datetime
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            today = datetime.now().date()
            
            # Filter for today's trades
            today_trades = trades_df[trades_df['timestamp'].dt.date == today]
            
            if today_trades.empty:
                continue
                
            # Count wins and losses
            sells = today_trades[today_trades['type'] == 'SELL']
            wins = len(sells[sells['profit'] > 0])
            losses = len(sells[sells['profit'] <= 0])
            total = wins + losses
            total_trades += total
            
            if total == 0:
                continue
                
            win_rate = (wins / total * 100) if total > 0 else 0
            today_pnl = sells['profit'].sum()
            total_pnl += today_pnl
            
            # Calculate ROI for today
            today_roi = (today_pnl / initial_capital * 100) if initial_capital > 0 else 0
            
            summary.append(f"*{coin}:*")
            summary.append(f"Trades: {total} \\(W: {wins}, L: {losses}\\)")
            summary.append(f"Win Rate: {win_rate:.1f}%")
            summary.append(f"PnL: ${today_pnl:.2f} \\(ROI: {today_roi:.2f}%\\)\n")
        
        if total_trades == 0:
            return f"No trades executed today \\({tariff_label} Tariff view\\)\\."
            
        # Calculate portfolio ROI for the day
        portfolio_roi = (total_pnl / total_initial_capital * 100) if total_initial_capital > 0 else 0
            
        summary.append(f"*Total Daily PnL: ${total_pnl:.2f}*")
        summary.append(f"*Total Trades: {total_trades}*")
        summary.append(f"*Portfolio ROI Today: {portfolio_roi:.2f}%*")
        
        return "\n".join(summary)
        
    except Exception as e:
        logger.error(f"Error generating daily summary: {e}")
        return f"⚠️ Error generating daily summary: {e}"

async def daily_summary_loop(app):
    """Background task to send daily trading summaries"""
    while True:
        try:
            now = datetime.now()
            if now.hour == 0 and now.minute == 0:
                # Send after tariff summary first (default)
                global show_before_tariff
                show_before_tariff = False
                after_summary = generate_daily_summary()
                
                # Add buttons to view detailed metrics
                buttons = [
                    [InlineKeyboardButton("📊 Full Metrics", callback_data="metrics_menu")]
                ]
                markup = InlineKeyboardMarkup(buttons)
                
                await app.bot.send_message(
                    chat_id=ALLOWED_CHAT_ID,
                    text=after_summary,
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
                
                # Now send before tariff summary
                show_before_tariff = True
                before_summary = generate_daily_summary()
                
                await app.bot.send_message(
                    chat_id=ALLOWED_CHAT_ID,
                    text=before_summary,
                    reply_markup=markup,
                    parse_mode="Markdown"
                )
                
                # Reset back to default
                show_before_tariff = False
        except Exception as e:
            logger.error(f"[Daily Summary Error] {e}")
        await asyncio.sleep(60)


# This code should be added to the part of your system that generates metrics CSVs

def update_metrics_with_initial_capital():
    """
    Updates existing metric CSV files to include initial_capital column
    if it doesn't already exist.
    """
    global show_before_tariff
    
    # Process both before and after tariff files
    for tariff_type in ['before', 'after']:
        for coin in SUPPORTED_COINS:
            metrics_file = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_metrics.csv")
            
            if os.path.exists(metrics_file):
                try:
                    # Read existing metrics
                    metrics_df = pd.read_csv(metrics_file)
                    
                    # If initial_capital doesn't exist, add it
                    if 'initial_capital' not in metrics_df.columns:
                        # Calculate initial capital from existing data
                        final_capital = metrics_df['final_capital'].iloc[0]
                        total_profit = metrics_df['total_profit'].iloc[0]
                        initial_capital = final_capital - total_profit
                        
                        # Add the new column
                        metrics_df['initial_capital'] = initial_capital
                        
                        # Save the updated CSV
                        metrics_df.to_csv(metrics_file, index=False)
                        logger.info(f"Updated {coin}_{tariff_type}_tariff_metrics.csv with initial_capital")
                except Exception as e:
                    logger.error(f"Error updating {metrics_file}: {e}")

# Example for adding initial_capital when generating new metrics
def save_trading_metrics(coin, metrics, tariff_type='after'):
    """
    Save trading metrics to a CSV file with initial_capital included.
    
    Args:
        coin (str): The coin symbol (e.g., 'ETH')
        metrics (dict): Dictionary containing the metrics
        tariff_type (str): Either 'before' or 'after'
    """
    try:
        # Ensure initial_capital is part of the metrics
        if 'initial_capital' not in metrics and 'final_capital' in metrics and 'total_profit' in metrics:
            metrics['initial_capital'] = metrics['final_capital'] - metrics['total_profit']
        
        # Convert metrics to DataFrame and save
        metrics_df = pd.DataFrame([metrics])
        file_path = os.path.join(LIVE_BACKUP_DIR, f"{coin}_{tariff_type}_tariff_metrics.csv")
        metrics_df.to_csv(file_path, index=False)
        logger.info(f"Saved {coin} metrics to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics for {coin}: {e}")

async def safe_reply(update, text, **kwargs):
    """Reply whether the trigger is a command or a callback."""
    if update.message:
        return await update.message.reply_text(text, **kwargs)
    if getattr(update, "callback_query", None):
        return await update.callback_query.message.reply_text(text, **kwargs)
    # fallback – very rare
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id:
        return await update.get_bot().send_message(chat_id, text, **kwargs)

async def safe_edit(message, *args, **kwargs):
    """
    Edit a Telegram message but silently ignore
    the "Message is not modified" BadRequest.
    """
    try:
        return await message.edit_text(*args, **kwargs)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            return  # nothing to change – swallow
        raise

@auth
async def whale_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/whale on|off  – toggle whale alerts, or show status."""
    global whale_enabled
    arg = (context.args[0].lower() if context.args else "")
    if arg == "off":
        whale_enabled = False
        await safe_reply(update, "🐋 Whale alerts turned *OFF*\\.", parse_mode="Markdown")
    elif arg == "on":
        whale_enabled = True
        await safe_reply(update, "🐋 Whale alerts turned *ON*\\.", parse_mode="Markdown")
    else:
        status = "ON" if whale_enabled else "OFF"
        buttons = [
            [InlineKeyboardButton("🟢 Turn ON",  callback_data="whale_on"),
             InlineKeyboardButton("🔴 Turn OFF", callback_data="whale_off")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
        ]
        await safe_reply(
            update,
            f"🐋 Whale alerts are currently: *{status}*\n\nUse the buttons below to toggle\\.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
        )


@auth
async def sentiment_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display sentiment analysis for a specific coin"""
    try:
        coin = context.args[0].upper() if context.args else None
        if not coin:
            # Show sentiment menu if no coin specified
            await send_sentiment_menu(update, context)
            return
            
        await send_sentiment_analysis(update, context, coin)
    except Exception as e:
        logger.error(f"Error in sentiment_cmd: {e}")
        error_msg = f"⚠️ Error analyzing sentiment: {str(e)}".replace(".", "\\.")
        await safe_reply(update, error_msg, parse_mode="Markdown")

async def send_sentiment_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display sentiment analysis menu with coin options"""
    buttons = []
    for coin in SUPPORTED_COINS:
        buttons.append([InlineKeyboardButton(f"{coin} Sentiment", callback_data=f"sentiment_{coin}")])
    
    buttons.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
    markup = InlineKeyboardMarkup(buttons)
    
    await safe_reply(update,
        "📝 *Reddit Sentiment Analysis*\n\nSelect a coin to view market sentiment from Reddit:",
        reply_markup=markup,
        parse_mode="Markdown"
    )

async def send_sentiment_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, coin, timeframe='week', is_callback=False):
    """Send sentiment analysis for a specific coin"""
    try:
        # First send a "processing" message
        if is_callback:
            processing_message = await update.callback_query.message.reply_text(
                f"🔍 Analyzing Reddit sentiment for {coin}... This may take a moment."
            )
        else:
            processing_message = await safe_reply(update,
                f"🔍 Analyzing Reddit sentiment for {coin}... This may take a moment."
            )
        
        # Run the sentiment analysis
        sentiment_data = reddit_sentiment.get_coin_sentiment(coin, timeframe)
        
        if 'error' in sentiment_data:
            # Handle error
            await processing_message.edit_text(
                f"⚠️ Error analyzing sentiment: {sentiment_data['error']}"
            )
            return
        
        # Create buttons with timeframe options
        timeframe_buttons = [
            [
                InlineKeyboardButton("📅 Day", callback_data=f"sentiment_{coin}_day"),
                InlineKeyboardButton("📅 Week", callback_data=f"sentiment_{coin}_week"),
                InlineKeyboardButton("📅 Month", callback_data=f"sentiment_{coin}_month")
            ],
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"sentiment_{coin}_refresh")],
            [InlineKeyboardButton("📊 Metrics", callback_data=f"metrics_{coin}")],
            [InlineKeyboardButton("🔙 All Coins", callback_data="sentiment_menu")]
        ]
        markup = InlineKeyboardMarkup(timeframe_buttons)
        
        # Send the text analysis
        message_text = sentiment_data['message']
        
        # Delete the processing message
        await processing_message.delete()
        
        # Send the formatted message
        if is_callback:
            sent_message = await update.callback_query.message.reply_text(
                message_text,
                reply_markup=markup,
                parse_mode="Markdown"
            )
        else:
            sent_message = await safe_reply(update,
                message_text,
                reply_markup=markup,
                parse_mode="Markdown"
            )
        
        # If we have chart data, send it as a photo
        if 'chart_data' in sentiment_data and sentiment_data['chart_data']:
            caption = f"📊 Sentiment Chart for {coin} (Past {timeframe.title()})"
            
            if is_callback:
                await update.callback_query.message.reply_photo(
                    photo=InputFile(sentiment_data['chart_data'], filename=f"{coin}_sentiment.png"),
                    caption=caption
                )
            else:
                await update.message.reply_photo(
                    photo=InputFile(sentiment_data['chart_data'], filename=f"{coin}_sentiment.png"),
                    caption=caption
                )
    
    except Exception as e:
        logger.error(f"Error in send_sentiment_analysis: {str(e)}\n{traceback.format_exc()}")
        error_message = f"⚠️ Error displaying sentiment: {str(e)}"
        
        if is_callback:
            await update.callback_query.message.reply_text(error_message)
        else:
            await safe_reply(update, error_message)

async def post_init(app):
    """Post-initialization tasks"""
    asyncio.create_task(whale_alert_loop(app))
    asyncio.create_task(daily_summary_loop(app))
    logger.info("🚀 Telegram bot running with WhaleAlert and Daily Summary...")

def main():
    global show_before_tariff
    show_before_tariff = False

    """Main function to start the bot"""
    try:
        app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
        
        # Register command handlers
        app.add_handler(CommandHandler("start", start_cmd))
        app.add_handler(CommandHandler("help", help_cmd))
        app.add_handler(CommandHandler("status", status_cmd))
        app.add_handler(CommandHandler("metrics", metrics_cmd))
        app.add_handler(CommandHandler("logs", logs_cmd))
        app.add_handler(CommandHandler("config", config_cmd))
        app.add_handler(CommandHandler("whale", whale_cmd))
        app.add_handler(CommandHandler("restart", restart_cmd))
        app.add_handler(CommandHandler("stop", stop_cmd))
        app.add_handler(CommandHandler("reload", reload_cmd))
        app.add_handler(CommandHandler("sentiment", sentiment_cmd))
        # Register callback handler for buttons
        app.add_handler(CallbackQueryHandler(button_handler))
        
        # Start the bot
        logger.info("Starting bot...")
        app.run_polling()
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}\n{traceback.format_exc()}")
        
if __name__ == '__main__':
    main()