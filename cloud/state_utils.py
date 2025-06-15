# state_utils.py
import os
import json

STATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'state'))
os.makedirs(STATE_DIR, exist_ok=True)

def get_state_path(symbol):
    return os.path.join(STATE_DIR, f"{symbol.lower()}_state.json")

def load_state(symbol):
    path = get_state_path(symbol)
    if not os.path.exists(path):
        return {
            "in_position": False,
            "entry_price": 0,
            "last_trade_time": 0,
            "win_count": 0,
            "total_trades": 0
        }
    with open(path, 'r') as f:
        return json.load(f)

def save_state(symbol, state):
    path = get_state_path(symbol)
    with open(path, 'w') as f:
        json.dump(state, f)
