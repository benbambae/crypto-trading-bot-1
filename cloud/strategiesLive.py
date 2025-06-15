# strategies.py
import pandas as pd
def eth_strategy(df):
    """
    Ethereum strategy focused on trend following with multiple confirmations.
    Well-suited for ETH's relatively stable trends and high liquidity,
    using moving averages and momentum indicators to capture ETH's tendency
    to form sustained trends while managing volatility.
    """
    # Calculate multiple moving averages for more complex analysis
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma100'] = df['close'].rolling(window=100).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    # Enhanced volume analysis
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ma_long'] = df['volume'].rolling(window=50).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Multiple momentum indicators
    df['momentum_short'] = df['close'].diff(3)
    df['momentum_medium'] = df['close'].diff(7)
    df['momentum_long'] = df['close'].diff(14)
    
    # Advanced oscillators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_slow'] = calculate_rsi(df['close'], 21)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    # Volatility measures
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = calculate_bollinger_bands(df['close'])
    df['atr'] = calculate_atr(df, 14)
    df['atr_percent'] = df['atr'] / df['close'] * 100
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Complex entry conditions with multiple confirmations
    entry_condition = (
        last['ma20'] > last['ma50'] * 1.02 and 
        last['ma50'] > last['ma100'] * 1.01 and
        last['rsi'] < 70 and last['rsi'] > 40 and
        last['rsi_slow'] > prev['rsi_slow'] and
        last['stoch_k'] > last['stoch_d'] and
        last['macd'] > last['macd_signal'] and
        (last['volume_ratio'] > 1.2 or last['close'] < last['bollinger_lower']) and
        last['close'] > last['ma20'] * 0.98
    )
    
    # Sophisticated exit strategy with multiple risk factors
    exit_condition = (
        (last['volume'] > df['volume_ma'].iloc[-1] * 1.5 and last['close'] < prev['close']) or 
        last['close'] < last['ma20'] * 0.98 or
        last['momentum_short'] < 0 and last['momentum_medium'] < 0 or
        last['rsi'] > 80 or
        last['macd'] < last['macd_signal'] and prev['macd'] > prev['macd_signal'] or
        last['close'] < last['bollinger_lower'] * 0.99
    )
    
    return (
        entry_condition,
        exit_condition,
        last['close']
    )

def link_strategy(df):
    """
    Chainlink strategy optimized for its higher volatility and rapid price movements.
    Uses exponential moving averages which respond faster to price changes than simple MAs,
    with volume confirmation and trend strength indicators (ADX) to filter out false signals
    in LINK's sometimes choppy market conditions.
    """
    # Multiple exponential moving averages
    df['ema_very_fast'] = df['close'].ewm(span=3).mean()
    df['ema_fast'] = df['close'].ewm(span=5).mean()
    df['ema_medium'] = df['close'].ewm(span=8).mean()
    df['ema_slow'] = df['close'].ewm(span=13).mean()
    df['ema_very_slow'] = df['close'].ewm(span=21).mean()
    
    # Advanced momentum indicators
    df['momentum_short'] = df['close'].diff(3)
    df['momentum_medium'] = df['close'].diff(7)
    df['rate_of_change'] = df['close'].pct_change(5) * 100
    
    # Volatility and trend strength
    df['atr'] = calculate_atr(df, 14)
    df['atr_percent'] = df['atr'] / df['close'] * 100
    df['adx'] = calculate_adx(df, 14)
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = calculate_bollinger_bands(df['close'])
    
    # Volume analysis
    df['volume_ma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['obv'] = calculate_obv(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Complex entry with multiple confirmations and trend strength
    entry_condition = (
        last['ema_fast'] > last['ema_slow'] * 1.01 and 
        last['ema_medium'] > last['ema_slow'] and
        last['momentum_short'] > 0 and 
        last['rate_of_change'] > 1.5 and
        prev['ema_fast'] <= prev['ema_slow'] and
        last['adx'] > 25 and  # Strong trend
        last['volume_ratio'] > 1.1 and
        last['obv'] > df['obv'].shift(3).iloc[-1] and
        last['close'] > last['bollinger_middle'] and
        (last['close'] - last['bollinger_lower']) / (last['bollinger_upper'] - last['bollinger_lower']) < 0.7  # Not overbought
    )
    
    # Sophisticated exit with multiple risk factors
    exit_condition = (
        last['close'] < last['ema_slow'] * 0.99 or 
        last['momentum_short'] < 0 and last['momentum_medium'] < 0 or
        last['ema_very_fast'] < last['ema_fast'] and prev['ema_very_fast'] > prev['ema_fast'] or
        last['close'] < last['bollinger_lower'] or
        last['adx'] < 20 and last['momentum_short'] < 0 or  # Weakening trend
        last['volume_ratio'] > 2.0 and last['close'] < prev['close']  # Volume spike with price drop
    )

    return (
        entry_condition,
        exit_condition,
        last['close']
    )

def matic_strategy(df):
    """
    Polygon (MATIC) strategy designed for its growth-oriented but sometimes volatile market.
    Focuses on market structure (higher highs/lows) and decreasing volatility patterns
    to enter during steady rises rather than volatile spikes, with multiple timeframe
    analysis to confirm sustainable trends in this relatively newer asset.
    """
    # Multiple timeframe analysis
    df['ema_very_fast'] = df['close'].ewm(span=5).mean()
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_medium'] = df['close'].ewm(span=21).mean()
    df['ema_slow'] = df['close'].ewm(span=30).mean()
    df['ema_very_slow'] = df['close'].ewm(span=50).mean()
    
    # Price action analysis
    df['momentum_short'] = df['close'] - df['close'].shift(3)
    df['momentum_medium'] = df['close'] - df['close'].shift(7)
    df['momentum_long'] = df['close'] - df['close'].shift(14)
    df['price_change'] = df['close'].pct_change() * 100
    
    # Volatility measures
    df['volatility_short'] = df['close'].rolling(window=10).std() / df['close'] * 100
    df['volatility_medium'] = df['close'].rolling(window=20).std() / df['close'] * 100
    df['atr'] = calculate_atr(df, 14)
    df['atr_percent'] = df['atr'] / df['close'] * 100
    
    # Advanced oscillators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
    df['cci'] = calculate_cci(df, 20)
    
    # Market structure
    df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['higher_low'] = (df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2))
    df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Complex entry with multiple confirmations and volatility filters
    entry_condition = (
        last['ema_fast'] > last['ema_slow'] * 1.03 and 
        last['ema_medium'] > last['ema_slow'] * 1.01 and
        last['momentum_short'] > 0 and 
        last['momentum_medium'] > 0 and
        prev['ema_fast'] <= prev['ema_slow'] and
        last['volatility_short'] < last['close'] * 0.03 and
        last['volatility_short'] < last['volatility_medium'] * 0.9 and  # Decreasing volatility
        last['rsi'] > 50 and last['rsi'] < 70 and  # Not overbought
        last['stoch_k'] > last['stoch_d'] and
        last['cci'] > 0 and
        last['higher_high'] and
        last['higher_low'] and
        last['price_change'] > 0 and last['price_change'] < 3  # Steady rise, not a spike
    )
    
    # Sophisticated exit with multiple risk factors and trend reversal signals
    exit_condition = (
        last['momentum_short'] < 0 or 
        last['close'] < last['ema_fast'] * 0.98 or
        (last['ema_very_fast'] < last['ema_fast'] and prev['ema_very_fast'] > prev['ema_fast']) or
        last['rsi'] < 40 or last['rsi'] > 80 or
        (last['stoch_k'] < last['stoch_d'] and prev['stoch_k'] > prev['stoch_d']) or
        last['cci'] < -100 or
        last['lower_low'] or
        (last['volatility_short'] > last['volatility_medium'] * 1.5 and last['price_change'] < 0)  # Volatility spike with price drop
    )

    return (
        entry_condition,
        exit_condition,
        last['close']
    )

def doge_strategy(df):
    """
    Dogecoin strategy tailored for its unique market characteristics - high volatility,
    sentiment-driven price action, and occasional dramatic rallies.
    Uses multiple timeframe analysis with strong trend confirmation requirements
    and volume analysis to identify genuine momentum from social media-driven surges,
    with quick exit conditions to protect profits in this highly volatile asset.
    """
    # Multiple moving averages for trend analysis
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_100'] = df['close'].rolling(window=100).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential moving averages for more responsive signals
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_13'] = df['close'].ewm(span=13).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # Price action and momentum
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    df['momentum_short'] = df['close'].pct_change(3) * 100
    df['momentum_medium'] = df['close'].pct_change(7) * 100
    df['momentum_long'] = df['close'].pct_change(14) * 100
    
    # Advanced oscillators and indicators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
    
    # Volume analysis
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['obv'] = calculate_obv(df)
    
    # Volatility measures
    df['atr'] = calculate_atr(df, 14)
    df['atr_percent'] = df['atr'] / df['close'] * 100
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = calculate_bollinger_bands(df['close'])
    
    # Market structure and patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['higher_low'] = (df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2))
    df['bullish_engulfing'] = (df['open'] < df['prev_close']) & (df['close'] > df['prev_open'])
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Complex entry with multiple confirmations, trend alignment, and pattern recognition
    entry_condition = (
        last['ma_20'] > last['ma_50'] * 1.01 and 
        last['ma_50'] > last['ma_100'] * 1.01 and 
        last['ma_100'] > last['ma_200'] * 1.005 and  # Strong uptrend across timeframes
        last['close'] > prev['close'] * 1.05 and  # Significant price increase
        last['ema_5'] > last['ema_13'] and
        last['ema_13'] > last['ema_26'] and
        last['rsi'] > 50 and last['rsi'] < 75 and  # Strong but not overbought
        last['macd'] > last['macd_signal'] and
        last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 80 and
        last['volume_ratio'] > 1.2 and  # Above average volume
        last['obv'] > df['obv'].shift(5).iloc[-1] and  # Rising OBV
        (last['higher_high'] or last['bullish_engulfing']) and
        last['close'] > last['bollinger_middle'] and
        last['momentum_short'] > 2 and last['momentum_medium'] > 5  # Strong momentum
    )
    
    # Sophisticated exit with multiple risk factors, trend reversal signals, and volatility-based stops
    exit_condition = (
        last['close'] < last['ma_20'] * 0.98 or
        (last['ema_5'] < last['ema_13'] and prev['ema_5'] > prev['ema_13']) or
        last['rsi'] < 40 or last['rsi'] > 80 or
        (last['macd'] < last['macd_signal'] and prev['macd'] > prev['macd_signal']) or
        last['close'] < last['bollinger_lower'] or
        (last['volume_ratio'] > 2.0 and last['close'] < prev['close']) or  # Volume spike with price drop
        last['momentum_short'] < -3 or  # Sharp momentum reversal
        (last['atr_percent'] > 5 and last['close'] < prev['close'])  # High volatility with price drop
    )

    return (
        entry_condition,
        exit_condition,
        last['close']
    )

def arb_strategy(df):
    """
    Arbitrum (ARB) strategy designed for this Layer 2 scaling solution with growing adoption.
    Employs comprehensive technical analysis with emphasis on trend strength (ADX),
    market structure, and multiple timeframe confirmation to capture ARB's growth phases
    while filtering out noise. Includes sophisticated volatility analysis to manage risk
    in this relatively newer token with developing market patterns.
    """
    # Multiple exponential moving averages for trend analysis
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_30'] = df['close'].ewm(span=30).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_100'] = df['close'].ewm(span=100).mean()
    
    # Price action and momentum analysis
    df['momentum_short'] = df['close'] - df['close'].shift(3)
    df['momentum_medium'] = df['close'] - df['close'].shift(7)
    df['momentum_long'] = df['close'] - df['close'].shift(14)
    df['rate_of_change'] = df['close'].pct_change(5) * 100
    
    # Volatility measures
    df['range'] = df['high'] - df['low']
    df['avg_range'] = df['range'].rolling(window=14).mean()
    df['range_ratio'] = df['range'] / df['avg_range']
    df['atr'] = calculate_atr(df, 14)
    df['atr_percent'] = df['atr'] / df['close'] * 100
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = calculate_bollinger_bands(df['close'])
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle'] * 100
    
    # Advanced oscillators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_ma'] = df['rsi'].rolling(window=5).mean()  # Smoothed RSI
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['cci'] = calculate_cci(df, 20)
    df['adx'] = calculate_adx(df, 14)
    
    # Volume analysis
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['obv'] = calculate_obv(df)
    df['obv_ma'] = df['obv'].rolling(window=10).mean()
    
    # Market structure and patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['higher_low'] = (df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2))
    df['lower_high'] = (df['high'] < df['high'].shift(1)) & (df['high'].shift(1) < df['high'].shift(2))
    df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Complex entry with multiple confirmations, trend alignment, volatility filters, and oscillator signals
    entry_condition = (
        last['ema_10'] > last['ema_30'] * 1.01 and 
        last['ema_30'] > last['ema_50'] * 1.005 and 
        last['ema_50'] > last['ema_100'] * 1.002 and  # Strong uptrend across timeframes
        last['momentum_short'] > 0 and 
        last['momentum_medium'] > 0 and
        prev['ema_10'] <= prev['ema_30'] and  # Fresh crossover
        last['range'] < last['avg_range'] * 1.2 and  # Not excessive volatility
        last['bollinger_width'] < df['bollinger_width'].rolling(window=20).mean().iloc[-1] * 1.1 and  # Not expanding volatility
        last['rsi'] > 50 and last['rsi'] < 70 and  # Strong but not overbought
        last['rsi'] > last['rsi_ma'] and  # Rising RSI
        last['macd'] > last['macd_signal'] and
        last['macd_hist'] > prev['macd_hist'] and  # Increasing histogram
        last['adx'] > 25 and  # Strong trend
        last['volume_ratio'] > 1.1 and  # Above average volume
        last['obv'] > last['obv_ma'] and  # Rising OBV
        (last['higher_high'] or last['higher_low']) and  # Bullish structure
        last['close'] > last['bollinger_middle'] and
        last['cci'] > 0 and last['cci'] < 200  # Positive but not extreme CCI
    )
    
    # Sophisticated exit with multiple risk factors, trend reversal signals, and dynamic stops
    exit_condition = (
        last['momentum_short'] < 0 or 
        last['close'] < last['ema_10'] * 0.99 or
        (last['ema_5'] < last['ema_10'] and prev['ema_5'] > prev['ema_10']) or  # Fast EMA crossover
        last['rsi'] < 40 or last['rsi'] > 80 or
        (last['rsi'] < last['rsi_ma'] and prev['rsi'] > prev['rsi_ma']) or  # RSI crossing below its MA
        (last['macd'] < last['macd_signal'] and prev['macd'] > prev['macd_signal']) or
        last['adx'] < 20 and last['momentum_short'] < 0 or  # Weakening trend
        last['close'] < last['bollinger_lower'] or
        (last['volume_ratio'] > 2.0 and last['close'] < prev['close']) or  # Volume spike with price drop
        (last['range_ratio'] > 1.5 and last['close'] < prev['close']) or  # Volatility spike with price drop
        (last['lower_high'] and last['lower_low']) or  # Bearish structure
        (last['cci'] < -100 and prev['cci'] > -100)  # CCI crossing below -100
    )

    return (
        entry_condition,
        exit_condition,
        last['close']
    )

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range.rolling(period).mean()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    return upper_band, middle_band, lower_band

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = prices.ewm(span=fast_period).mean()
    slow_ema = prices.ewm(span=slow_period).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic(df, k_period=14, d_period=3):
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_cci(df, period=20):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    ma = typical_price.rolling(window=period).mean()
    mean_deviation = abs(typical_price - ma).rolling(window=period).mean()
    cci = (typical_price - ma) / (0.015 * mean_deviation)
    return cci

def calculate_adx(df, period=14):
    # Calculate True Range
    df['tr'] = calculate_atr(df, 1)
    
    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = ((df['up_move'] > df['down_move']) & (df['up_move'] > 0)) * df['up_move']
    df['minus_dm'] = ((df['down_move'] > df['up_move']) & (df['down_move'] > 0)) * df['down_move']
    
    # Calculate Directional Indicators
    df['plus_di'] = 100 * df['plus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean()
    df['minus_di'] = 100 * df['minus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean()
    
    # Calculate Directional Index
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    
    # Calculate Average Directional Index
    adx = df['dx'].rolling(window=period).mean()
    return adx

def calculate_obv(df):
    obv = pd.Series(index=df.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv
