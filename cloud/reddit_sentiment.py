# reddit_sentiment.py

import os
import praw
import pandas as pd
import numpy as np
import logging
import yaml
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import traceback
import matplotlib.pyplot as plt
from io import BytesIO
import asyncio
import time

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Update config path to match your project structure
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))

# Define supported coins and their subreddits
COIN_SUBREDDITS = {
    'ETH': ['ethereum', 'ethtrader', 'ethfinance'],
    'LINK': ['Chainlink', 'LinkTrader'],
    'DOGE': ['dogecoin', 'dogemarket'],
    'ARB': ['arbitrum', 'ArbitrumFoundation']
}

# Define keyword mappings for each coin to catch relevant posts
COIN_KEYWORDS = {
    'ETH': ['ethereum', 'eth', 'ether', 'vitalik', 'buterin', 'gwei', 'ethbtc'],
    'LINK': ['chainlink', 'link', 'sergey', 'nazarov', 'oracle', 'smartcontract'],
    'DOGE': ['dogecoin', 'doge', 'shiba', 'elon', 'musk', 'such wow'],
    'ARB': ['arbitrum', 'arb', 'layer2', 'l2', 'rollup', 'offchain']
}

# Sentiment categories
SENTIMENT_CATEGORIES = {
    'very_negative': (-1.0, -0.6),
    'negative': (-0.6, -0.2),
    'neutral': (-0.2, 0.2),
    'positive': (0.2, 0.6),
    'very_positive': (0.6, 1.0)
}

# Initialize global variables
CONFIG = {}
reddit = None
vader = None
sentiment_data = {}  # Cache for sentiment data

# Load configuration
def load_config():
    global CONFIG, reddit
    try:
        with open(config_path, 'r') as f:
            CONFIG = yaml.safe_load(f)
        
        # Initialize Reddit API connection
        reddit_config = CONFIG.get('reddit', {})
        reddit = praw.Reddit(
            client_id=reddit_config.get('client_id'),
            client_secret=reddit_config.get('client_secret'),
            user_agent=reddit_config.get('user_agent', 'crypto_sentiment_bot v1.0')
        )
        
        return True
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return False

# Initialize sentiment analyzer
def initialize_sentiment():
    global vader
    try:
        vader = SentimentIntensityAnalyzer()
        return True
    except Exception as e:
        logger.error(f"Error initializing sentiment analyzer: {e}")
        return False

def is_coin_related(text, coin):
    """Check if text is related to the specified coin"""
    text = text.lower()
    keywords = COIN_KEYWORDS.get(coin.upper(), [])
    
    # Check if any keyword is present in the text
    return any(keyword.lower() in text for keyword in keywords)

def get_reddit_posts(coin, limit=100, timeframe='week'):
    """
    Fetch Reddit posts related to a specific coin
    
    Args:
        coin (str): Coin symbol (ETH, LINK, etc.)
        limit (int): Maximum number of posts to retrieve per subreddit
        timeframe (str): 'day', 'week', 'month', 'year', or 'all'
    
    Returns:
        list: List of dictionaries containing post data
    """
    if reddit is None:
        logger.error("Reddit API not initialized")
        return []
    
    coin = coin.upper()
    posts = []
    
    # Convert timeframe to seconds for time filtering
    time_filters = {
        'day': 86400,
        'week': 604800,
        'month': 2592000,
        'year': 31536000,
        'all': None
    }
    
    time_filter = time_filters.get(timeframe.lower(), time_filters['week'])
    current_time = time.time()
    
    try:
        subreddits = COIN_SUBREDDITS.get(coin, [])
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Search for coin-specific posts with different sort methods
                for sort_method in ['hot', 'new', 'top']:
                    if sort_method == 'hot':
                        submissions = subreddit.hot(limit=limit)
                    elif sort_method == 'new':
                        submissions = subreddit.new(limit=limit)
                    else:  # top
                        submissions = subreddit.top(limit=limit, time_filter=timeframe)
                    
                    for submission in submissions:
                        # Skip if post is too old based on timeframe
                        if time_filter and (current_time - submission.created_utc) > time_filter:
                            continue
                            
                        # Only include non-stickied posts that are related to the coin
                        if (not submission.stickied and 
                            is_coin_related(submission.title + ' ' + submission.selftext, coin)):
                            
                            # Get post statistics
                            comments_count = len(list(submission.comments))
                            
                            posts.append({
                                'id': submission.id,
                                'subreddit': subreddit_name,
                                'title': submission.title,
                                'selftext': submission.selftext,
                                'score': submission.score,
                                'upvote_ratio': submission.upvote_ratio,
                                'num_comments': comments_count,
                                'created_utc': datetime.fromtimestamp(submission.created_utc),
                                'sort_method': sort_method,
                                'url': submission.url,
                                'permalink': f"https://reddit.com{submission.permalink}"
                            })
            except Exception as e:
                logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
                continue
                
        # Remove duplicates based on post ID
        unique_posts = []
        seen_ids = set()
        
        for post in posts:
            if post['id'] not in seen_ids:
                seen_ids.add(post['id'])
                unique_posts.append(post)
                
        return unique_posts
    
    except Exception as e:
        logger.error(f"Error getting Reddit posts for {coin}: {e}")
        return []

def analyze_sentiment(text):
    """
    Analyze sentiment of text using VADER
    
    Args:
        text (str): Text to analyze
    
    Returns:
        float: Compound sentiment score between -1 (negative) and 1 (positive)
    """
    if vader is None:
        logger.error("Sentiment analyzer not initialized")
        return 0.0
    
    if not text or text.strip() == '':
        return 0.0
    
    try:
        # Clean text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        
        # Get sentiment scores
        sentiment_scores = vader.polarity_scores(text)
        
        # Return compound score
        return sentiment_scores['compound']
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return 0.0

def get_sentiment_category(score):
    """Get sentiment category based on score"""
    for category, (min_val, max_val) in SENTIMENT_CATEGORIES.items():
        if min_val <= score < max_val:
            return category
    return 'neutral'

def analyze_coin_sentiment(coin, timeframe='week', limit=100, force_refresh=False):
    """
    Analyze sentiment for a specific coin
    
    Args:
        coin (str): Coin symbol (ETH, LINK, etc.)
        timeframe (str): 'day', 'week', 'month', 'year', or 'all'
        limit (int): Maximum number of posts to retrieve per subreddit
        force_refresh (bool): Force refresh of data instead of using cache
    
    Returns:
        dict: Sentiment analysis results
    """
    global sentiment_data
    
    coin = coin.upper()
    cache_key = f"{coin}_{timeframe}_{limit}"
    
    # If we have cached data and not force refreshing, return it
    if not force_refresh and cache_key in sentiment_data:
        # Check if the cached data is less than 1 hour old
        if (datetime.now() - sentiment_data[cache_key]['timestamp']).total_seconds() < 3600:
            return sentiment_data[cache_key]
    
    try:
        # Fetch Reddit posts
        posts = get_reddit_posts(coin, limit, timeframe)
        
        if not posts:
            logger.warning(f"No posts found for {coin}")
            return {
                'coin': coin,
                'timestamp': datetime.now(),
                'overall_sentiment': 0,
                'sentiment_category': 'neutral',
                'post_count': 0,
                'top_posts': [],
                'sentiment_distribution': {category: 0 for category in SENTIMENT_CATEGORIES},
                'timeline_data': [],
                'subreddit_sentiment': {}
            }
        
        # Create DataFrame for analysis
        df = pd.DataFrame(posts)
        
        # Calculate sentiment for each post
        df['title_sentiment'] = df['title'].apply(analyze_sentiment)
        df['content_sentiment'] = df['selftext'].apply(analyze_sentiment)
        
        # Calculate weighted sentiment (title has more weight)
        df['weighted_sentiment'] = df['title_sentiment'] * 0.6 + df['content_sentiment'] * 0.4
        
        # Add sentiment categories
        df['sentiment_category'] = df['weighted_sentiment'].apply(get_sentiment_category)
        
        # Weight sentiment by score and comments
        df['score_weight'] = np.log1p(df['score']) / np.log1p(df['score'].max()) if df['score'].max() > 0 else 0
        df['comment_weight'] = np.log1p(df['num_comments']) / np.log1p(df['num_comments'].max()) if df['num_comments'].max() > 0 else 0
        df['engagement_weight'] = (df['score_weight'] * 0.7) + (df['comment_weight'] * 0.3)
        
        # Apply engagement weight to sentiment
        df['weighted_engagement_sentiment'] = df['weighted_sentiment'] * (0.5 + 0.5 * df['engagement_weight'])
        
        # Calculate overall sentiment
        if len(df) > 0:
            overall_sentiment = df['weighted_engagement_sentiment'].mean()
            sentiment_category = get_sentiment_category(overall_sentiment)
        else:
            overall_sentiment = 0
            sentiment_category = 'neutral'
        
        # Get sentiment distribution
        sentiment_distribution = {}
        for category in SENTIMENT_CATEGORIES:
            count = len(df[df['sentiment_category'] == category])
            sentiment_distribution[category] = count
        
        # Get top posts by engagement, sorted by sentiment categories
        top_positive_posts = df[df['weighted_sentiment'] > 0.2].sort_values('engagement_weight', ascending=False).head(3)
        top_negative_posts = df[df['weighted_sentiment'] < -0.2].sort_values('engagement_weight', ascending=False).head(3)
        top_neutral_posts = df[abs(df['weighted_sentiment']) <= 0.2].sort_values('engagement_weight', ascending=False).head(1)
        
        # Combine and sort by engagement
        top_posts_df = pd.concat([top_positive_posts, top_negative_posts, top_neutral_posts])
        top_posts_df = top_posts_df.sort_values('engagement_weight', ascending=False)
        
        # Limit to 5 top posts overall
        top_posts_df = top_posts_df.head(5)
        
        # Create a list of top posts
        top_posts = []
        for _, post in top_posts_df.iterrows():
            sentiment_emoji = "😀" if post['weighted_sentiment'] > 0.2 else "😐" if abs(post['weighted_sentiment']) <= 0.2 else "😟"
            
            top_posts.append({
                'title': post['title'],
                'score': post['score'],
                'comments': post['num_comments'],
                'sentiment': post['weighted_sentiment'],
                'sentiment_category': post['sentiment_category'],
                'sentiment_emoji': sentiment_emoji,
                'subreddit': post['subreddit'],
                'created_utc': post['created_utc'],
                'url': post['permalink']
            })
        
        # Create timeline data (average sentiment per day)
        df['date'] = df['created_utc'].dt.date
        timeline_data = df.groupby('date')['weighted_sentiment'].mean().reset_index()
        timeline_data['date_str'] = timeline_data['date'].astype(str)
        
        # Group by subreddit to show sentiment per community
        subreddit_sentiment = df.groupby('subreddit')['weighted_sentiment'].mean().to_dict()
        
        # Calculate post volume by subreddit
        subreddit_volume = df['subreddit'].value_counts().to_dict()
        
        # Create final result
        result = {
            'coin': coin,
            'timestamp': datetime.now(),
            'overall_sentiment': overall_sentiment,
            'sentiment_category': sentiment_category,
            'post_count': len(df),
            'top_posts': top_posts,
            'sentiment_distribution': sentiment_distribution,
            'timeline_data': timeline_data.to_dict('records'),
            'subreddit_sentiment': subreddit_sentiment,
            'subreddit_volume': subreddit_volume
        }
        
        # Cache the result
        sentiment_data[cache_key] = result
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {coin}: {str(e)}\n{traceback.format_exc()}")
        return {
            'coin': coin,
            'timestamp': datetime.now(),
            'overall_sentiment': 0,
            'sentiment_category': 'neutral',
            'post_count': 0,
            'top_posts': [],
            'error': str(e)
        }

def generate_sentiment_chart(sentiment_result):
    """
    Generate a sentiment chart from analysis results
    
    Args:
        sentiment_result (dict): Sentiment analysis results
    
    Returns:
        BytesIO: PNG image data
    """
    try:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Chart 1: Sentiment Distribution
        sentiment_distribution = sentiment_result['sentiment_distribution']
        categories = list(sentiment_distribution.keys())
        values = list(sentiment_distribution.values())
        
        # Define colors for sentiment categories
        colors = ['#d7301f', '#fc8d59', '#fee08b', '#78c679', '#1a9641']
        
        # Plot the sentiment distribution as a bar chart
        bars = ax1.bar(categories, values, color=colors)
        ax1.set_title(f"Sentiment Distribution for {sentiment_result['coin']}")
        ax1.set_ylabel('Number of Posts')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Chart 2: Sentiment Timeline
        timeline_data = sentiment_result['timeline_data']
        
        if timeline_data:
            dates = [record['date_str'] for record in timeline_data]
            sentiments = [record['weighted_sentiment'] for record in timeline_data]
            
            # Define colors based on sentiment values
            line_colors = ['#d7301f' if s < -0.2 else '#fee08b' if s < 0.2 else '#1a9641' for s in sentiments]
            
            # Plot the sentiment timeline
            ax2.plot(dates, sentiments, marker='o', linestyle='-', color='#5a9bd5')
            
            # Plot colored points based on sentiment
            for i, (date, sentiment) in enumerate(zip(dates, sentiments)):
                ax2.scatter(date, sentiment, color=line_colors[i], s=50, zorder=5)
            
            # Add a horizontal line at y=0
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add color zones for sentiment categories
            ax2.axhspan(-1, -0.2, alpha=0.1, color='red')
            ax2.axhspan(-0.2, 0.2, alpha=0.1, color='yellow')
            ax2.axhspan(0.2, 1, alpha=0.1, color='green')
            
            ax2.set_title(f"Sentiment Timeline for {sentiment_result['coin']}")
            ax2.set_ylabel('Sentiment Score')
            ax2.set_xlabel('Date')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
        else:
            ax2.text(0.5, 0.5, "No timeline data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO object
        img_data = BytesIO()
        plt.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        plt.close(fig)
        
        return img_data
    
    except Exception as e:
        logger.error(f"Error generating sentiment chart: {str(e)}")
        return None

def format_sentiment_message(sentiment_result):
    """
    Format sentiment analysis results as a Telegram-friendly message
    
    Args:
        sentiment_result (dict): Sentiment analysis results
    
    Returns:
        str: Formatted message
    """
    coin = sentiment_result['coin']
    sentiment_score = sentiment_result['overall_sentiment']
    category = sentiment_result['sentiment_category']
    post_count = sentiment_result['post_count']
    
    # Map sentiment categories to emojis
    sentiment_emojis = {
        'very_negative': '😡',
        'negative': '😟',
        'neutral': '😐',
        'positive': '🙂',
        'very_positive': '😀'
    }
    
    emoji = sentiment_emojis.get(category, '😐')
    
    # Format the message
    message = [f"🔍 *Reddit Sentiment Analysis: {coin}*\n"]
    
    if post_count == 0:
        message.append("No relevant posts found for analysis\\.")
        return "\n".join(message)
    
    # Overall sentiment
    message.append(f"*Overall Sentiment:* {emoji} {category.replace('_', ' ').title()} ({sentiment_score:.2f})")
    message.append(f"*Based on:* {post_count} posts")
    
    # Add subreddit breakdown
    if 'subreddit_volume' in sentiment_result and sentiment_result['subreddit_volume']:
        message.append("\n*Subreddit Breakdown:*")
        for subreddit, sentiment in sentiment_result['subreddit_sentiment'].items():
            volume = sentiment_result['subreddit_volume'].get(subreddit, 0)
            sub_emoji = sentiment_emojis.get(get_sentiment_category(sentiment), '😐')
            message.append(f"r/{subreddit}: {sub_emoji} {sentiment:.2f} \\({volume} posts\\)")
    
    # Add top posts
    if sentiment_result['top_posts']:
        message.append("\n*Top Posts:*")
        for i, post in enumerate(sentiment_result['top_posts'], 1):
            title = post['title'].replace('*', '').replace('_', '').replace('[', '').replace(']', '')
            if len(title) > 60:
                title = title[:57] + "..."
            
            post_emoji = post['sentiment_emoji']
            date_str = post['created_utc'].strftime('%Y-%m-%d')
            
            message.append(f"{i}\\. {post_emoji} {title}")
            message.append(f"   r/{post['subreddit']} • ⬆️ {post['score']} • 💬 {post['comments']} • 📅 {date_str}")
    
    message.append(f"\n_Last updated: {sentiment_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}_")
    
    return "\n".join(message)

# Initialize functions
def initialize():
    """Initialize the sentiment analysis module"""
    if load_config() and initialize_sentiment():
        logger.info("Reddit sentiment analysis module initialized successfully")
        return True
    else:
        logger.error("Failed to initialize Reddit sentiment analysis module")
        return False

# Main function to analyze sentiment
def get_coin_sentiment(coin, timeframe='week', limit=100, force_refresh=False):
    """
    Get sentiment analysis for a coin
    
    Args:
        coin (str): Coin symbol (ETH, LINK, etc.)
        timeframe (str): 'day', 'week', 'month', or 'all'
        limit (int): Maximum number of posts to retrieve
        force_refresh (bool): Force refresh instead of using cache
    
    Returns:
        dict: Sentiment analysis results and formatted message
    """
    # Make sure everything is initialized
    if reddit is None or vader is None:
        if not initialize():
            return {
                'error': 'Failed to initialize sentiment analysis module',
                'message': '⚠️ Error: Sentiment analysis module not initialized properly.'
            }
    
    try:
        # Get the sentiment analysis result
        result = analyze_coin_sentiment(coin, timeframe, limit, force_refresh)
        
        # Format the message
        formatted_message = format_sentiment_message(result)
        
        # Generate chart
        chart_data = generate_sentiment_chart(result)
        
        return {
            'result': result,
            'message': formatted_message,
            'chart_data': chart_data
        }
    except Exception as e:
        logger.error(f"Error in get_coin_sentiment: {str(e)}\n{traceback.format_exc()}")
        return {
            'error': str(e),
            'message': f'⚠️ Error analyzing sentiment for {coin}: {str(e)}'
        }

if __name__ == "__main__":
    # For testing
    initialize()
    result = get_coin_sentiment('ETH', timeframe='week')
    print(result['message'])