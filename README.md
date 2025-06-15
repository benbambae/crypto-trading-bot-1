# Crypto Trading Bot - AWS EC2 Deployment Guide

A sophisticated multi-currency cryptocurrency trading bot with sentiment analysis, technical indicators, and Telegram integration.

## Features

- **Multi-Currency Support**: Trade ETH, LINK, MATIC, DOGE, and ARB simultaneously
- **Technical Analysis**: Multiple trading strategies based on technical indicators
- **Sentiment Analysis**: Integration with Reddit and Twitter for market sentiment
- **Whale Alert Monitoring**: Track large cryptocurrency transactions
- **Telegram Bot Interface**: Monitor and control your bot via Telegram
- **Paper & Live Trading**: Test strategies with paper trading before going live
- **Docker Support**: Easy deployment with containerization

## Prerequisites

Before deploying to AWS EC2, ensure you have:

1. AWS account with EC2 access
2. API keys for:
   - Binance (trading)
   - Telegram Bot (monitoring/control)
   - Reddit (sentiment analysis)
   - Twitter (sentiment analysis)
   - WhaleAlert (transaction monitoring)

## AWS EC2 Setup Guide

### Step 1: Launch EC2 Instance

1. **Choose Instance Type**:
   - Minimum: `t3.small` (2 vCPU, 2 GB RAM)
   - Recommended: `t3.medium` (2 vCPU, 4 GB RAM) for better performance

2. **Configure Instance**:
   - AMI: Ubuntu Server 22.04 LTS
   - Storage: 20 GB gp3 (General Purpose SSD)
   - Security Group: Allow SSH (port 22) from your IP

3. **Create/Select Key Pair**:
   - Download the `.pem` file and keep it secure

### Step 2: Connect to EC2 Instance

```bash
# Make key file read-only
chmod 400 your-key.pem

# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### Step 3: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Git
sudo apt install git -y

# Logout and login again for docker permissions
exit
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### Step 4: Clone and Configure the Bot

```bash
# Clone the repository
git clone <your-repository-url> crypto-trading-bot
cd crypto-trading-bot/crypto-trading-bot-1/cloud

# Configure your API keys
nano config.yaml
```

### Step 5: Update Configuration

Edit `config.yaml` and replace all placeholder values:

```yaml
alerts:
  telegram:
    token: "<ENTER YOUR TELEGRAM BOT TOKEN>"
    chat_id: "<ENTER YOUR TELEGRAM CHAT ID>"

binance:
  api_key: "<ENTER YOUR BINANCE API KEY>"
  secret_key: "<ENTER YOUR BINANCE SECRET KEY>"
  test_api_key: "<ENTER YOUR BINANCE TESTNET API KEY>"
  test_secret_key: "<ENTER YOUR BINANCE TESTNET SECRET KEY>"

whaleAlert:
  api_key: "<ENTER YOUR WHALE ALERT API KEY>"

twitter:
  api_key: "<ENTER YOUR TWITTER API KEY>"
  api_secret_key: "<ENTER YOUR TWITTER API SECRET KEY>"
  access_token: "<ENTER YOUR TWITTER ACCESS TOKEN>"
  access_token_secret: "<ENTER YOUR TWITTER ACCESS TOKEN SECRET>"
  bearer_token: "<ENTER YOUR TWITTER BEARER TOKEN>"

reddit:
  client_id: "<ENTER YOUR REDDIT CLIENT ID>"
  client_secret: "<ENTER YOUR REDDIT CLIENT SECRET>"
  user_agent: "<ENTER YOUR REDDIT USER AGENT>"
```

### Step 6: Build and Run with Docker

```bash
# Build Docker image
docker build -t crypto-bot .

# Run in detached mode
docker run -d --name crypto-bot --restart=unless-stopped crypto-bot

# View logs
docker logs -f crypto-bot
```

### Step 7: Bot Management

**Docker Commands**:
```bash
# Stop bot
docker stop crypto-bot

# Start bot
docker start crypto-bot

# Restart bot
docker restart crypto-bot

# Remove bot container
docker rm -f crypto-bot

# View running containers
docker ps
```

**Telegram Commands** (once bot is running):
- `/start` - Initialize bot
- `/menu` - Show main menu
- `/start_all` - Start all enabled trading bots
- `/stop_all` - Stop all trading bots
- `/metrics` - View trading metrics
- `/logs` - View recent logs
- `/whale_alerts` - Toggle whale alerts
- `/sentiment` - Check market sentiment

## Configuration Guide

### Trading Bot Settings

Each bot can be configured individually in `config.yaml`:

```yaml
eth_bot:
  enabled: true          # Enable/disable this bot
  quantity: 0.05         # Amount to trade
  tp_pct: 0.04          # Take profit percentage (4%)
  sl_pct: 0.03          # Stop loss percentage (3%)
  interval: '15m'        # Trading interval
  max_trades: 3          # Maximum concurrent trades
  cooldown: 180          # Cooldown between trades (seconds)
```

### Getting API Keys

1. **Binance**:
   - Create account at https://www.binance.com
   - Enable 2FA
   - Go to API Management
   - Create new API key with spot trading permissions
   - For testnet: https://testnet.binance.vision/

2. **Telegram Bot**:
   - Message @BotFather on Telegram
   - Create new bot with `/newbot`
   - Get your chat ID by messaging @userinfobot

3. **Reddit**:
   - Go to https://www.reddit.com/prefs/apps
   - Create new app (script type)
   - Note client ID and secret

4. **Twitter/X**:
   - Apply for developer account at https://developer.twitter.com
   - Create new app
   - Generate all required tokens

5. **WhaleAlert**:
   - Sign up at https://whale-alert.io
   - Get free API key (limited requests)

## Security Best Practices

1. **API Key Security**:
   - Never commit API keys to version control
   - Use environment variables for production
   - Restrict API key permissions (only enable spot trading)
   - Whitelist EC2 IP in exchange API settings

2. **EC2 Security**:
   - Keep security group restrictive (only SSH from your IP)
   - Use IAM roles if accessing other AWS services
   - Enable CloudWatch monitoring
   - Regular system updates

3. **Bot Security**:
   - Set conservative stop-loss limits
   - Start with paper trading
   - Monitor bot activity via Telegram
   - Set up alerts for unusual activity

## Monitoring and Maintenance

### CloudWatch Setup (Optional)

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure to monitor Docker logs
```

### Backup Strategy

1. **State Backup**:
   ```bash
   # Backup bot state
   docker exec crypto-bot tar -czf - /app/state > bot-state-backup.tar.gz
   
   # Backup trade logs
   docker exec crypto-bot tar -czf - /app/logs > bot-logs-backup.tar.gz
   ```

2. **Automated Backups to S3**:
   ```bash
   # Install AWS CLI
   sudo apt install awscli -y
   
   # Configure AWS credentials
   aws configure
   
   # Create backup script
   nano backup.sh
   ```

### Troubleshooting

**Bot not starting?**
```bash
# Check logs
docker logs crypto-bot

# Check config file
docker exec crypto-bot cat config.yaml

# Test connectivity
docker exec crypto-bot ping -c 4 api.binance.com
```

**Telegram not responding?**
- Verify bot token and chat ID
- Check if bot is running: `docker ps`
- Ensure EC2 has internet access

**Trading not executing?**
- Verify API keys have spot trading enabled
- Check account balance
- Review bot logs for errors
- Ensure market is open

## Cost Optimization

1. **EC2 Costs**:
   - Use Reserved Instances for long-term savings (up to 72% discount)
   - Consider Spot Instances for non-critical testing
   - Use t3.small for minimal viable deployment

2. **Data Transfer**:
   - Bot uses minimal bandwidth (~1-2 GB/month)
   - Keep bot in same region as you for Telegram commands

3. **Storage**:
   - 20 GB is sufficient for months of operation
   - Clean old logs periodically

## Support and Contribution

For issues, feature requests, or contributions:
1. Check existing issues
2. Create detailed bug reports
3. Follow code style guidelines
4. Test thoroughly before submitting PRs

## Disclaimer

This bot is for educational purposes. Cryptocurrency trading involves substantial risk of loss. Always:
- Start with paper trading
- Never invest more than you can afford to lose
- Understand the strategies before using real funds
- Monitor your bot regularly


---

Happy Trading! 🚀