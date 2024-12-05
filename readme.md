# cryptoBandit 2.0 - Buy low & Sell High

CryptoBandit is your color-coded, Slack-integrated crypto trading assistant. 

What it does:
1. Connects to Binance, checks your USDT balance, and obeys your buy/sell rules.
2.	Calculates RSI and tracks percent changes to decide trades.
3.	Logs profits/losses with dramatic colors and updates Slack.
4.	Handles errors gracefully and exits politely on CTRL+C.

It’s stylish, efficient, and totally avoids rogue trades (unless your inputs are wild). Keep your API keys secure, and let cryptoBandit hustle for you! 🥷

## Installation

1. Virtual Env

    `python3 -m venv venv`
    
    `source venv/bin/activate`

1. Dependencies

    `pip install -r requirements.txt`