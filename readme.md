# cryptoBandit 2.0 - Buy low & Sell High

Binance API color-coded, Slack-integrated crypto trading automated bot.

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

1. Set your Slack API token and channel name (line 50)
1. Set up the Binance API client using your API keys (line 82)

## Example Run

<img width="335" alt="image" src="https://github.com/user-attachments/assets/19ecf258-300c-4187-a073-6a7bff9e562e">


