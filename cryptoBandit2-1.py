import requests
import json
import hmac
import hashlib
import datetime
import time
from termcolor import colored
import signal
import sys
import pandas as pd
from binance.client import Client
import tenacity
import slack_sdk
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
sys.path.append('../')  # Adjust path to include the parent directory
from _secrets import api_key, secret_key

# #############################################################################################################

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
END = '\033[0m'
LIGHTGRAY = '\033[37m'
DARKGRAY = '\033[90m'
EXTRADARKGRAY = '\033[30m'
LIGHTGREEN = '\033[92m'
LIGHTRED = '\033[91m'

# #############################################################################################################

# Define varibles if you want to clone this bot (txt files)
botVersion = 'X' # Change for any other version of your bot (A, B, C ,D, E, ... )
filename_order_id = f'order_id_{botVersion}.txt'
filename_output = f'output_{botVersion}.txt'

# #############################################################################################################

print("")
print(f"Hello from {CYAN}cryptoBandit{END}_{botVersion} (by hexoBandit.com)")

# #############################################################################################################

# Set your Slack API token and channel name
slack_client = WebClient(token='')
channel_id = ""
channel_name = 'cryptobandit_x' # Change for any other version of your bot (A, B, C ,D, E, ... )

# Find the channel ID for the given channel name
try:
    response = slack_client.conversations_list()
    channels = response['channels']
    channel_id = None
    for channel in channels:
        if channel['name'] == channel_name:
            channel_id = channel['id']
            break
except SlackApiError as e:
    print(f"{RED}Slack Error: {e}{END}")

# Send a message to the specified channel
message = f"cryptoBandit_{botVersion} initiated\n" \
           "==========================="

try:
    response = slack_client.chat_postMessage(
        channel=channel_id,
        text=message
    )
    print("==============================================")
    print("Connection to Slack API established")
except SlackApiError as e:
    print(f"{RED}Slack Error: {e}{END}")

# #############################################################################################################

# Set up the Binance API client using your API keys
# api_key = ''
# secret_key = ''
client = Client(api_key, secret_key)

def signature(params):
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    return params

# Set the API endpoint and parameters
endpoint = 'https://api.binance.com/api/v3/account'
params = {}

# Set the timestamp and sign the request
timestamp = int(time.time() * 1000)
params['timestamp'] = timestamp
params = signature(params)

# Set the request headers
headers = {
    'X-MBX-APIKEY': api_key
}

# Send the request and print the response
response = requests.get(endpoint, params=params, headers=headers)
data = json.loads(response.text)
if 'balances' in data:
    for balance in data['balances']:
        if balance['asset'] == 'USDT':
            print("==============================================")
            print(f"Connection to Binance API established")
            print("==============================================")
            print(f"Starting USDT balance:           {GREEN}{balance['free']}{END}")
            print("==============================================")
else:
    print(data)

# #############################################################################################################

#  Promt user to enter variables
symbol = input("Enter the trading symbol: ") 
usd_amount = float(input("Enter the USDT amount: ")) 
buy_threshold = float(input("Enter the buy-low threshold in %: "))/100 
sell_threshold = float(input("Enter the sell-high threshold in %: "))/100 
stop_loss_threshold = float(input("Enter the stop-loss threshold in %: "))/100 
reset_initial_price = float(input("Enter the reset-init-price threshold in %: "))/100
endpoint = 'https://api.binance.com/api/v3/ticker/price'
params = {'symbol': symbol}

buy_price = None
sell_price = None
last_change = 0
print("==============================================")
print("Variables assimilated into the code:")
print("==============================================")
print(f"{YELLOW} - Crypto:{END}           {CYAN}{symbol}{END}")
print(f"{YELLOW} - Buying:           {usd_amount} USDT{END}")
print(f"{YELLOW} - Buy Low:         -{buy_threshold * 100}%{END}")
print(f"{YELLOW} - Sell High:        {sell_threshold * 100}%{END}")
print(f"{YELLOW} - Stop Loss:        {stop_loss_threshold * 100}%{END}")
print(f"{YELLOW} - Reset if bullish: {reset_initial_price * 100}%{END}")
print(f"{YELLOW} - Buy if low & RSI:  < 30 {END}")
print(f"{YELLOW} - Sell if (% change > Sell High) or Stop-Loss{END}")
print("==============================================")

message =  "Assimilated variables:\n" \
           "===========================\n" \
          f"Crypto: *{symbol}*\n" \
          f"Buying: {usd_amount} USDT\n" \
          f"Buy Low: {buy_threshold * 100}%\n" \
          f"Sell High: {sell_threshold * 100}%\n" \
          f"Stop Loss: {stop_loss_threshold * 100}%\n" \
          f"Reset if bullish: {reset_initial_price * 100}%\n" \
          f"Buy if low & RSI: < 30\n" \
          f"Sell if (% change > Sell High) or Stop-Loss\n" \
           "===========================\n" \

try:
    response = slack_client.chat_postMessage(
        channel=channel_id,
        text=message
    )   
    print("Slack message dispatched")
    print("==============================================")
except SlackApiError as e:
    print(f"Slack error: {e}")


# #############################################################################################################

def buy(symbol, usd_amount):
    endpoint = 'https://api.binance.com/api/v3/order'
    # Set the parameters
    params = {
        'symbol': symbol,
        'side': 'BUY',
        'type': 'MARKET',
        'quoteOrderQty': usd_amount # If you want to spend 100 USDT to get some BTC, you can send a MARKET quoteQtyOrder of 100 USDT on the BUY side.
    }
    # Set the timestamp and sign the request
    timestamp = int(time.time() * 1000)
    params['timestamp'] = timestamp
    query_string = '&'.join([f"{k}={v}" for k,v in params.items()])
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    # Set the headers
    headers = {
        'X-MBX-APIKEY': api_key
    }
    # Send the API request and parse the response
    time.sleep(2)
    response = requests.post(endpoint, params=params, headers=headers)
    data = json.loads(response.text)
    if 'orderId' in data:
        # Get the amount of cryptocurrency bought
        amount_bought = float(data['executedQty'])
        print("==============================================")
        print(f"Bought {amount_bought:.4f} {symbol} for {data['cummulativeQuoteQty'][:-4]} USDT")
        print("==============================================")
        print(f"Commission: {data['fills'][0]['commissionAsset']} {data['fills'][0]['commission']}")
        print("==============================================")
        print(f"Order ID: {data['orderId']}")
        # Save the order ID for later use
        order_id = str(data['orderId'])
        with open(filename_order_id, 'w') as f:
            f.write(order_id)
        return amount_bought
    else:
        print(data)


# #############################################################################################################

def get_executed_qty(symbol):
    endpoint = 'https://api.binance.com/api/v3/order'
    with open(filename_order_id, 'r') as f:
        order_id = f.read().strip()
    # Set the parameters
    params = {
        'symbol': symbol,
        'orderId': order_id
    }
    # Set the timestamp and sign the request
    timestamp = int(time.time() * 1000)
    params['timestamp'] = timestamp
    query_string = '&'.join([f"{k}={v}" for k,v in params.items()])
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    # Set the headers
    headers = {
        'X-MBX-APIKEY': api_key
    }
    # Send the API request and parse the response
    response = requests.get(endpoint, params=params, headers=headers)
    data = json.loads(response.text)
    if 'executedQty' in data:
        return float(data['executedQty'])
    else:
        print(data)
        return 0

# #############################################################################################################

def sell(symbol):
    endpoint = 'https://api.binance.com/api/v3/order'
    # Read the order ID from the file
    with open(filename_order_id, 'r') as f:
        order_id = f.read().strip()
    print("==============================================")
    print(f"Previous Order ID: {order_id}")  # Display previous order ID
    # Get the executed quantity from buy function
    executed_qty = get_executed_qty(symbol)
    # Set the parameters
    params = {
        'symbol': symbol,
        'side': 'SELL',
        'type': 'MARKET',
        'quantity': executed_qty
    }
    # Set the timestamp and sign the request
    timestamp = int(time.time() * 1000)
    params['timestamp'] = timestamp
    query_string = '&'.join([f"{k}={v}" for k,v in params.items()])
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    # Set the headers
    headers = {
        'X-MBX-APIKEY': api_key
    }
    # Send the API request and parse the response
    response = requests.post(endpoint, params=params, headers=headers)
    data = json.loads(response.text)
    # Print the response
    if 'orderId' in data:
        print("==============================================")
        print(f"Sold {data['executedQty'][:-4]} {symbol} for {data['cummulativeQuoteQty'][:-4]} USDT")
        print("==============================================")
        print(f"Commission: {data['fills'][0]['commissionAsset']} {data['fills'][0]['commission']}")
        print("==============================================")
        print(f"Order Id: {data['orderId']}")
        return executed_qty
    else:
        print(data)

# #############################################################################################################

def get_balance(USDT):
    endpoint = 'https://api.binance.com/api/v3/account'
    params = {}

    # Set the timestamp and sign the request
    timestamp = int(time.time() * 1000)
    params['timestamp'] = timestamp
    params = signature(params)
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.get(endpoint, params=params, headers=headers)
    data = json.loads(response.text)
    if 'balances' in data:
        for balance in data['balances']:
            if balance['asset'] == 'USDT':
                print("==============================================")
                print(f"Available USDT balance:          {BLUE}{balance['free']}{END}")
                print("==============================================")
    else:
        print(data)

# #############################################################################################################

# Define the function for calculating the RSI with retry logic
@tenacity.retry(wait=tenacity.wait_fixed(10), stop=tenacity.stop_after_delay(300)) #  The stop_after_delay argument specifies that the function should stop retrying after XXX seconds.
def calculate_rsi_with_retry(symbol, interval, start_time, end_time):
    try:
        # Retrieves historical price data for the given symbol 
        data = client.get_historical_klines(symbol, interval, start_time, end_time)
        # Creates a Pandas DataFrame from the retrieved data, where each row represents a candlestick with columns for the timestamp
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df = df.astype(float)
        # Converts the timestamp column to a Pandas datetime object and sets it as the DataFrame index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # calculates the delta (difference) between the closing price of each candlestick and the closing price of the previous candlestick
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculates the average gain and loss over the last 14 candlesticks using a rolling mean
        #avg_gain = gain.rolling(window=14).mean()
        #avg_loss = loss.rolling(window=14).mean()

        # Simple Moving Average (SMA) to match Binance graphs
        avg_gain = gain.rolling(window=14).sum() / 14
        avg_loss = loss.rolling(window=14).sum() / 14

        # Calculates the relative strength (RS) as the ratio of the average gain to the average loss
        rs = avg_gain / avg_loss
        # Calculates the RSI as 100 minus 100 divided by 1 plus the RS
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]
    except requests.exceptions.ReadTimeout:
        # Retry if the request timed out
        raise tenacity.TryAgain

# #############################################################################################################

# Exiting gracefully
def signal_handler(sig, frame):
    print("")
    print("==============================================")
    print("CTRL+C detected.. Exiting the script")
    print("==============================================")

    # Open the file for reading
    with open(filename_output, 'r') as f:
        lines = f.readlines()

    total = 0

    # Loop over each line in the lines list
    for line in lines:

        # Convert the line to a float and add it to the sum
        value = float(line.strip())
        total += value

    # Print the total
    print("==============================================")
    print(f"{PURPLE}The Almighty Running Total: {total:.2f}{END}")
    print("==============================================")

    message = "===========================\n" \
              "CTRL+C detected.. Exiting the script\n" \
              "===========================\n" \
             f"The Almighty Running Total: {total:.2f} \n" \
              "===========================\n" \
              "...........................\n" \

    try:
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text=message
        )   
        print("Slack message dispatched")
        print("==============================================")
        print("")
    except SlackApiError as e:
        print(f"Slack error: {e}")

    sys.exit(0)

# Set up the signal handler
signal.signal(signal.SIGINT, signal_handler)

# #############################################################################################################

# Function to calculate and print the running total
def print_running_total():
    with open(filename_output, 'r') as f:
        lines = f.readlines()

    total = 0

    for line in lines:
        value = float(line.strip())
        total += value

    print("==============================================")
    print(f"{YELLOW}Running Total: {total:.2f}{END}")
    print("==============================================")

# #############################################################################################################

def print_summary(tiktak, symbol, percent_change, rsi, price, sell_price):

    print("==============================================")
    print(f"{YELLOW}{tiktak} - {symbol}{END} threshold reached: ")
    print("==============================================")
    print(f"- Initial price % change: {format_number_with_color(percent_change * 100, 2)}")
    print(f"- Current price: {price:.4f}")
    print(f"- RSI value: {rsi:.2f}")
    print("==============================================")
    print(f"Action triggered at trading price {sell_price} USDT")
    print("==============================================")

# #############################################################################################################

def calculate_profit_or_loss(symbol: str, sell_price: float, buy_price: float,
                             get_executed_qty: callable, filename_output: str,
                             print_running_total: callable, slack_client: slack_sdk.WebClient,
                             channel_id: str):
    result = (sell_price - buy_price) * get_executed_qty(symbol)
    if result > 0:
        print(f"{GREEN}HOORAY, Your profit is: {result:.2f} USDT{END}")
        with open(filename_output, 'a') as f:
            f.write(f"+{result:.2f}\n")
        print_running_total()
        message = f"*{symbol} - profit is: {result:.2f}* USDT :smile: \n"
    elif result < 0:
        print(f"{RED}OH NO, Your loss is: -{result:.2f} USDT{END}")
        with open(filename_output, 'a') as f:
            f.write(f"{result:.2f}\n")
        print_running_total()
        message = f"{symbol} - *loss* is: -{result:.2f} USDT  \n"
    else:
        print(f"{YELLOW}No profit, no loss{END}")
        print_running_total()
        message = f"{symbol} - No profit, no loss \n"

    try:
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        print("==============================================")
        print("Slack message dispatched")
    except SlackApiError as e:
        print(f"Slack error: {e}")

# #############################################################################################################

# Helper function to format a number with light green or light red color based on its sign
def format_number_with_color(number, decimal_places=2):
    color = LIGHTGREEN if number >= 0 else DARKGRAY
    formatted_number = f"{number:.{decimal_places}f}"
    return f"{color}{formatted_number}{END}"

def format_number_fibonacci(number, decimal_places=4):
    color = LIGHTGREEN if number >= price else DARKGRAY
    formatted_number = f"{number:.{decimal_places}f}"
    return f"{color}{formatted_number}{END}"

# #############################################################################################################

# Define a function to reset the bot to the current price
def reset_bot():
    global buy_price
    buy_price = None
    global position_is_open
    position_is_open = False
    global sell_price
    sell_price = None

# #############################################################################################################

kline_interval = Client.KLINE_INTERVAL_1MINUTE

# #############################################################################################################


# Calculate technical indicators
rsi = calculate_rsi_with_retry(symbol, kline_interval, int(time.time() * 1000) - 1000 * 60 * 1000, int(time.time() * 1000))

tik = datetime.datetime.now().strftime("%m-%d %H:%M:%S")

print("Technical indicators check:")
print("==============================================")
print(f"{DARKGRAY}               - RSI value at: {rsi:.2f}{END}")
print("==============================================")


# Set the request headers
headers = {
    'X-MBX-APIKEY': api_key
}

# First run = don't buy
position_is_open = False

# Initialize the candle counter variable
counter = 0

# Open the file for writing and overwrite existing values
with open(filename_output, 'a') as f:

# Continuously fetch the price and take action based on the thresholds
    while True:
        tiktak = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        
        # Query endpoint = 'https://api.binance.com/api/v3/ticker/price' & params = {'symbol': symbol}
        response = requests.get(endpoint, params=params, headers=headers)
        data = json.loads(response.text)
        
        if 'price' in data:
            price = float(data['price'])
            
            # Check if I made a purchase already and set initial buy price
            if buy_price is None:
                buy_price = price
                print(f"{symbol} initial buy price set to {buy_price} USDT")
                print("==============================================")
            percent_change = (price - buy_price) / buy_price
            
            if position_is_open:
                
                rsi = calculate_rsi_with_retry(symbol, kline_interval, int(time.time() * 1000) - 1000 * 60 * 1000, int(time.time() * 1000))

                if percent_change >= sell_threshold or percent_change <= -stop_loss_threshold: # or counter == 10: # <<<<<<<<<<<<<<<<<<<<<<<<<< SELL LOGIC
                        
                    sell_price = price
                        
                    print("==============================================")
                    print(f"{tiktak} - Trashold reached: {percent_change * 100:.2f}%")
                    print("==============================================")
                    print(f"Sell triggered at trading price {sell_price} USDT")
                    print("==============================================")            
                        
                    print_summary("Tiktak", symbol, percent_change, rsi, price, sell_price)
                        
                    calculate_profit_or_loss(symbol, sell_price, buy_price, get_executed_qty, filename_output, print_running_total, slack_client, channel_id)
                    
                    # Execute the sell order
                    sell(symbol)
                    
                    # Update variables and balances
                    position_is_open = False
                    updated_usdt_balance = get_balance('USDT')
                    buy_price = None
                    sell_price = None    
                    counter = 0 # Reset the counter after a sell order is executed

                else:
                    rsi = calculate_rsi_with_retry(symbol, kline_interval, int(time.time() * 1000) - 1000 * 60 * 1000, int(time.time() * 1000))

                    counter += 1 # Increment the candle counter  

                    print(f"{DARKGRAY}{tiktak} - New position price: {percent_change * 100:.2f}%{END}")
                    print(f"{DARKGRAY}               - RSI value at: {rsi:.2f}{END}")
                    print(f"{DARKGRAY}               - Candle Counter: {counter}{END}")
         
            else:

                rsi = calculate_rsi_with_retry(symbol, kline_interval, int(time.time() * 1000) - 1000 * 60 * 1000, int(time.time() * 1000))

                if rsi < 30 and percent_change <= -buy_threshold: # <<<<<<<<<<<<<<<<<<<<<<<<<<  BUY LOGIC
                    
                    buy_price = price 
                    
                    print("==============================================")
                    print(f"{tiktak} - Threshold reached: {percent_change * 100:.2f}%")
                    print(f"                 with RSI below 30: {rsi:.2f}")
                    
                    buy(symbol, usd_amount)
                    
                    print("==============================================")
                    print(f"Buy triggered at trading price {buy_price} USDT")
                    print_summary("Tiktak", symbol, percent_change, rsi, price, sell_price)
                                      
                    position_is_open = True
                    updated_usdt_balance = get_balance('USDT')
                    sell_price = None
                    counter = 0 # Reset the candle counter when a buy order is executed

                elif not position_is_open and percent_change >= reset_initial_price: #0.5: # not position_is_open checks if the bot has an open position or not. If the bot has not bought anything yet, then position_is_open is False, so not position_is_open is True.
                    reset_bot()
                    print("==============================================")
                    print(f"Price has risen more than {reset_initial_price * 100}% : ({percent_change * 100:.2f}%)")  
                    print("==============================================")
                    print(f"Initial price optimized to match the market")
                    print("==============================================")

                else:
                    # If there was no action (buy or sell), print the current progress percent change
                    print(f"{DARKGRAY}{tiktak} - New position price: {percent_change * 100:.2f}%{END}")
                    print(f"{DARKGRAY}               - RSI value at: {rsi:.2f}{END}")
                    print(f"{DARKGRAY}               - Candle Counter: {counter}{END}")

            last_change = abs(percent_change)
        else:
            print(data)

        time.sleep(60)

        pass # Beutiful exit upon CTRL+C