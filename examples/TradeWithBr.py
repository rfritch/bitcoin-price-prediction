
import pickle
from bayesian_regression import *
import json
import pandas as pd
import time
import  json
from typing import Union
import pandas as pd
import logging
from websocket import WebSocketApp
from binance.client import Client
from binance.enums import *
from binance.helpers import round_step_size
from decimal import ROUND_DOWN, Decimal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='BR_bot_log.txt')

# Configure console logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)




SOCKET = "wss://stream.binance.us:9443/ws"  #/BTCUSDC@kline_1m"

IGNORE_COST_BASIS = True
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 31
STOCH_RSI_OVERBOUGHT = 90
STOCH_RSI_OVERSOLD = 10
TRADE_SYMBOL = 'BTCUSDC'
BTC_STEP_SIZE = 0.00001
USD_STEP_SIZE = 0.01
MIN_NOTIONAL = .00001
MAX_CANDLES = 1000
SMA_99 = 99
#candles = []
in_position = False
buy_orders = []

# Initialize Binance Client
# Load API credentials from JSON file
with open('binance_api_key.json', 'r') as f:
    credentials = json.load(f)
    api_key = credentials['api_key']
    api_secret = credentials['api_secret']
    
client = Client(api_key, api_secret, tld='us')

# Candle interval in seconds (1 minute)
candle_interval = 60

# Data storage
trades = []

# Initialize an empty DataFrame to store the candle data
candles = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Symbol to track
symbol = "btcusdc"

def create_candle(ws, message):
    global trades, order_book, candles

    # Parse the received message
    data = json.loads(message)

    # Handle trade data
    if 'e' in data and data['e'] == 'trade':
        trade = {
            'timestamp': data['T'],
            'price': float(data['p']),
            'quantity': float(data['q'])
        }
        trades.append(trade)
        
        trades = trades[-10000:]

    #if the trade info is the same as the last trade info then  increment timestamp by 1 second 
    #this is to make sure that the timestamp is unique
    if len(trades) > 1 and trades[-1]['timestamp'] == trades[-2]['timestamp']:
        trades[-1]['timestamp'] += 1000

    # Aggregate trades into 1s candles
    if trades:
        # Convert timestamp to datetime
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
        
        # Resample to 1-second intervals
        resampled_data = trades_df.set_index('timestamp').resample('1s').agg({
            'price': ['first', 'max', 'min', 'last'],
            'quantity': 'sum'
        })

        # Flatten the MultiIndex columns
        resampled_data.columns = ['open', 'high', 'low', 'close', 'volume']

        # Reset index to get timestamp as a column
        resampled_data = resampled_data.reset_index()

        # Append the new candle data to the candles DataFrame
        candles = pd.concat([candles, resampled_data])
        
        candles = candles.drop_duplicates(subset='timestamp')
        
        # Fill in missing timestamps with previous candle data
        candles = candles.set_index('timestamp').asfreq('1s', method='ffill').reset_index()

        #make sure candles are sorted by timestamp
        #candles = candles.sort_values('timestamp')

        #make sure the candles data is only 10000 rows long
        candles = candles[-10000:]
        
        
        
        # Print the latest candle
        print(candles.tail(30))

        # Clear trades list for the next interval
        trades = []
    
    return candles






# Function to get balances
def get_balances():
    balance = client.get_account()
    usd_balance = float(next(item for item in balance['balances'] if item['asset'] == 'USDC')['free'])
    btc_balance = float(next(item for item in balance['balances'] if item['asset'] == 'BTC')['free'])
    return usd_balance, btc_balance


def order(side, quantity, symbol,order_type=ORDER_TYPE_MARKET):
    try:
        logging.info("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        logging.info(order)
    except Exception as e:
        logging.info("an exception occured - {}".format(e))
        return False

    return True

    
def on_open(ws):
    
    logging.info("### opened ###")
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": [
            f"{symbol}@trade"
        ],
        "id": 1
    }
    
    ws.send(json.dumps(subscribe_message))
    

def on_close(ws):
    logging.info('closed connection')

# Function to calculate weighted average price
def calculate_weighted_average(buy_orders):
    total_cost = sum(order['price'] * order['quantity'] for order in buy_orders)
    total_quantity = sum(order['quantity'] for order in buy_orders)
    returnVal = total_cost / total_quantity if total_quantity > 0 else 0
    logging.info(f"Weighted average price: {returnVal}")
    return returnVal


def get_tick_size(symbol):
    exchange_info = client.get_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    return float(f['tickSize'])
    return None

def round_price(price, tick_size):
    return round(price / tick_size) * tick_size


#function that will get the last orders and check if the last order is a buy or sell order if a but return price
def get_last_buy_orders():
    logging.info("getting last buy orders")
    orders = client.get_all_orders(symbol=TRADE_SYMBOL, limit=100, startTime=int(time.time() * 1000) - (60 * 60 * 36 * 1000))
    
    if len(orders) == 0:
        return []
        
    #filter orders by status
    orders = [order for order in orders if order['status'] == 'FILLED']
    
    #sort orders by time descending
    orders = sorted(orders, key=lambda x: x['time'], reverse=True)
        
    #if latest order is a sell order return None as we should not have outstanding buys
    if len(orders) > 0 and orders[0]['side'] == 'SELL':
        return []
    
    if len(orders) == 0:
        return []    
    
    buy_orders = []
    #get only buy orders after the last sell order
    for order in orders:
        if order['side'] == 'SELL':
            break
        if order['side'] == 'BUY':
            buy_orders.append({
                'price': float(order['price']),
                'quantity': float(order['executedQty']),
                'dateFilled': order['time']
            })
      
    
    if len(buy_orders) == 0:
        return []      
    
    
    #calculate_weighted_average(orders)
    return buy_orders


def round_step_size_decimal(quantity: Union[float, Decimal], step_size: Union[float, Decimal]) -> float:
    """Rounds a given quantity to a specific step size

    :param quantity: required
    :param step_size: required

    :return: decimal
    """
    quantity = Decimal(str(quantity))
    return  Decimal(float(quantity - quantity % Decimal(str(step_size)))).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN)





# Global variables
initial_buy_amount = None



with open('model_1s_07_2024.pkl', 'rb') as f:
    s1, s2, s3 = pickle.load(f)

with open('w_dpi_r_dp_object_1s_07_2024.pkl', 'rb') as f:
    w, Dpi_r, Dp,  = pickle.load(f)


def predict_price(prices, v_bid, v_ask):
    br = BayesianRegression( )
    
    # Predict average price changes over the third time period.
    dps = br.predict_dps(prices, v_bid, v_ask, s1, s2, s3, w)
    logging.info(f"Predicted price change: {dps[-1]}")    
    if dps[-1] > .05:
        return 'buy'
    elif dps[-1] < -.05:
        return 'sell'
    else:
        return 'hold'   
    
    
        

def on_message(ws, message):
    global  in_position, buy_orders, initial_buy_amount
    
    candles = create_candle(ws, message)

    if(len(candles) < 721):
        return
    
    last_price = candles['close'].iloc[-1]
  
    # Slicing the 'close' column for the last 722 elements
    last_720_closes = candles['close'][-722:]
    last_720_closes = last_720_closes.reset_index(drop=True)
   
    last_720_opens = candles['open'][-722:]
    last_720_opens = last_720_opens.reset_index(drop=True)

  
    signal = predict_price(last_720_closes , last_720_closes, last_720_opens)
    logging.info(f"Signal: {signal}  ")
    if signal == 'sell' and in_position:
        logging.info("Overbought! Sell! Sell! Sell!") 
        usd_balance, btc_balance = get_balances()
        
        if  round_step_size_decimal(btc_balance, BTC_STEP_SIZE) > 0:
            
            buy_orders = get_last_buy_orders()
            weighted_avg_price = calculate_weighted_average(buy_orders)
            logging.info(f"Weighted average price: {weighted_avg_price} , current price : {last_price}")
                        
            if IGNORE_COST_BASIS or ((last_price >= (weighted_avg_price * (1 + BTC_STEP_SIZE)))   ):
                # Sell all BTC holdings
                trade_amount_btc = round_step_size_decimal(btc_balance, BTC_STEP_SIZE)
                
                if(trade_amount_btc <= BTC_STEP_SIZE):
                    logging.info("trade amount btc is less than equal to step size nothing to sell")
                    return
                
                logging.info(f"Selling {trade_amount_btc} BTC ")
                
                order =  client.order_market_sell(
                    symbol=TRADE_SYMBOL,
                    quantity=str( Decimal( trade_amount_btc))
                )
                in_position = False
                
                logging.info(f"Limit sell order placed: {order}")
                
                # Reset the initial buy amount
                initial_buy_amount = None
                
            else:
                logging.info("CANT SELL :(  Cost basis is higher than current price")
                return  
        else:
            logging.info("No BTC to sell")
            return
    
    elif signal == 'buy' and not in_position:    
        logging.info("Oversold! Buy! Buy! Buy!")
        usd_balance, btc_balance = get_balances()

        if  round_step_size_decimal(usd_balance, USD_STEP_SIZE) > 0:
            
            if(round_step_size_decimal(usd_balance/last_price, BTC_STEP_SIZE) < MIN_NOTIONAL):
                logging.info("trade amount usd is too low to buy")
                return
                            
            trade_amount_usd = usd_balance 
            trade_amount_btc = trade_amount_usd / last_price
            trade_amount_btc = round_step_size_decimal(trade_amount_btc, BTC_STEP_SIZE)
            
            if trade_amount_btc <= MIN_NOTIONAL:
                logging.info("trade amount btc is too low to buy")
                return     

            logging.info(f"Buying { trade_amount_btc } BTC ")
            
            order = client.order_market_buy(
                symbol=TRADE_SYMBOL,
                quantity=str( trade_amount_btc)
            )
            in_position = True
            
            logging.info(f"Limit buy order placed: {order} ")
            
        else:
            logging.info("No USD to buy")
            return
    
            
              
            
def on_error(ws, error):
    logging.info(error)                

def on_reconnect(ws):
    logging.info('reconnected')    
           
           
if len(buy_orders) == 0:
    buy_orders = get_last_buy_orders()
    
    weightedAvg = calculate_weighted_average(buy_orders)
    logging.info("buy orders")
    logging.info(buy_orders)
   

usd_balance, btc_balance = get_balances()

if(btc_balance >= MIN_NOTIONAL):
    logging.info(f"{btc_balance} btc balance is greater than min notional")
    in_position = True
               
            
ws =   WebSocketApp( url=SOCKET, on_reconnect=on_reconnect, on_open=on_open, on_close=on_close, on_message=on_message ,  on_error=on_error)
ws.run_forever()


   