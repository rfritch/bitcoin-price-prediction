import requests
from pytz import utc
from datetime import datetime
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler

"""Script to gather market data from OKCoin Spot Price API."""

client = MongoClient()
database = client['okcoindb']
collection = database['historical_data']

def tick():
    """Gather market data from OKCoin Spot Price API and insert them into a
       MongoDB collection."""
       #/api/v5/market/tickers?instType=SPOT
    ticker = requests.get('https://www.okcoin.com/api/v5/market/ticker?instId=BTC-USD').json()
    #print(ticker)
    
    depth = requests.get('https://www.okcoin.com/api/v5/market/books?instId=BTC-USD&sz=60').json()
    #print(depth)    
    
#    print (ticker['data'][0]['ts'])
#    for bid in depth['data'][0]['bids']:
#        print(bid)
    
    #depth , base quanity, not used, number of orders
    #['58553.61', '0.0037', '0', '1']
    
    date = datetime.fromtimestamp( timestamp=float(ticker['data'][0]['ts'] )/ 1000, tz=utc)
    price = float( ticker['data'][0]['last'] )
    v_bid = sum([ float(bid[1]) * float(bid[3]) for bid in depth['data'][0]['bids']])
    v_ask = sum([ float(ask[1]) * float(ask[3]) for ask in depth['data'][0]['asks']])
    print(date, price, v_bid, v_ask)
    collection.insert_one({'date': date, 'price': price, 'v_bid': v_bid, 'v_ask': v_ask})
    

def main():
    """Run tick() at the interval of every ten seconds."""
    scheduler = BlockingScheduler(timezone=utc)
    scheduler.add_job(tick, 'interval', seconds=10)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == '__main__':
    main()
