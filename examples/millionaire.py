import datetime
import pickle
from pymongo import MongoClient
from sklearn.discriminant_analysis import StandardScaler
import pandas as pd

from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy
import pandas as pd
from backtesting.lib import resample_apply

import matplotlib
#plot the clusters
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from  entropy import ys_sampEntropy
from bayesian_regression import *



def plot_clusters(s):
    # Enable interactive mode
    plt.ion()
    
    # s is a 2d array of cluster centers
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    s = scaler.fit_transform(s)
    
    # Remove NaN or infinite values
    s = np.nan_to_num(s)
    
    # Plot the clusters
    for i in range(len(s)):
        plt.plot(s[i])
    
    # Show the plot and block execution until the plot window is closed
    plt.show(block=True)
    

def findCentersWithEntropy(centers180, centers360, centers720):
    clusters  = 100
            
    # use sample entropy to choose interesting/effective patterns
    entropy180 = np.zeros(clusters)
    entropy360 = np.zeros(clusters)
    entropy720 = np.zeros(clusters)
    
    for i in range(clusters):
        entropy180[i] = ys_sampEntropy(centers180[i, :180])
        entropy360[i] = ys_sampEntropy(centers360[i, :180])
        entropy720[i] = ys_sampEntropy(centers720[i, :180])

    # sort by 20 most interesting, and save these
    # first pattern for 360s  is the flat pattern/ all 0s
    # the lower the entropy, the more interesting the pattern
    IX180 = np.argsort(entropy180)[::-1][:20]
    IX360 = np.argsort(entropy360)[::-1][:20]
    IX720 = np.argsort(entropy720)[::-1][:20]
    kmeans180s = centers180[IX180, :]
    kmeans360s = centers360[IX360, :]
    kmeans720s = centers720[IX720, :]
    
    plot_clusters(kmeans180s)
    plot_clusters(kmeans360s)
    plot_clusters(kmeans720s)
    
 
         

def predict():
    
    br = BayesianRegression()
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('BTCUSDC.csv', delimiter='|')

    # Print column names to verify 'Date' is present
    print("Columns in CSV:", df.columns)

    # Ensure the 'Date' column is parsed as datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    else:
        raise ValueError("'Date' column is not present in the CSV file")

    # Verify the DataFrame
    print(df.head())

    # Retrieve price, v_ask, and v_bid data points from the DataFrame.
    prices = df['Close'].tolist()
    v_ask = df['Open'].tolist()
    v_bid = df['Close'].tolist()


    # Print the first few prices to verify
    print("First few prices:", prices[:10])

    # Divide prices into three, roughly equal sized, periods:
    # prices1, prices2, and prices3.
    [prices1, prices2, prices3] = np.array_split(prices, 3)

    # Divide v_bid into three, roughly equal sized, periods:
    # v_bid1, v_bid2, and v_bid3.
    [v_bid1, v_bid2, v_bid3] = np.array_split(v_bid, 3)

    # Divide v_ask into three, roughly equal sized, periods:
    # v_ask1, v_ask2, and v_ask3.
    [v_ask1, v_ask2, v_ask3] = np.array_split(v_ask, 3)

    # Use the first time period (prices1) to generate all possible time series of
    # appropriate length (180, 360, and 720).
    timeseries180 = br.generate_timeseries(prices1, 60)#180)
    timeseries360 = br.generate_timeseries(prices1, 120)#360)
    timeseries720 = br.generate_timeseries(prices1, 240)#720)

    # Print the current time
    print("Current time before find cluster:", datetime.datetime.now())

    ReEvaluate = False
    #check if cluster model is already saved 
    try:
        with open('model_1minute.pkl', 'rb') as f:
            s1, s2, s3 = pickle.load(f)
    except:
        s1 = None
        s2 = None
        s3 = None

    if s1 is None or s2 is None or s3 is None or ReEvaluate:
        
        #open centers file
        try:
            with open('centers_1minute.pkl', 'rb') as f:
                centers180, centers360, centers720 = pickle.load(f)
        except:
            centers180 = None
            centers360 = None
            centers720 = None
            
        if centers180 is None or centers360 is None or centers720 is None or ReEvaluate:
            # Cluster timeseries180 in 100 clusters using k-means, return the cluster
            # centers (centers180), and choose the 20 most effective centers (s1).
            centers180 = br.find_cluster_centers(timeseries180, 100)
            centers360 = br.find_cluster_centers(timeseries360, 100)
            centers720 = br.find_cluster_centers(timeseries720, 100)
            
            #save centers 180, 360, 720 to file
            with open('centers_1minute.pkl', 'wb') as f:
                pickle.dump((centers180, centers360, centers720), f)
                
            #findCentersWithEntropy(centers180, centers360, centers720)
            
            s1 = br.choose_effective_centers(centers180, 20)
            s2 = br.choose_effective_centers(centers360, 20)
            s3 = br.choose_effective_centers(centers720, 20)
            
            plot_clusters(s1)
            plot_clusters(s2)
            plot_clusters(s3)
            #plot_clusters(centers180)
            #plot_clusters(centers360)
            #plot_clusters(centers720)

            # Save the model to a file
            with open('model_1minute.pkl', 'wb') as f:
                pickle.dump((s1, s2, s3), f)


    # Print the current time
    print("Current time before after cluster:", datetime.datetime.now())


    #check if cluster model is already saved 
    try:
        with open('w_dpi_r_dp_object_1minute.pkl', 'rb') as f:
            w, Dpi_r, Dp,  = pickle.load(f)
    except:
        w = None
        Dpi_r = None
        Dp = None
    
    if w is None or Dpi_r is None or Dp is None:
        # Use the second time period to generate the independent and dependent
        # variables in the linear regression model:
        # Δp = w0 + w1 * Δp1 + w2 * Δp2 + w3 * Δp3 + w4 * r.
        Dpi_r, Dp = br.linear_regression_vars(prices2, v_bid2, v_ask2, s1, s2, s3)

        # Print the current time
        print("Current time before find cluster:", datetime.datetime.now())


        # Find the parameter values w (w0, w1, w2, w3, w4).
        w = br.find_parameters_w(Dpi_r, Dp)

        # Save the model to a file
        with open('w_dpi_r_dp_object_1minute.pkl', 'wb') as f:
            pickle.dump((w, Dpi_r, Dp), f)
        
        
    # Print the current time
    print("Current time before after find parameter:", datetime.datetime.now())


    try:
        with open('dps_1minute.pkl', 'rb') as f:
            dps = pickle.load(f)
    except:
        dps = None
        
    if dps is None:
        # Predict average price changes over the third time period.
        dps = br.predict_dps(prices3, v_bid3, v_ask3, s1, s2, s3, w)


        # Save the model to a file
        with open('dps_1minute.pkl', 'wb') as f:
            pickle.dump(dps, f)  
            
              
    # Predict average price changes over the third time period.
    #dps = predict_dps(prices3, v_bid3, v_ask3, s1, s2, s3, w)
    #384869.7199999515  @  .04 
    
    #get max and min from prices3
    # max_price = max(prices3)
    # min_price = min(prices3)
    # print("Max price:", max_price)
    # print("Min price:", min_price)
    
    # #prices3 get first and last and compute difference
    # first_price = prices3[0]
    # last_price = prices3[-1]
    # print("First price:", first_price)
    # print("Last price:", last_price)
    # print("Price difference:", last_price - first_price)    
    
    t = 1
    max_bank_balance = 0
     # What's your  number?
    for i in range(10):
        t += 1
        bank_balance = br.evaluate_performance(prices3, dps, t=t  , step=1)
        #max bank balance
        if bank_balance > max_bank_balance:
            max_bank_balance = bank_balance
            print('---------------------------------')
            print('Max bank balance:', max_bank_balance)
            print('Threshold:', t)
            
        
        print('---------------------------------')
        

    # print("Bank balance:", bank_balance)
    
    #return prices3 starting at index720 and remove last price, and dps
    #return prices3[720:-1], dps 
    return s1, s2, s3, w
    

#back test this strategy using bt
predict()


    
class BrStrat(Strategy):
    def init(self):
        with open('dps_1minute.pkl', 'rb') as f:
            dps = pickle.load(f)
        
        self.dps = dps
        self.i = 0
          
    def evaluate_performance_single(self, dpsI, t):
        """Evaluate whether to buy or sell based on a single dps value and price.

        

        Args:
            dpsI: A float representing the dps value.
            price: A float representing the current price.
            t: A number representing a threshold.

        Returns:
            A string indicating 'buy', 'sell', or 'hold'.
        
        """
        
        
        if dpsI > t:
            return 'buy'
        elif dpsI < -t:
            return 'sell'
        else:
            return 'hold'      
                  
    def next(self):
        
        signal = self.evaluate_performance_single(self.dps[self.i], 1.4)   # .01)
        if(signal == 'buy'):
            self.buy()
        elif(signal == 'sell'):
            self.sell()
        self.i += 1
            


df = pd.read_csv('BTCUSDC.csv', delimiter='|')

# Print column names to verify 'Date' is present
print("Columns in CSV:", df.columns)

# Ensure the 'Date' column is parsed as datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
else:
    raise ValueError("'Date' column is not present in the CSV file")

# Verify the DataFrame
print(df.head())

prices = df['Close'].tolist()
v_ask = df['Open'].tolist()
v_bid = df['Close'].tolist()
[prices1, prices2, prices3] = np.array_split(prices, 3)
[v_bid1, v_bid2, v_bid3] = np.array_split(v_bid, 3)
[v_ask1, v_ask2, v_ask3] = np.array_split(v_ask, 3)

#split df into 3 parts
df1, df2, df3 = np.array_split(df, 3)

#start df3 at 720 and remove last row
df1 = df1[720:-1]
        

bt = Backtest(df1, BrStrat, cash=100000, commission=.00,  exclusive_orders=True)

output = bt.run()


print(output)
#bt.plot()