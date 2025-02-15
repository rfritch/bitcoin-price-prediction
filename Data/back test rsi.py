import os
import pandas as pd
import pytz
from datetime import datetime


# Define the statistical analysis function
def statistical_analysis(data, window=24*60):  # 24 hours * 60 minutes
    data['mean_price'] = data['Close'].rolling(window=window).mean()
    data['std_dev'] = data['Close'].rolling(window=window).std()
    
    #calcuate the high and low of the last 24 hours
    data['high_24h'] = data['High'].rolling(window=window).max()
    data['low_24h'] = data['Low'].rolling(window=window).min()
    return data


# Directory containing the data files
data_dir = 'Data'

# Initialize an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Function to standardize timestamps
def standardize_timestamps(timestamps):
    if len(str(timestamps.iloc[0])) > 13:
        return pd.to_datetime(timestamps, unit='us').dt.tz_localize('UTC')
        
    else:
        return pd.to_datetime(timestamps, unit='ms').dt.tz_localize('UTC') 

# Read and combine data files
for file in os.listdir(data_dir):
    if file.startswith('BTCUSDC-1m-202'):
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path)
        

        if data.columns[0] != 'Date':
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'DateClosed', '2', '3', '4', '5', '6']

        # Standardize Date and DateClosed columns
        data['Date'] = standardize_timestamps(data['Date'])
        data['DateClosed'] = standardize_timestamps(data['DateClosed'])
        
        combined_data = pd.concat([combined_data, data])

# Sort and reset index
combined_data.sort_values(by='DateClosed', inplace=True)
combined_data.reset_index(drop=True, inplace=True)


# Perform statistical analysis
combined_data = statistical_analysis(combined_data)



# Define the timezone for PST
pst = pytz.timezone('US/Pacific')


print (len(combined_data))
print(combined_data['Close'].max())
print(combined_data['Close'].min())
print(combined_data['Close'].median())
print(combined_data['Close'].mean())
print(combined_data['Close'].std())


#print first 10 rows of the data
print(combined_data.head(10))



max_profit_loss = -10000000000000000000

for exit_loss in range(150, 2000, 100):
    
    print(exit_loss)
        
    for exit_profit in range(150, 2000, 100):
    
        profit_loss = 0
        entry_price = 0
        position = None
        trades = []
                
        print (exit_profit)
      
        
        for i in range(1, len(combined_data)):  # Start from 1 to avoid indexing issues
            
            current_time = combined_data['DateClosed'].iloc[i].tz_convert(pst)
            if (current_time.hour == 14 and current_time.minute >= 0 and current_time.minute < 60) or \
            (current_time.weekday() == 4 and current_time.hour >= 14) or \
            (current_time.weekday() == 5) or \
            (current_time.weekday() == 6 and current_time.hour < 14):
                continue  # Skip trading during the specified times


            if position is None:
                
                fee = (combined_data['Close'].iloc[i] * .001)
                
                #check that mean is defined and that the price is below the mean - std_dev
                if pd.notnull(combined_data['mean_price'].iloc[i]) and pd.notnull(combined_data['std_dev'].iloc[i]):
                    if combined_data['Close'].iloc[i] < (combined_data['mean_price'].iloc[i] - combined_data['std_dev'].iloc[i]):
    
                        #print(f"Buying at {combined_data['Close'].iloc[i]} on {combined_data['DateClosed'].iloc[i]}")
                        #print(f"\tMean price yesterday {combined_data['mean_price'].iloc[i]} std dev {combined_data['std_dev'].iloc[i]}"  )
                        #print(f"\tHigh {combined_data['high_24h'].iloc[i]} Low {combined_data['low_24h'].iloc[i]}" )
                                         
                        position = 'long'
                        entry_price = combined_data['Close'].iloc[i] 
                        entry_time = combined_data['DateClosed'].iloc[i]

                        
                    elif combined_data['Close'].iloc[i] > (combined_data['mean_price'].iloc[i] + combined_data['std_dev'].iloc[i]):

                        #print(f"Selling Short at {combined_data['Close'].iloc[i]} on {combined_data['DateClosed'].iloc[i]}")
                        #print(f"\tMean price yesterday {combined_data['mean_price'].iloc[i]} std dev {combined_data['std_dev'].iloc[i]}"  )
                        #print(f"\thigh {combined_data['high_24h'].iloc[i]} low {combined_data['low_24h'].iloc[i]}" )

                        position = 'short'
                        entry_price = combined_data['Close'].iloc[i] 
                        entry_time = combined_data['DateClosed'].iloc[i]

                 
            elif position == 'short':
                
                if combined_data['Close'].iloc[i] <= entry_price - exit_profit:
                    #print(f"\tClosing at {combined_data['Close'].iloc[i]} on {combined_data['DateClosed'].iloc[i]}")
                    #print(f"\tlength {(combined_data['DateClosed'].iloc[i] - entry_time).total_seconds() / 60.0} minutes")

                    position = None  # Reset position after selling
                    gain = entry_price - combined_data['Close'].iloc[i]  - fee  #fees  # Update profit/loss
                    profit_loss +=  gain
                    
                    #print(f"GAIN Short {gain} , current profit_loss {profit_loss} " )
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': combined_data['DateClosed'].iloc[i],
                        'entry_price': entry_price,
                        'exit_price': combined_data['Close'].iloc[i],
                        'profit': gain,
                    })
                    
                elif combined_data['Close'].iloc[i] >= entry_price + exit_loss:
                    #print(f"\tClosing at {combined_data['Close'].iloc[i]} on {combined_data['DateClosed'].iloc[i]}")
                    #print(f"\tlength {(combined_data['DateClosed'].iloc[i] - entry_time).total_seconds() / 60.0} minutes")

                    position = None
                    loss = entry_price - combined_data['Close'].iloc[i] - fee
                    profit_loss += loss     # Update profit/loss
                    
                    #print(f"LOSS Short {loss} , current profit_loss {profit_loss} " )
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': combined_data['DateClosed'].iloc[i],
                        'entry_price': entry_price,
                        'exit_price': combined_data['Close'].iloc[i],
                        'profit': loss,
                    })
                                                
            elif position == 'long':
                
                if combined_data['Close'].iloc[i] >= entry_price + exit_profit:
                    
                    #print(f"\tClosing at {combined_data['Close'].iloc[i]} on {combined_data['DateClosed'].iloc[i]}")
                    #print(f"\tlength {(combined_data['DateClosed'].iloc[i] - entry_time).total_seconds() / 60.0} minutes")
                        
                    position = None  # Reset position after selling
                    gain = combined_data['Close'].iloc[i] - entry_price - fee #fees  # Update profit/loss
                    profit_loss += gain    #fees  # Update profit/loss
                    
                    #print(f"GAIN Long {gain} , current profit_loss {profit_loss} " )
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': combined_data['DateClosed'].iloc[i],
                        'entry_price': entry_price,
                        'exit_price': combined_data['Close'].iloc[i],
                        'profit': gain,
                    })
                    
                    
                elif combined_data['Close'].iloc[i] <= entry_price - exit_loss:
                    
                    #print(f"\tClosing at {combined_data['Close'].iloc[i]} on {combined_data['DateClosed'].iloc[i]}")
                    #print(f"\tlength {(combined_data['DateClosed'].iloc[i] - entry_time).total_seconds() / 60.0} minutes")

                    position = None
                    loss =  combined_data['Close'].iloc[i] - entry_price - fee  # Update profit/loss
                    profit_loss += loss   # Update profit/loss
                    
                    #print(f"LOSS Long {loss} , current profit_loss {profit_loss} " )
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': combined_data['DateClosed'].iloc[i],
                        'entry_price': entry_price,
                        'exit_price': combined_data['Close'].iloc[i],
                        'profit': loss,
                    })
                    
                    
        # Print the final profit/loss
        print(f"Final Profit/Loss {profit_loss}  after adjusting positions  : {(profit_loss * .01   )}  Exit Profit {exit_profit} Exit Loss {exit_loss}  Trades {len(trades)}")
        
        
        if profit_loss >= max_profit_loss:
            max_profit_loss = profit_loss
            best_profit = exit_profit
            best_trades = trades.copy()
            
            
#print data start date and end date
print(f"Data from {combined_data['DateClosed'].iloc[0]} to {combined_data['DateClosed'].iloc[-1]}")
print(f"Max Profit/Loss at 100 positions : {max_profit_loss} adjusted for 1 positions : {max_profit_loss * .01 } ")
print(f"Best Profit {best_profit} ")


if(len(best_trades) > 0):
    trades = best_trades.copy()

#---------------------------------------------------
# After all trades are processed, compute and display average trade length/ profit
# and then plot the trades
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if len(trades) > 0:
    # Calculate average trade length in minutes and average profit
    trade_lengths = []
    trade_profits = []
    for t in trades:
        length_minutes = (
            pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time'])
        ).total_seconds() / 60.0
        trade_lengths.append(length_minutes)
        trade_profits.append(t['profit'])

    # Calculate winning percentage
    winning_trades = [p for p in trade_profits if p > 0]
    losing_trades = [p for p in trade_profits if p < 0]
    total_trades = len(trades)
    
    print(len(winning_trades))
    
    winning_percentage = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0.0
    print(f"Winning Percentage: {winning_percentage:.2f}%")
    
    # Print total number of trades
    print(f"Total Trades: {total_trades}")
    print(f"Total Profit/Loss: {sum(trade_profits)}")
    
    
    # Print highest profit and loss
    print(f"Highest Profit: {max(trade_profits)}")
    print(f"Highest Loss: {min(trade_profits)}")
    
    # Print average loss and average profit
    avg_loss = np.mean([p for p in trade_profits if p < 0])
    avg_profit = np.mean([p for p in trade_profits if p > 0])
    print(f"Average Loss: {avg_loss:.2f}")
    print(f"Average Profit: {avg_profit:.2f}")
    
    avg_length = np.mean(trade_lengths)
    print(f"Average Trade Length: {avg_length:.2f} minutes")
    
    # Calculate most losing trades in a row and most winning trades in a row
    max_losing_streak = 0
    max_winning_streak = 0
    current_losing_streak = 0
    current_winning_streak = 0
    
    for profit in trade_profits:
        if profit > 0:
            current_winning_streak += 1
            current_losing_streak = 0
        else:
            current_losing_streak += 1
            current_winning_streak = 0
        
        if current_winning_streak > max_winning_streak:
            max_winning_streak = current_winning_streak
        if current_losing_streak > max_losing_streak:
            max_losing_streak = current_losing_streak
    
    print(f"Most Winning Trades in a Row: {max_winning_streak}")
    print(f"Most Losing Trades in a Row: {max_losing_streak}")
    
else:
    print("No trades were made, so no average length or profit to report.")

# Plot the trades
#plt.figure(figsize=(12, 6))
# #plt.plot(combined_data['DateClosed'], combined_data['Close'], label='Close Price')

# # To avoid repeated legend entries, track if we've labeled a buy or sell yet
# buy_labeled = False
# sell_labeled = False

# for t in trades:
#     if not buy_labeled:
#         plt.scatter(t['entry_time'], t['entry_price'], color='green', marker='^', label='Buy')
#         buy_labeled = True
#     else:
#         plt.scatter(t['entry_time'], t['entry_price'], color='green', marker='^')
    
#     if not sell_labeled:
#         plt.scatter(t['exit_time'], t['exit_price'], color='red', marker='v', label='Sell')
#         sell_labeled = True
#     else:
#         plt.scatter(t['exit_time'], t['exit_price'], color='red', marker='v')

# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('Trades')
# plt.legend()
# plt.show()
# # ...existing code...