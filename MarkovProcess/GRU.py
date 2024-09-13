import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Load data
# Assuming 'data.csv' contains the 1-minute candle data with columns: ['datetime', 'open', 'high', 'low', 'close', 'volume']

df1 = pd.read_csv('.\\Data\\BTCUSDC-1m-2024-01.csv')
df2 = pd.read_csv('.\\Data\\BTCUSDC-1m-2024-02.csv')
df3 = pd.read_csv('.\\Data\\BTCUSDC-1m-2024-03.csv')
df4 = pd.read_csv('.\\Data\\BTCUSDC-1m-2024-04.csv')
df5 = pd.read_csv('.\\Data\\BTCUSDC-1m-2024-05.csv')
df6 = pd.read_csv('.\\Data\\BTCUSDC-1m-2024-06.csv')
df7 = pd.read_csv('.\\Data\\BTCUSDC-1m-2024-07.csv')

# Append dataframes to a single dataframe
data = pd.concat([df1, df2, df3, df4, df5, df6, df7])

# Print column names to verify 'Date' is present
print("Columns in CSV:", data.columns)

# Ensure the 'Date' column is parsed as datetime
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], unit='ms')
    data.set_index('Date', inplace=True)
else:
    raise ValueError("'Date' column is not present in the CSV file")

# Verify the DataFrame
print(data.head())

# Feature extraction
data['returns'] = data['Close'].pct_change()
data['log_returns'] = np.log1p(data['returns'])
data['volatility'] = data['returns'].rolling(window=10).std()
data['high_low_range'] = data['High'] - data['Low']
data['moving_avg_10'] = data['Close'].rolling(window=10).mean()
data['moving_avg_50'] = data['Close'].rolling(window=50).mean()
data.dropna(inplace=True)

# Prepare data for GRU
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['log_returns', 'volatility', 'high_low_range', 'moving_avg_10', 'moving_avg_50', 'Close']])

# Create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, -1])  # Use the closing price as the target
    return np.array(X), np.array(y)

seq_length = 60  # Use past 60 minutes to predict the next minute
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Model file path
model_file_path = 'gru_model_v2.h5'

# Check if model file exists
if os.path.exists(model_file_path):
    # Load the model
    model = tf.keras.models.load_model(model_file_path)
    print("Model loaded from file.")
else:
    # Build GRU model with improvements
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(100, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(100),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train GRU model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Save the model to file
    model.save(model_file_path)
    print("Model saved to file.")

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()


model.summary() 
#model.compile_metrics = ['accuracy']


# Make predictions
y_pred = model.predict(X_test)



# Inverse transform the predictions and actual values
y_test_actual = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), scaled_data.shape[1]-1)), y_test.reshape(-1, 1)), axis=1))[:, -1]
y_pred_actual = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), scaled_data.shape[1]-1)), y_pred), axis=1))[:, -1]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
print(f"RMSE: {rmse}")

#calculate mean absolute percentage error
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
print(f"Mean Absolute Percentage Error: {mape}")

#calctulate mean absolute error
mae = np.mean(np.abs(y_test_actual - y_pred_actual))
print(f"Mean Absolute Error: {mae}")

#calculate r2 score
r2 = r2_score(y_test_actual, y_pred_actual)
print(f"R2 Score: {r2}")


# Plot predictions vs actual prices
#plot last 100 predictions vs actual prices
y_test_actual = y_test_actual[-100:]
y_pred_actual = y_pred_actual[-100:]

#print last 100 predictions vs actual prices
print("Last 100 predictions vs actual prices")
print("Actual Prices: ", y_test_actual)
print("Predicted Prices: ", y_pred_actual)



plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_pred_actual, label='Predicted Price')
plt.legend()
plt.show()