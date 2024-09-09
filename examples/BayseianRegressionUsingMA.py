import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pymc3 as pm

# Load the time series data
df = pd.read_csv('time_series_data.csv')

# Ensure the 'Date' column is parsed as datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df[['Value']])

# Create additional features (e.g., moving average)
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df.dropna(inplace=True)  # Drop rows with NaN values

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# Prepare the data for Bayesian regression
X_train = train[['Close', 'MA_10', 'MA_30']].values
y_train = train['Close'].values
X_test = test[['Close', 'MA_10', 'MA_30']].values
y_test = test['Close'].values

# Bayesian regression model using PyMC3
with pm.Model() as model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=X_train.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = alpha + pm.math.dot(X_train, beta)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_train)

    # Inference
    trace = pm.sample(1000, tune=1000, cores=2)

# Posterior predictive checks
with model:
    ppc = pm.sample_posterior_predictive(trace, samples=500, var_names=['Y_obs'])

# Evaluate the model
y_pred = np.mean(ppc['Y_obs'], axis=0)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(test.index, y_test, label='True Values')
plt.plot(test.index, y_pred, label='Predicted Values')
plt.legend()
plt.show()