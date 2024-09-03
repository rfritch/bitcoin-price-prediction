"""Bayesian regression for latent source model and Bitcoin.

This module implements the 'Bayesian regression for latent source model' method
for predicting price variation of Bitcoin. You can read more about the method
at https://arxiv.org/pdf/1410.1231.pdf.
"""
import numpy as np
#import bigfloat as bg
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.cluster import KMeans

from mpmath import mp, exp
from concurrent.futures import ProcessPoolExecutor


class BayesianRegression:

    def __init__(self):
        self.timeseries1 = 60
        self.timeseries2 = 120
        self.timeseries3 = 240
        
    
    def generate_timeseries(self, prices, n):
        """Use the first time period to generate all possible time series of length n
        and their corresponding label.

        Args:
            prices: A numpy array of floats representing prices over the first time
                period.
            n: An integer (180, 360, or 720) representing the length of time series.

        Returns:
            A 2-dimensional numpy array of size (len(prices)-n) x (n+1). Each row
            represents a time series of length n and its corresponding label
            (n+1-th column).
        """
        m = len(prices) - n
        ts = np.empty((m, n + 1))
        for i in range(m):
            ts[i, :n] = prices[i:i + n]
            ts[i, n] = prices[i + n] - prices[i + n - 1]
        return ts


    def find_cluster_centers(self,timeseries, k):
        """Cluster timeseries in k clusters using k-means and return k cluster centers.

        Args:
            timeseries: A 2-dimensional numpy array generated by generate_timeseries().
            k: An integer representing the number of centers (e.g. 100).

        Returns:
            A 2-dimensional numpy array of size k x num_columns(timeseries). Each
            row represents a cluster center.
        """
        k_means = KMeans(n_clusters=k)
        k_means.fit(timeseries)
        return k_means.cluster_centers_



    # To find the 20 most effective cluster centers from the given text, you can follow these steps:

    # Cluster the Data: Use the k-means algorithm to cluster your data into 100 clusters. This step is already mentioned in your text.
    # Evaluate Cluster Effectiveness: Determine the effectiveness of each cluster. This can be done using various metrics such as:
    # Inertia: Measures the sum of squared distances of samples to their closest cluster center.
    # Silhouette Score: Measures how similar a point is to its own cluster compared to other clusters.
    # Cluster Purity: Measures the extent to which clusters contain a single class.
    # Select Top Clusters: Rank the clusters based on the chosen effectiveness metric and select the top 20 clusters.
    # Extract Cluster Centers: Once you have identified the top 20 clusters, extract their centers. These centers represent the most effective clusters.
    # 

    def choose_effective_centers(self,centers, n):
        """Choose n most effective cluster centers with high price variation."""
        #centersSum = centers[np.argsort(np.ptp(centers, axis=1).sum() )[-n:]]
        
        return centers[np.argsort(np.ptp(centers, axis=1) )[-n:]]


    def predict_dpi(self, x, s):
        """Predict the average price change Δp_i, 1 <= i <= 3.

        Args:
            x: A numpy array of floats representing previous 180, 360, or 720 prices.
            s: A 2-dimensional numpy array generated by choose_effective_centers().

        Returns:
            A big float representing average price change Δp_i.
        """
        num = 0
        den = 0
        for i in range(len(s)):
            y_i = s[i, len(x)]
            x_i = s[i, :len(x)]
            
            # Assuming norm is a function that returns a float or an mpmath-compatible type
            # If norm returns a float, convert it to mpmath's mpf type
            norm_value = mp.mpf(norm(x - x_i))

            # Replace bigfloat exp with mpmath exp
            exp_value = exp(-0.25 * norm_value ** 2)
            #exp = bg.exp(-0.25 * norm(x - x_i) ** 2)
            
            num += y_i * exp_value
            den += exp_value
        return num / den


    def linear_regression_vars(self,prices, v_bid, v_ask, s1, s2, s3):
        """Use the second time period to generate the independent and dependent variables
        in the linear regression model Δp = w0 + w1 * Δp1 + w2 * Δp2 + w3 * Δp3 + w4 * r.

        Args:
            prices: A numpy array of floats representing prices over the second time
                period.
            v_bid: A numpy array of floats representing total volumes people are
                willing to buy over the second time period.
            v_ask: A numpy array of floats representing total volumes people are
                willing to sell over the second time period.
            s1: A 2-dimensional numpy array generated by choose_effective_centers()
            s2: A 2-dimensional numpy array generated by choose_effective_centers().
            s3: A 2-dimensional numpy array generated by choose_effective_centers().

        Returns:
            A tuple (X, Y) representing the independent and dependent variables in
            the linear regression model. X is a 2-dimensional numpy array and each
            row represents [Δp1, Δp2, Δp3, r]. Y is a numpy array of floats and
            each array element represents Δp.
        """
        
        lens1 = self.timeseries1
        lens2 = self.timeseries2
        lens3 = self.timeseries3
        
        X = np.empty((len(prices) - lens3 + 1, 4))
        Y = np.empty(len(prices) - lens3 + 1)
        for i in range(lens3, len(prices) - 1):
            dp = prices[i + 1] - prices[i]
            dp1 = self.predict_dpi(prices[i - lens1:i], s1)
            dp2 = self.predict_dpi(prices[i - lens2:i], s2)
            dp3 = self.predict_dpi(prices[i - lens3:i], s3)
            r = (v_bid[i] - v_ask[i]) / (v_bid[i] + v_ask[i])
            X[i - lens3, :] = [dp1, dp2, dp3, r]
            Y[i - lens3] = dp
        return X, Y


    def find_parameters_w(self,X, Y):
        """Find the parameter values w for the model which best fits X and Y.

        Args:
            X: A 2-dimensional numpy array representing the independent variables
                in the linear regression model.
            Y: A numpy array of floats representing the dependent variables in the
                linear regression model.

        Returns:
            A tuple (w0, w1, w2, w3, w4) representing the parameter values w.
        """
        clf = linear_model.LinearRegression()
        clf.fit(X, Y)
        w0 = clf.intercept_
        w1, w2, w3, w4 = clf.coef_
        return w0, w1, w2, w3, w4


    def predict_dps(self,prices, v_bid, v_ask, s1, s2, s3, w):
        """Predict average price changes (final estimations Δp) over the third
        time period.

        Args:
            prices: A numpy array of floats representing prices over the third time
                period.
            v_bid: A numpy array of floats representing total volumes people are
                willing to buy over the third time period.
            v_ask: A numpy array of floats representing total volumes people are
                willing to sell over the third time period.
            s1: A 2-dimensional numpy array generated by choose_effective_centers()
            s2: A 2-dimensional numpy array generated by choose_effective_centers().
            s3: A 2-dimensional numpy array generated by choose_effective_centers().
            w: A tuple (w0, w1, w2, w3, w4) generated by find_parameters_w().

        Returns:
            A numpy array of floats. Each array element represents the final
            estimation Δp.
        """
        
        lens1 = self.timeseries1
        lens2 = self.timeseries2
        lens3 = self.timeseries3
        
        dps = []
        w0, w1, w2, w3, w4 = w
        for i in range(lens3, len(prices) - 1):
            dp1 = self.predict_dpi(prices[i - lens1:i], s1)
            dp2 = self.predict_dpi(prices[i - lens2:i], s2)
            dp3 = self.predict_dpi(prices[i - lens3:i], s3)
            r = (v_bid[i] - v_ask[i]) / (v_bid[i] + v_ask[i])
            dp = w0 + w1 * dp1 + w2 * dp2 + w3 * dp3 + w4 * r
            dps.append(float(dp))
        return dps


    def evaluate_performance(self,prices, dps, t, step):
        """Use the third time period to evaluate the performance of the algorithm.

        Args:
            prices: A numpy array of floats representing prices over the third time
                period.
            dps: A numpy array of floats generated by predict_dps().
            t: A number representing a threshold.
            step: An integer representing time steps (when we make trading decisions).

        Returns:
            A number representing the bank balance.
        """
        bank_balance = 0
        position = 0
        max_bank_balance = 0
        min_bank_balance = 0
        
        for i in range(self.timeseries3, len(prices) - 1, step):
            dpsI = dps[i - self.timeseries3]
            #print(dpsI)
            #print(bank_balance)
            
            # long position - BUY
            if dpsI > t and position <= 0:
                position += 1
                bank_balance -= prices[i]
            # short position - SELL
            if dpsI < -t and position >= 0:
                position -= 1
                bank_balance += prices[i]

            #calculate min and max bank balance
            if bank_balance < min_bank_balance:
                min_bank_balance = bank_balance
            if bank_balance > max_bank_balance:
                max_bank_balance = bank_balance
                
        print("Max bank balance:", max_bank_balance)
        print("Min bank balance:", min_bank_balance)
            
        print(bank_balance)
        print(position)
        # sell what you bought
        if position == 1:
            bank_balance += prices[len(prices) - 1]
        # pay back what you borrowed
        if position == -1:
            bank_balance -= prices[len(prices) - 1]
            
        print(bank_balance)
        return bank_balance
