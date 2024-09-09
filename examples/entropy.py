#adapted from   https://github.com/panditanvita/BTCpredictor/blob/master/ys_sampEntropy.m   author by YangSong 2010.11.16 C230

import numpy as np

def ys_sampEntropy(xdata):
    m = 2
    n = len(xdata)
    r = 0.2 * np.std(xdata)  # Matching tolerance value
    cr = []
    gn = 1
    gnmax = m
    
    while gn <= gnmax:
        d = np.zeros((n - m + 1, n - m))  # Distance matrix
        x2m = np.zeros((n - m + 1, m))  # Embedding matrix
        cr1 = np.zeros(n - m + 1)  # Correlation matrix
        k = 0
        
        for i in range(n - m + 1):
            x2m[i] = xdata[i:i+m]
        
        for i in range(n - m + 1):
            for j in range(n - m + 1):
                if i != j:
                    d[i, k] = np.max(np.abs(x2m[i] - x2m[j]))  # Maximum absolute difference
                    k += 1
            k = 0
        
        for i in range(n - m + 1):
            l = np.sum(d[i] < r)  # Count the number of elements less than r
            cr1[i] = l
        
        cr1 = (1 / (n - m)) * cr1
        sum1 = 0
        
        for i in range(n - m + 1):
            if cr1[i] != 0:
                sum1 += cr1[i]
        
        cr1 = (1 / (n - m + 1)) * sum1
        cr.append(cr1)
        gn += 1
        m += 1
    
    sampEntropy = np.log(cr[0]) - np.log(cr[1])
    
    #check if sampEntropy is NaN something went wrong and  dont use it  
    if np.isnan(sampEntropy):
        sampEntropy = 1000
    
    return sampEntropy
