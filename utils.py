# RL challenge in the Platform Environment
# Code author: Sergi Andreu

import numpy as np

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def linear_epsilon(e_min, e_max, k, Z):
    ek = np.maximum(e_min, e_max - ((e_max-e_min)*(k-1)) / (Z-1))
    return ek

def exponential_epsilon(e_min, e_max, k, Z):
    ek = np.maximum(e_min, e_max * (e_min/e_max)**((k-1)/(Z-1)))
    return ek