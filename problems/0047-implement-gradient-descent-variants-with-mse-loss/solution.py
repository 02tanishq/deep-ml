from math import ceil

import numpy as np


def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m = X.shape[0]
    n = X.shape[1]
    gradient = lambda theta,X,y : (2/X.shape[0]) * X.T @ (X @  theta - y)
    if method == 'batch':
        index = 1
    elif method == 'stochastic':
        index = m
    else:
        index = ceil(m/batch_size)
    
    for i in range(n_epochs):
        for j in range(index):
            if method == 'batch':
                start,end = 0,m
            elif method == 'stochastic':
                start,end = j,j+1
            else:
                start = j * batch_size
                end = min(start+batch_size,m)
                
            delta = gradient(weights,X[start:end],y[start:end])
                    

            weights = weights - learning_rate * delta
    
    return  weights
