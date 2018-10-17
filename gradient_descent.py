import numpy as np
import math

def df(x):
    return np.array([2 * x[0]])

def gradient_descent(df, x_init, eta):
    X = df(x_init)
    step = 0
    while (np.linalg.norm(X) > 0.0001):
        step += 1
        Y = x_init - (eta * X)
        x_init = Y
        X = df(Y)
    #print("num steps is ", step)
    return Y

x = gradient_descent(df, np.array([5.0]), 0.1)
