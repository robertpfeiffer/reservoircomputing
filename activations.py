import numpy as np
import math

def logistic(matrix):
    """logistic function, applied element-wise to a matrix.
    Can be used as a sigmoidal activation function"""
    return np.reciprocal(1+np.exp(np.negative(matrix)))

def smoothstep(matrix):
    """Can be used as a sigmoidal activation function"""
    clamped = np.clip(matrix,0.0,1.0)
    p1 = 3 * np.square(clamped)
    p2 = 2 * np.power(clamped,3*np.ones(clamped.shape))
    return p1-p2

def asymmetric_sigmoid(matrix):
    """this function is 0 at 0 but grows more quickly on the positive side"""
    return np.tanh(matrix)/2+smoothstep(matrix)

def softmax(vector):
    exp = np.exp(vector)
    return exp/np.sum(exp)

def fermi(x,a,b):
    return np.reciprocal(1+np.exp(-a*x-b))

def ip_fermi(gamma,mean,n):
    def inner(x):
        y = fermi(x, inner.a, inner.b)
        if inner.learn:
            delta_b = gamma * (1 - (2+1/mean)*y + 1/mean*y*y)
            delta_a = gamma * 1/inner.a + x*delta_b
            inner.a += delta_a
            inner.b += delta_b
        return y

    inner.a=np.ones(n)
    inner.b=np.ones(n)*0.1
    inner.learn=True

    return inner

def ip_tanh(learning_rate, mean, std, n):
    def inner(x):
        y = np.tanh(inner.a*x + inner.b)
        if inner.learn:
            var = std*std
            delta_b = -learning_rate*(-mean/var + y/var*(2*var + 1 - y*y + mean*y))
            delta_a = learning_rate/inner.a + delta_b*x
            inner.a += delta_a
            inner.b += delta_b
        return y

    inner.a=np.ones(n)
    inner.b=np.zeros(n)
    inner.learn=True

    return inner
