import numpy, math

def logistic(matrix):
    """logistic function, applied element-wise to a matrix.
    Can be used as a sigmoidal activation function"""
    return numpy.reciprocal(1+numpy.exp(numpy.negative(matrix)))

def smoothstep(matrix):
    """Can be used as a sigmoidal activation function"""
    clamped = numpy.clip(matrix,0.0,1.0)
    p1 = 3 * numpy.square(clamped)
    p2 = 2 * numpy.power(clamped,3*numpy.ones(clamped.shape))
    return p1-p2

def asymmetric_sigmoid(matrix):
    """this function is 0 at 0 but grows more quickly on the positive side"""
    return numpy.tanh(matrix)/2+smoothstep(matrix)

def softmax(vector):
    exp = numpy.exp(vector)
    return exp/numpy.sum(exp)

def fermi(x,a,b):
    return numpy.reciprocal(1+numpy.exp(-a*x-b))

def ip_exp_logistic(learnrate,mean,n):
    """Construct an activation function that adapts
    to produce an exponential activation distribution.

    Online Reservoir Adaptation by Intrinsic
    Plasticity for Backpropagation-Decorrelation
    and Echo State Learning
    Jochen J. Steil """

    def inner(x):
        y = fermi(x, inner.a, inner.b)
        if inner.learn:
            delta_b = learnrate * (1 - (2+1/mean)*y + 1/mean*y*y)
            delta_a = learnrate * 1/inner.a + x*delta_b
            inner.a += delta_a
            inner.b += delta_b
        return y

    inner.a=numpy.ones(n)
    inner.b=numpy.ones(n)*0.1
    inner.learn=True

    return inner

def ip_gaussian_tanh(learnrate,mean,stddev,n):
    """Construct an activation function that adapts
    to produce a gaussian activation distribution.

    On-line learning rule adapted from
    Adapting reservoirs to get Gaussian distributions
    David Verstraeten, Benjamin Schrauwen, Dirk Stroobandt
    ESANN 2007 proceedings"""

    def inner(x):
        y = numpy.tanh(inner.a*x + inner.b)
        if inner.learn:
            delta_b = -learnrate * (- mean/stddev + y/stddev *(2*stddev+1-y**2+mean*y))
            delta_a = learnrate/inner.a + x*delta_b
            inner.a += delta_a
            inner.b += delta_b
        return y

    inner.a=numpy.ones(n)
    inner.b=numpy.ones(n)*0.1
    inner.learn=True

    return inner
