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
    """logistic function, applied element-wise to a matrix.
    Can be used as a sigmoidal activation function"""
    return numpy.tanh(matrix)/2+smoothstep(matrix)

def softmax(vector):
    exp = numpy.exp(vector)
    return exp/numpy.sum(exp)
