import numpy,math,random
import numpy.linalg as linalg

def gradient_descent(esn,w_out,guess,y,gamma=0.0001, mask=1, n=1000):
    s=esn.current_state
    o=numpy.ones(1)
    h = numpy.dot(w_out.T,numpy.append(o,esn.step(s,guess)).ravel())
    c = numpy.linalg.norm(h-y)
    for i in range(n):
        J = numpy.dot(w_out[1:,:].T,esn.jacobian(guess,s))
        diff = numpy.dot(J.T,h-y).ravel()
        guess = guess- mask*gamma*diff
        h = numpy.dot(w_out.T,numpy.append(o,esn.step(s,guess)).ravel())
        c = numpy.linalg.norm(h-y)
        if numpy.allclose(h,y):
            break
    print c
    return guess

def random_search(esn,w_out,x,y,stddev=1,mask=1,n=1000):
    s=esn.current_state
    o=numpy.ones(1)
    h = numpy.dot(w_out.T,numpy.append(o,esn.step(s,x)).ravel())
    cost = numpy.linalg.norm(h-y)
    best = x
    for i in xrange(n):
        guess = x + mask*numpy.random.normal(0,stddev,x.size)
        h = numpy.dot(w_out.T,numpy.append(o,esn.step(s,guess)).ravel())
        c = numpy.linalg.norm(h-y)
        if c<cost:
            cost=c
            best=guess
    print cost
    return best

def random_search2(esn,w_out,x,y,stddev=1,mask=1,n=1000):
    s=esn.current_state
    o=numpy.ones(1)
    h = numpy.dot(w_out.T,numpy.append(o,esn.step(s,x)).ravel())
    cost = numpy.linalg.norm(h-y)
    best = x
    for i in xrange(n):
        guess = best + mask*numpy.random.normal(0,stddev,x.size)
        h = numpy.dot(w_out.T,numpy.append(o,esn.step(s,guess)).ravel())
        c = numpy.linalg.norm(h-y)
        if c<cost:
            cost=c
            best=guess
    print cost
    return best
