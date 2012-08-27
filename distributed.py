import mincemeat
import itertools
from mackey_glass  import mackey_glass

def mapfn(series_id, values):
    import shelve
    import liquid
    import numpy
    d = shelve.open("esn.shlv")
    machine = d["mackeyglass"]
    d.close()
    xs,ys=values

    A = None 
    b = None 
    for xi,yi in zip(machine.run(xs,ys),ys):
        xi=numpy.append(xi,numpy.ones((1,1),dtype=numpy.double),axis=1)
        yi=yi[0]
        if A==None:
            assert b==None
            A=numpy.zeros((xi.size,xi.size),dtype=numpy.double)
            b=numpy.zeros(xi.size,dtype=numpy.double)

        xiT=numpy.transpose(xi)
        XTX1=numpy.dot(xiT,xi)
        A = numpy.add(XTX1,A)

        XTy1=numpy.dot(xi.ravel(),yi)
        b = numpy.add(XTy1,b)

    yield 1, (A,b)

def mapfn_sentiment(job_id, path_pol):
    import shelve
    import liquid
    import numpy
    import sentiment

    d = shelve.open("esn.shlv")
    machine = d["sentiment"]
    d.close()
    sentiment.config_vocab(1000)
    xs,ys=sentiment.make_training_example(*path_pol)

    A = None 
    b = None 
    for xi,yi in zip(machine.run(xs,ys),ys):
        xi=numpy.append(xi,numpy.ones((1,1),dtype=numpy.double),axis=1)
        yi=yi[0]
        if A==None:
            assert b==None
            A=numpy.zeros((xi.size,xi.size),dtype=numpy.double)
            b=numpy.zeros(xi.size,dtype=numpy.double)

        xiT=numpy.transpose(xi)
        XTX1=numpy.dot(xiT,xi)
        A = numpy.add(XTX1,A)

        XTy1=numpy.dot(xi.ravel(),yi)
        b = numpy.add(XTy1,b)

    yield 1, (A,b)
    
def reducefn(k, vs):
    import numpy.linalg as linalg
    import numpy

    A = None
    b = None

    for a1,b1 in vs:
        if A == None:
            assert b == None
            A=a1
            b=b1
        else:
            A=numpy.add(a1,A)
            b=numpy.add(b1,b)
    return numpy.dot(linalg.pinv(A),b)

### TRAINING DATA
#sim_length=1000
#u_input=[[1] for t in range(sim_length)]
#u_train=[[t] for t in itertools.islice(mackey_glass(beta=2.0,gamma=1.0,tau=2.0,n=9.65,x=1.0,dt=0.001),sim_length)]

import sentiment
sentiment.config_vocab(1000)

paths = sentiment.gen_example_paths(15)

### MAPREDUCE
s=mincemeat.Server()
#s.datasource=dict(enumerate([(u_input, u_train)]))
s.datasource=dict(enumerate(paths))
s.mapfn=mapfn_sentiment
s.reducefn=reducefn
results=s.run_server(password="blubberquark")

### get output weights
w = results[1]
w_out = w.reshape(1,w.size)

### make prediction
import liquid
import shelve

d = shelve.open("esn.shlv")
#machine = d["mackeyglass"]
machine = d["sentiment"]
d.close()

#u_in_test=[[1] for t in range(sim_length)]
#u_target_test=[[t] for t in itertools.islice(mackey_glass(beta=2,gamma=1,tau=2,n=9.65,x=0.5,dt=0.001),100)]

#print liquid.square_error(machine,w_out,[(u_in_test,u_target_test)])

test_set=list(sentiment.gen_examples(35))
print "test set created"
print liquid.accuracy(machine,w_out,test_set,5,0)

### VALIDATE
#train_in,train_target = list(liquid.run_all([(u_input,u_train)],machine))
#w_out_2 = liquid.linear_regression(train_in,train_target)

#print liquid.square_error(machine,w_out_2,[(u_in_test,u_target_test)])
