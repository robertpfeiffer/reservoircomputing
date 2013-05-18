import numpy
import esn_readout as r
import reservoir
import scipy.signal
import matplotlib.pyplot
import error_metrics
import random

Plot=False

data=numpy.loadtxt('../data/MackeyGlass_t17.txt')

window=50
w1=scipy.signal.gaussian(window,5)
w2=scipy.signal.morlet(window,s=0.25)
w3=scipy.signal.gaussian(window,5)*numpy.array([x-(window/2) for x in range(window)])

w1=w1/numpy.linalg.norm(w1)
w2=w2/numpy.linalg.norm(w2)
w3=w3/numpy.linalg.norm(w3)

data1=numpy.convolve(data,w1,'same')
data2=numpy.convolve(data,w2,'same')
data3=numpy.convolve(data,w3,'same')

random.seed(42)
numpy.random.seed(42)
causal = False

if causal:
    data1=numpy.convolve(data,w1,'full')
    data2=numpy.convolve(data,w2,'full')
    data3=numpy.convolve(data,w3,'full')
    
    data1 = data1[:-(window-1)]
    data2 = data2[:-(window-1)]
    data3 = data3[:-(window-1)]
    
offset=30
bubble=50
split=1000

ridge = 1e-2

reload(reservoir)
net1=reservoir.ESN(2,bubble)
trainer = r.FeedbackReadout(net1, r.LinearRegressionReadout(net1, ridge))
_ = trainer.train(data[split:-offset,None],data1[split+offset:,None])
n1=net1.fold_in_feedback()
_,d1 = trainer.predict(data[:split,None])

net2=reservoir.ESN(2,bubble)
trainer = r.FeedbackReadout(net2, r.LinearRegressionReadout(net2, ridge))
_ = trainer.train(data[split:-offset,None],data2[split+offset:,None])
n2=net2.fold_in_feedback()
_,d2 = trainer.predict(data[:split,None])

net3=reservoir.ESN(2,bubble)
trainer = r.FeedbackReadout(net3, r.LinearRegressionReadout(net3, ridge))
_ = trainer.train(data[split:-offset,None],data3[split+offset:,None])
n3=net3.fold_in_feedback()
_,d3 = trainer.predict(data[:split,None])

net4=reservoir.ESN(1,bubble)

n12=reservoir.glue_esns_bias(n1,n2)
n123=reservoir.glue_esns_bias(n12,n3)
n1234=reservoir.glue_esns_bias(n123,net4,True)
trainer = r.LinearRegressionReadout(n1234, ridge)
_   = trainer.train(data[split:-offset,None],data[split+offset:,None])
_,x = trainer.predict(data[:split,None])

n5 = reservoir.ESN(2,bubble*4)
trainer = r.FeedbackReadout(n5, r.LinearRegressionReadout(n5, ridge))
_   = trainer.train(data[split:-offset,None],data[split+offset:,None])
_,x1 = trainer.predict(data[:split,None])

print 'Standard NRMSE:', error_metrics.nrmse(x1,data[offset:split+offset])
print 'Glued NRMSE:', error_metrics.nrmse(x,data[offset:split+offset])

if Plot:
    matplotlib.pyplot.plot(x.ravel())
    matplotlib.pyplot.plot(x1.ravel())
    matplotlib.pyplot.plot(data[offset:split+offset].ravel())
    matplotlib.pyplot.show()
    
    matplotlib.pyplot.plot(numpy.zeros(0))
    matplotlib.pyplot.plot(w1)
    matplotlib.pyplot.plot(w2)
    matplotlib.pyplot.plot(w3)
    matplotlib.pyplot.show()
    
    matplotlib.pyplot.plot(data[:1000].ravel())
    matplotlib.pyplot.plot(data1[:1000].ravel()+3)
    matplotlib.pyplot.plot(data2[:1000].ravel()+6)
    matplotlib.pyplot.plot(data3[:1000].ravel()+9)
    #matplotlib.pyplot.plot(numpy.zeros (1000) +9,color= "black")
    #matplotlib.pyplot.plot(numpy.zeros (1000) +6,color= "black")
    #matplotlib.pyplot.plot(numpy.zeros (1000) +3,color= "black")
    #matplotlib.pyplot.plot(numpy.zeros (1000) +0,color= "black")
    matplotlib.pyplot.show()
    
    matplotlib.pyplot.matshow(n1234.w_echo, cmap="bone")
    matplotlib.pyplot.matshow(n5.w_echo, cmap="bone")
    matplotlib.pyplot.show()
    
    matplotlib.pyplot.plot(data1[1000+offset:2000].ravel()+3)
    matplotlib.pyplot.plot(data2[1000+offset:2000].ravel()+6)
    matplotlib.pyplot.plot(data3[1000+offset:2000].ravel()+9)
    matplotlib.pyplot.plot(d1+3)
    matplotlib.pyplot.plot(d2+6)
    matplotlib.pyplot.plot(d3+9)
    matplotlib.pyplot.show()
