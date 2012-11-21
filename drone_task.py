from liquid import *
from esn_readout import *
from activations import *
import numpy as np
import Oger
import matplotlib
import matplotlib.pyplot as plt

def run_drone_task():
    print "create data..."

    data=np.array(eval(open("drone_data").read()))
    data[:,0:3]=data[:,0:3]/10.0
    data[:,3:6]=data[:,3:6]/10.0

    diff = 5
    split=600
    size=1000

    train_input=data[0:split-diff,:]
    train_target=data[diff:split,:]
    print train_input.shape
    print train_target.shape

    test_input=data[split-diff:size-diff,:]
    test_target=data[split:size,:]
    print test_input.shape
    print test_target.shape

    test_input=data[0:size-diff,:]
    test_target=data[diff:size,:]
    print test_input.shape
    print test_target.shape

    print "train machine..."
    machine = ESN(10,400,leak_rate=0.1,frac_exc=0.8,gamma=logistic)
    #trainer = FeedbackReadout(machine,LinearRegressionReadout,ridge=1)
    trainer = LinearRegressionReadout(machine,ridge=1)

    trainer.train(train_input, train_target)
    
    print "predict..."
    # HACK
    prediction = trainer.predict(test_input)
    for dim in range(10):
        plt.plot(test_target[:,dim].ravel())
        plt.plot(prediction[:,dim].ravel())
        plt.plot(test_input[:,dim].ravel()-test_target[:,dim].ravel())
        plt.plot(test_input[:,dim].ravel()-prediction[:,dim].ravel())

        plt.show()
    mse = Oger.utils.mse(prediction,test_target)
    print "MSE:", mse
        
run_drone_task()
