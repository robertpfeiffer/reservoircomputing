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

    size,dim=data.shape
    diff = 1
    split=600

    unshifted = data[0:size-diff,:]
    shifted = data[diff:size,:]
    changes = shifted - unshifted

    train_input=unshifted[:split,:]
    train_target=changes[:split,:]
    print train_input.shape
    print train_target.shape

    test_input=unshifted[split:,:]
    test_target=changes[split:,:]
    print test_input.shape
    print test_target.shape

    test_input=unshifted
    test_target=changes
    print test_input.shape
    print test_target.shape

    print "train machine..."
    #machine = ESN(20,400,leak_rate=1.0,frac_exc=0.5,gamma=np.tanh)
    #trainer = FeedbackReadout(machine,LinearRegressionReadout,ridge=1)
    machine = ESN(20,400,leak_rate=0.8,frac_exc=0.7,gamma=logistic)
    trainer = LinearRegressionReadout(machine,ridge=1)

    trainer.train(train_input, train_target)
    
    print "predict..."
    # HACK
    prediction = trainer.predict(test_input)
    for dim in range(10):
        plt.plot(test_target[:,dim].ravel())
        plt.plot(prediction[:,dim].ravel())
        plt.show()
    mse = Oger.utils.mse(prediction,test_target)
    print "MSE:", mse
        
run_drone_task()
