from liquid import *
from esn_trainer import *
from one_two_a_x_task import *
import itertools
import shelve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import Oger
import time

def printf(format, *args):
    sys.stdout.write(format % args)  

def run_memory_task():
    delay = 20
    print "create data..."
    train_input, train_target = Oger.datasets.memtest(n_samples=1, sample_len=10000, n_delays=delay)
    test_input, test_target = Oger.datasets.memtest(n_samples=1, sample_len=1000, n_delays=delay)
    
    print "train machine..."
    machine = ESN(1,15)
    trainer = LinearRegressionTrainer(machine)
    trainer.train(train_input[0], train_target[0])
    
    print "predict..."
    prediction = trainer.predict(test_input[0])
    memory_capacity = -Oger.utils.mem_capacity(prediction, test_target[0])
    mse = Oger.utils.mse(prediction,test_target[0])
    printf("MSE: %f Memory Capacity: %f\n", mse, memory_capacity)
        
def run_NARMA_task():
    [train_input, train_target] = Oger.datasets.narma30(n_samples=1, sample_len=10000)
    [test_input, test_target] = Oger.datasets.narma30(n_samples=1, sample_len=10000)
    
    print "train machine..."
    machine = ESN(1,100)
    trainer = LinearRegressionTrainer(machine)
    trainer.train(train_input[0], train_target[0])
    
    print "predict..."
    machine.reset_state()
    prediction = trainer.predict(test_input[0])
    mse = Oger.utils.nrmse(prediction,test_target[0])
    printf("NRMSE: %f\n", mse)

def run_one_two_a_x_task():
    length = 10000
    [train_input, train_target] = one_two_ax_task(length)
    [test_input, test_target] = one_two_ax_task(length)
    
    print "train machine..."
    machine = ESN(9,100)
    trainer = LinearRegressionTrainer(machine)
    trainer.train(train_input, train_target)
    
    print "predict..."
    #machine.reset_state()
    prediction = trainer.predict(test_input)
    prediction[prediction<0.5] = 0
    prediction[prediction>0.5] = 1
    error_percentage = (1-abs(test_target - prediction).mean())*100;
    printf("Success %.2f", error_percentage)
    
def multiple_superimposed_oscillators_task():
    input_range = np.arange(3000) #np.array([range(2000)])
    timescale=10.0
    osc1 = np.sin(input_range/timescale)
    osc2 = np.sin(2.1*input_range/timescale)
    osc3 = np.sin(3.4*input_range/timescale)
    train_target = np.column_stack((osc1, osc2, osc3))
    train_input = osc1*np.cos(osc2+2.345*osc3)
    train_input = train_input[:, None]
    
    machine = ESN(1, 800, leak_rate=0.5)
    print 'Starting training...'
    start = time.time()
    trainer = LinearRegressionTrainer(machine)
    trainer.train(train_input[:2000], train_target[:2000])
    print 'Training Time: ', time.time() - start, 's'
    prediction = trainer.predict(train_input[2000:])
    mse = Oger.utils.mse(prediction,train_target[2000:])
    print 'MSE: ', mse
    
    plt.subplot(3,1,1)
    plt.plot(train_input[2800:3000])
    plt.title('Input')
    plt.subplot(3,1,2)
    plt.plot(train_target[2800:3000])
    plt.title('Targets')
    plt.subplot(3,1,3)
    plt.plot(prediction[800:1000])
    plt.title('Predictions')
    plt.show()
     
if  raw_input("multiple superimposed oscillators task?[ja/nein] ").startswith('j'): 
    multiple_superimposed_oscillators_task()
elif raw_input("memory task?[ja/nein] ").startswith('j'): 
    run_memory_task()
elif raw_input("1_2_A_X task?[ja/nein] ").startswith('j'): 
    run_one_two_a_x_task()
elif raw_input("NARMA 30[ja/nein] ").startswith('j'): 
    run_NARMA_task()
else:
    print "do nothing"