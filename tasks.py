from liquid import *
from esn_readout import *
from one_two_a_x_task import *
from numpy import *
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
    trainer = LinearRegressionReadout(machine)
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
    trainer = LinearRegressionReadout(machine)
    trainer.train(train_input[0], train_target[0])
    
    print "predict..."
    machine.reset()
    prediction = trainer.predict(test_input[0])
    mse = Oger.utils.nrmse(prediction,test_target[0])
    printf("NRMSE: %f\n", mse)

def run_one_two_a_x_task():
    length = 10000
    [train_input, train_target] = one_two_ax_task(length)
    [test_input, test_target] = one_two_ax_task(length)
    
    print "train machine..."
    machine = ESN(9,100)
    trainer = LinearRegressionReadout(machine)
    trainer.train(train_input, train_target)
    
    print "predict..."
    #machine.reset_state()
    prediction = trainer.predict(test_input)
    prediction[prediction<0.5] = 0
    prediction[prediction>0.5] = 1
    error_percentage = (1-abs(test_target - prediction).mean())*100;
    printf("Success %.2f", error_percentage)
    
def mso_separation_task():
    """ multiple_superimposed_oscillators separation into the components"""
    input_range = np.arange(3000) #np.array([range(2000)])
    timescale=10.0
    osc1 = np.sin(input_range/timescale)
    osc2 = np.sin(2.1*input_range/timescale)
    osc3 = np.sin(3.4*input_range/timescale)
    train_target = np.column_stack((osc1, osc2, osc3))
    train_input = osc1*np.cos(osc2+2.345*osc3)
    train_input = train_input[:, None] #1d->2d
    
    machine = ESN(1, 800, leak_rate=0.5)
    print 'Starting training...'
    start = time.time()
    trainer = LinearRegressionReadout(machine)
    trainer.train(train_input[:2000], train_target[:2000])
    print 'Training Time: ', time.time() - start, 's'
    prediction = trainer.predict(train_input[2000:])
    mse = Oger.utils.mse(prediction,train_target[2000:])
    nrmse = Oger.utils.nrmse(prediction,train_target[2000:])
    print 'MSE: ', mse, 'NRMSE:' , nrmse
    
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
    
def mso_task():
    print 'MSO Task'
    washout_time = 100
    training_time = 1000
    testing_time = 600
    evaluation_time = 300 #only last 300 steps evaluated
    
    input_range = np.arange(10000) #np.array([range(2000)])
    #data = np.sin(0.2*input_range) + np.sin(0.311*input_range)
    data = sin(0.2*input_range) + sin(0.311*input_range) + sin(0.42*input_range) 
    #+ sin(0.42*input_range) + sin(0.51*input_range) + sin(0.74*input_range)
    data = data[:, None]
    N = 500
    
    T = 1
    nrmses = np.zeros(T)
    
    for i in range(T):
        leak_rate = 0.2
        #leak_rate = random.uniform(0.2, 0.3, N)
        #leak_rate = np.append(random.uniform(0.2, 0.25, N-100), random.uniform(0.8, 1, 100))
        #leak_rate = np.append(0.2*np.ones(480), 1*np.ones(20))
        machine = ESN(1, N, leak_rate=leak_rate, bias_scaling=0.5, reset_state=False, start_in_equilibrium=False)
        machine.run_batch(data[:washout_time])
        print 'Training...'
        trainer = FeedbackReadout(machine, LinearRegressionReadout(machine, ridge=1e-8));
        trainer.train(data[washout_time:washout_time+training_time])
        
        prediction = trainer.generate(testing_time)
        testData = data[washout_time+training_time:washout_time+training_time+testing_time]
        
        evaluation_data = data[washout_time+training_time+(testing_time-evaluation_time):washout_time+training_time+testing_time]
        evaluaton_prediction = prediction[-evaluation_time:]
        mse = Oger.utils.mse(evaluaton_prediction,evaluation_data)
        nrmse = Oger.utils.nrmse(evaluaton_prediction,evaluation_data)
        nrmses[i] = nrmse
        print 'TEST MSE: ', mse, 'NRMSE: ', nrmse
    
    mean_nrmse = mean(nrmses)
    std_nrmse = std(nrmses)
    min_nrmse = min(nrmses)
    print 'Min NRMSE: ', min_nrmse, 'Mean NRMSE: ', mean_nrmse, 'Std: ', std_nrmse
    
    plt.figure(1).clear()
    plt.plot( evaluation_data, 'g' )
    plt.plot( evaluaton_prediction, 'b' )
    plt.title('Test Performance')
    plt.legend(['Target signal', 'Free-running predicted signal'])
    plt.show()
    

def mackey_glass_task():
    #from http://minds.jacobs-university.de/mantas/code
    print 'Mackey-Glass t17 - Task'
    data = np.loadtxt('MackeyGlass_t17.txt') 
    data = data[:,None]
    trainLen = 2000
    testLen = 500
    N = 1000

    print 'Create ESN...'
    random.seed(42)
    machine = ESN(1, N, leak_rate=0.3, input_scaling=0.5, bias_scaling=0.5, spectral_radius_scaling=1.25, reset_state=False, start_in_equilibrium=False)
    trainer = FeedbackReadout(machine, LinearRegressionReadout(machine, ridge=1e-8));
    print 'Training...'
    start = time.time()
    trainer.train(data[:trainLen])
    print 'Training Time: ', time.time() - start, 's'
    
    #machine.reset()
    #trainer.initial_input = data[0,None]
    prediction = trainer.generate(testLen)
    testData = data[trainLen:trainLen+testLen]
    mse = Oger.utils.mse(prediction,testData)
    nrmse = Oger.utils.nrmse(prediction,testData)
    print 'TEST MSE: ', mse, 'NRMSE: ', nrmse
    
    plt.figure(1).clear()
    #plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
    #plt.plot( prediction, 'b' )
    plt.plot( testData, 'g' )
    plt.plot( prediction, 'b' )
    plt.title('Test Performance')
    plt.legend(['Target signal', 'Free-running predicted signal'])
    #plt.show()
    
    plt.figure(2).clear()
    plt.bar( range(1+N), trainer.w_out)
    plt.title('Output weights $\mathbf{W}^{out}$')
    #plt.show()
    
    """ Peformance on Training Data: """
    machine.reset()
    trainer.initial_input = data[0,None]
    prediction = trainer.generate(trainLen)
    testData = data[1:trainLen+1]
    mse = Oger.utils.mse(prediction,testData)
    nrmse = Oger.utils.nrmse(prediction,testData)
    print 'Training MSE: ', mse, 'NRMSE: ', nrmse
    
    plt.figure(3).clear()
    #plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
    #plt.plot( prediction, 'b' )
    plt.plot( testData, 'g' )
    plt.plot( prediction, 'b' )
    plt.title('Training Performance')
    plt.legend(['Target signal', 'Free-running predicted signal'])
    plt.show()
    
    
    """
    prediction = trainer.generate(testLen)
    mse = Oger.utils.mse(prediction,data[trainLen:trainLen+testLen])
    nrmse = Oger.utils.nrmse(prediction,data[trainLen:trainLen+testLen])
    print 'MSE: ', mse, 'NRMSE: ', nrmse
    
    plt.figure(1).clear()
    #plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
    #plt.plot( prediction, 'b' )
    plt.plot( data[trainLen+1:trainLen+100+1], 'g' )
    plt.plot( prediction[:100], 'b' )
    plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    plt.legend(['Target signal', 'Free-running predicted signal'])
    plt.show()
    
    plt.figure(2).clear()
    plt.bar( range(1+N), trainer.w_out)
    plt.title('Output weights $\mathbf{W}^{out}$')
    plt.show()
    """

if 1:
    mso_task()  
elif raw_input("mackey glass?[ja/nein] ").startswith('j'): 
    mackey_glass_task()
elif  raw_input("multiple superimposed oscillators separation task?[ja/nein] ").startswith('j'): 
    mso_separation_task()
elif raw_input("memory task?[ja/nein] ").startswith('j'): 
    run_memory_task()
elif raw_input("1_2_A_X task?[ja/nein] ").startswith('j'): 
    run_one_two_a_x_task()
elif raw_input("NARMA 30[ja/nein] ").startswith('j'): 
    run_NARMA_task()
else:
    print "do nothing"