from reservoir import *
from esn_readout import *
from one_two_a_x_task import *
from numpy import *
from py_utils import *
import shelve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import sys
import time
import error_metrics
import csv
import io
import sys
import ast
from esn_persistence import *
import esn_plotting
from activations import *
import drone_tasks
import copy
import random
#import Oger


class ESNTask(object):
    
    def rescale_after_ip(self, machine, activ_fct):
        """ returns the new spectral radius """
        machine.w_echo = (machine.w_echo.T*activ_fct.a).T
        machine.w_input = (machine.w_input.T *activ_fct.a).T
        machine.w_add = machine.w_add*activ_fct.a + activ_fct.b
        machine.gamma = TanhActivation()
        return machine.get_spectral_radius()
    
    def run(self, data, training_time, testing_time=None, washout_time=0, evaluation_time=None, target_columns=[0], 
                        fb=False, T=10, LOG=True, **machine_params):
        #TODO: fb_columns fuer den Fall, dass das fb!=target ist
            #if fb == True:
        #    fb_columns = target_columns
        
        """ washout_time is part of training_time, evaluation_time is the final part of testing_time
        if (machine_params == None or len(machine_params)==0):                       
            machine_params = {"output_dim":100, "leak_rate":0.5, "conn_input":0.3, "conn_recurrent":0.2, 
                          "input_scaling":0.1, "bias_scaling":0.2, "spectral_radius":1.1, 'recurrent_weight_dist':0, 
                          'ridge':1e-8, 'fb_noise_var':0, 'ip_learning_rate':0, 'ip_std':0.01,
                          "reset_state":False, "start_in_equilibrium": True}
        """
        ridge = 0
        if 'ridge' in machine_params:
            ridge = machine_params['ridge']
            del machine_params['ridge']
        
        use_ip = False
        if 'ip_learning_rate' in machine_params:    
            ip_learning_rate = machine_params['ip_learning_rate']
            ip_std = machine_params['ip_std']
            del machine_params['ip_learning_rate']
            del machine_params['ip_std']
            if ip_learning_rate > 0:
                use_ip = True
                
        fb_noise_var = 0
        if 'fb_noise_var' in machine_params:
            fb_noise_var = machine_params['fb_noise_var']
            del machine_params['fb_noise_var'] 
            
            
        if len(data.shape)==1:
            data = data[:, None]
        
        
        nr_dims = data.shape[1]
        if testing_time == None:
            testing_time = data.shape[0]-training_time-washout_time
        
        if evaluation_time == None:
            evaluation_time = testing_time
        
        #Generell gibt es input_columns, target_columns und fb_columns. Im Momement gilt target_columns=fb_columns
        #Fuer washout und IP besteht der input aus input_columns + fb_columns        
        all_columns = range(nr_dims)
        input_columns = list(set(all_columns) - set(target_columns))
        if fb:
            input_dim = nr_dims #-len(target_columns) + len(fb_columns)
            pre_train_input_columns = all_columns
            fb_columns = target_columns
            #washout_input = data[:washout_time,all_columns]
            if use_ip:
                ip_pre_train_input = data[:training_time,all_columns]
        else:
            input_dim = nr_dims - len(target_columns)
            pre_train_input_columns = input_columns
            #washout_input = data[:washout_time,input_columns]
            if use_ip:
                ip_pre_train_input = data[washout_time:training_time,input_columns]
        
        washout_input = data[:washout_time,pre_train_input_columns]
        if use_ip:
            ip_pre_train_input = data[:training_time,pre_train_input_columns]
            
        train_input = data[washout_time:training_time,input_columns]
        train_target = data[washout_time:training_time,target_columns] #x, y, z
        test_input = data[training_time:training_time+testing_time,input_columns]
        #test_target = data[training_time:training_time+testing_time,target_columns] 
        self.evaluation_target = data[training_time+(testing_time-evaluation_time):training_time+testing_time,target_columns]
        
         
        nrmses = np.empty(T)
        self.best_nrmse = float('Inf')
        np.random.seed(42)
        random.seed(42)
        
        for i in range(T):
            #IP
            if use_ip:
                activ_fct = IPTanhActivation(ip_learning_rate, 0, ip_std, machine_params["output_dim"], init_learn=False)
                machine = ESN(input_dim=input_dim, gamma=activ_fct, **machine_params)
                if washout_time > 0:
                    machine.run_batch(washout_input)
                #normal_echo = machine.run_batch(train_target)
                activ_fct.learn = True
                machine.run_batch(ip_pre_train_input)
                activ_fct.learn = False
                new_spectral_radius = self.rescale_after_ip(machine, activ_fct)
                machine.reset()
            else:
                machine = ESN(input_dim=input_dim, **machine_params)
    
            if washout_time > 0:
                machine.run_batch(washout_input)
                    
            if fb:
                trainer = FeedbackReadout(machine, LinearRegressionReadout(machine, ridge))
                train_echo, train_prediction = trainer.train(train_input=train_input, train_target=train_target, noise_var=fb_noise_var)
                machine.current_feedback = train_target[-1]
                test_echo, prediction = trainer.generate(testing_time, inputs=test_input)
            else: 
                trainer = LinearRegressionReadout(machine, ridge);
                train_echo, train_prediction = trainer.train(train_input=train_input, train_target=train_target)
                test_echo, prediction = trainer.predict(test_input)
    
            evaluaton_prediction = prediction[-evaluation_time:]
            nrmse = error_metrics.nrmse(evaluaton_prediction,self.evaluation_target)
            if (nrmse < self.best_nrmse):
                self.best_evaluation_prediction = evaluaton_prediction
                self.best_nrmse = nrmse
                self.best_machine = machine
                self.best_trainer = trainer
                self.best_train_echo = train_echo
                self.best_test_echo = test_echo
            nrmses[i] = nrmse
            
            #if Plots:
            #    esn_plotting.plot_output_distribution((normal_echo,train_echo), ('Output Distribution without IP','Output Distribution with IP',) )
            
            if (LOG):
                print i,'NRMSE:', nrmse #, 'New Spectral Radius:', new_spectral_radius  
            
            #if best_nrmse < math.pow(10,-4):
             #   T = i + 1
              #  break
        
        self.mean_nrmse = mean(nrmses[:T])
        self.std_nrmse = std(nrmses[:T])
        #self.best_nrmse = min(nrmses[:T])
        #print 'Min NRMSE: ', min_nrmse, 'Mean NRMSE: ', mean_nrmse, 'Std: ', std_nrmse
        if (LOG):
            print 'Min NRMSE: ', self.best_nrmse 
        
            
        return self.best_nrmse, self.best_machine

def memory_task(N=15, delay=20):
    print "Memory Task"
    #print "create data..."
    #train_input, train_target = Oger.datasets.memtest(n_samples=1, sample_len=10000, n_delays=delay)
    #test_input, test_target = Oger.datasets.memtest(n_samples=1, sample_len=1000, n_delays=delay)
    #np.savez('data/memory_task_data', train_input, train_target, test_input, test_target)
    
    train_input, train_target, test_input, test_target = load_arrays('data/memory_task_data')
    
    best_capacity = 0
    for i in range(5):
        #print "train machine..."
        machine = ESN(1,N)
        trainer = LinearRegressionReadout(machine)
        trainer.train(train_input[0], train_target[0])
        
        #print "predict..."
        echo, prediction = trainer.predict(test_input[0])
        memory_capacity = -error_metrics.mem_capacity(prediction, test_target[0])
        mse = error_metrics.mse(prediction,test_target[0])
        printf("%d Memory Capacity: %f MSE: %f\n" , i, memory_capacity, mse)
        
        if memory_capacity > best_capacity:
            best_capacity = memory_capacity
    
    print 'Highest Capacity: ', best_capacity 
    return best_capacity


        
def NARMA_task(T=3, Plots=True, LOG=True, **machine_params):
    if LOG:
        print 'NARMA task'
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {"output_dim":300, "input_scaling":0.01, "bias_scaling":0.1,
                          "reset_state":True, "start_in_equilibrium": True
                      #,'ip_learning_rate':0.0005, 'ip_std':0.1
                      }
        
    #[inputs, targets] = Oger.datasets.narma30(n_samples=10, sample_len=1100)
    #[test_input, test_target] = Oger.datasets.narma30(n_samples=1, sample_len=10000)
    #np.savez('data/NARMA30_data', inputs, targets)
    #train_input, train_target = load_arrays('data/NARMA30_data') 
    
    #[train_input, train_target] = Oger.datasets.narma30(n_samples=1, sample_len=10000)
    #[test_input, test_target] = Oger.datasets.narma30(n_samples=1, sample_len=5000)
    #np.savez('data/NARMA_task_data', train_input, train_target, test_input, test_target)
    train_input, train_target, test_input, test_target = load_arrays('data/NARMA_task_data') 
        
    
    training_time = len(train_input[0])
    testing_time = len(test_input[0])
    data = np.vstack((np.hstack((train_input[0], train_target[0])), np.hstack((test_input[0], test_target[0]))))
    
    task = ESNTask()
    nrmse, machine = task.run(data, training_time=training_time, testing_time=testing_time, 
                    target_columns=[1], T=T, LOG=LOG, **machine_params)
   
    return nrmse, machine

def one_two_a_x_task():
    length = 10000
    [train_input, train_target] = one_two_ax_task(length)
    [test_input, test_target] = one_two_ax_task(length)
    
    print "train machine..."
    machine = ESN(9,100)
    trainer = LinearRegressionReadout(machine)
    trainer.train(train_input, train_target)
    
    print "predict..."
    #machine.reset_state()
    echo, prediction = trainer.predict(test_input)
    
    
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
    mse = error_metrics.mse(prediction,train_target[2000:])
    nrmse = error_metrics.nrmse(prediction,train_target[2000:])
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
    
    return nrmse
    

def mso_task(task_type=1, T=10, Plots=False, LOG=True, **machine_params):    
    
    if (machine_params == None or len(machine_params)==0):
        """
        machine_params = {"output_dim":150, "leak_rate":0.5, "conn_input":0.3, "conn_recurrent":0.1, 
                      "input_scaling":0.4, "bias_scaling":0.2, "spectral_radius":1.3, 'recurrent_weight_dist':0, 
                      'ridge':1e-8, 'fb_noise_var':0, 'ip_learning_rate':0.00005, 'ip_std':0.01,
                      "reset_state":False, "start_in_equilibrium": True}
        """
        machine_params = {"output_dim":100, "leak_rate":0.7, "conn_input":0.4, "conn_recurrent":0.1, 
                      "input_scaling":1, "bias_scaling":1, "spectral_radius":0.9, 'recurrent_weight_dist':1, 
                      'ridge':1e-8, 'fb_noise_var':0, 
                      #'ip_learning_rate':0.00005, 'ip_std':0.1,
                      "reset_state":False, "start_in_equilibrium": True}
        
    if (LOG):
        print 'MSO Task Type', task_type

    input_range = np.arange(0, 10000, 1) #np.array([range(2000)])
    if task_type==1:
        data = np.sin(0.2*input_range)
    elif task_type==2:
        data = np.sin(0.2*input_range) + np.sin(0.311*input_range) 
    elif task_type==3:
        data = np.sin(0.2*input_range) + np.sin(0.311*input_range) + np.sin(0.42*input_range)
    elif task_type==4: 
        data = np.sin(0.2*input_range) + np.sin(0.311*input_range) + np.sin(0.42*input_range) + sin(0.51*input_range)
    elif task_type==5: 
        data = np.sin(0.2*input_range) + np.sin(0.311*input_range) + np.sin(0.42*input_range) + sin(0.51*input_range) + sin(0.74*input_range)
    elif task_type==6: 
        data = np.sin(0.2*input_range) * np.sin(0.311*input_range) + np.sin(0.42*input_range) * sin(0.51*input_range) + sin(0.74*input_range)**2
    else:
        print 'Unknown MSO Task Type: ', task_type
        raise ValueError 
    
    #data = np.sin(0.2*input_range) + np.sin(0.0311*input_range)                    
    #data = np.sin(2.92*input_range) + np.sin(1.074*input_range) 
    #data = np.sin(0.2*input_range) + np.sin(0.0311*input_range) + np.sin(2.92*input_range) + np.sin(1.074*input_range) 
    #data = np.sin(0.2*input_range) + np.sin(0.311*input_range) + np.sin(0.42*input_range) + np.sin(0.74*input_range) 
    #data = sin(0.2*input_range) + sin(0.311*input_range) + sin(0.42*input_range) #+ sin(0.51*input_range) + sin(0.74*input_range)
    #data = sin(0.2*input_range)  * sin(0.311*input_range) + sin(0.42*input_range) 
    ##data = np.sin(0.0311*input_range) + np.sin(0.74*input_range)
    
    task = ESNTask()
    nrmse, machine = task.run(data, 
                    training_time=400, testing_time=600, washout_time=10, evaluation_time=300, 
                    target_columns=[0], fb=True, T=T, LOG=LOG, **machine_params)
    
    if Plots==True:
        plt.figure(1).clear()
        plt.plot( task.evaluation_target, 'g' )
        plt.plot( task.best_evaluation_prediction, 'b' )
        plt.title('Test Performance')
        plt.legend(['Target signal', 'Free-running predicted signal'])
        plt.show()
        
        
        #plt.plot(3,1,3)
        #plt.pcolormesh(plot_echo,cmap="bone")
    
        #plt.matshow(machine.w_input.T,cmap="copper")
        plt.matshow(machine.w_echo,cmap="bone")
        plt.show()
        
        #plt.matshow(best_trainer.w_out,cmap="bone")
        hist=np.histogram(task.best_trainer.w_out,bins=np.linspace(0,6,num=61))
        plt.hist(task.best_trainer.w_out)
        plt.show()
        
    return nrmse, machine
    """
        
    washout_time = 100
    training_time = 300
    testing_time = 600
    evaluation_time = 300 #only last X steps evaluated
    
    ridge=1e-8
    if 'ridge' in machine_params:
        ridge = machine_params['ridge']
        del machine_params['ridge']
    
    use_ip = False
    if 'ip_learning_rate' in machine_params:    
        ip_learning_rate = machine_params['ip_learning_rate']
        ip_std = machine_params['ip_std']
        del machine_params['ip_learning_rate']
        del machine_params['ip_std']
        if ip_learning_rate > 0:
            use_ip = True
            
    fb_noise_var = 0
    if 'fb_noise_var' in machine_params:
        fb_noise_var = machine_params['fb_noise_var']
        del machine_params['fb_noise_var']
                
    washout_time = 100
    training_time = 300
    testing_time = 600
    evaluation_time = 300 #only last X steps evaluated
    
    data = data[:, None]
    
    train_target = data[washout_time:washout_time+training_time]
            
    #save_object(data, 'data')
    #data = load_object('data');
    
    nrmses = np.empty(T)
    best_nrmse = 100000;
    
    #leak_rate = random.uniform(0.3, 1, N)
    #leak_rate = np.append(np.append(random.uniform(0.3, 1, N/3), random.uniform(0.3, 1, N/3)), random.uniform(0.3, 1, N/3))
    #leak_rate = np.append(1*np.ones(N/2), 0.7*np.ones(N/2))
        
    for i in range(T):
        #IP
        if use_ip:
            activ_fct = IPTanhActivation(ip_learning_rate, 0, ip_std,machine_params["output_dim"], init_learn=False)
            machine = ESN(input_dim=1, gamma=activ_fct, **machine_params)
            machine.run_batch(data[:washout_time])
            normal_echo = machine.run_batch(train_target)
            activ_fct.learn = True
            machine.run_batch(train_target)
            activ_fct.learn = False
            machine.reset()
        else:
            machine = ESN(input_dim=1, **machine_params)
        #machine = ESN(gamma=np.tanh, **machine_params)
        #machine = BubbleESN(1, (N/4, N/4, N/4, N/4), bubble_type=3, leak_rate=leak_rate, bias_scaling=0.5, reset_state=False, start_in_equilibrium=False)
        #machine = BubbleESN(1, (N/2, N/2), bubble_type=1, leak_rate=leak_rate, bias_scaling=0.5, reset_state=False, start_in_equilibrium=False)
        #machine = load_object('m1');
        #machine.reset()

        machine.run_batch(data[:washout_time])
                
        #print 'Training...'
        #trainer = FeedbackReadout(machine, LinearRegressionReadout(machine));
        trainer = FeedbackReadout(machine, LinearRegressionReadout(machine, ridge))
        train_echo, train_prediction = trainer.train(train_input=None, train_target=train_target, noise_var=fb_noise_var)

        machine.current_feedback = train_target[-1]
        test_echo, prediction = trainer.generate(testing_time, None)
        #testData = data[washout_time+training_time:washout_time+training_time+testing_time]
        
        evaluation_data = data[washout_time+training_time+(testing_time-evaluation_time):washout_time+training_time+testing_time]
        evaluaton_prediction = prediction[-evaluation_time:]
        #mse = error_metrics.mse(evaluaton_prediction,evaluation_data)
        nrmse = error_metrics.nrmse(evaluaton_prediction,evaluation_data)
        if (nrmse < best_nrmse):
            best_evaluation_prediction = evaluaton_prediction
            best_nrmse = nrmse
            best_machine = machine
            best_trainer = trainer
            best_train_echo = train_echo
            best_test_echo = test_echo
        nrmses[i] = nrmse
        
        #if Plots:
        #    esn_plotting.plot_output_distribution((normal_echo,train_echo), ('Output Distribution without IP','Output Distribution with IP',) )
        
        
        #plt.pcolormesh(plot_echo,cmap="bone")
        
        #print 'TEST MSE: ', mse, 'NRMSE: ', nrmse
        if (LOG):
            print i,'NRMSE:', nrmse
        
        #if best_nrmse < math.pow(10,-4):
         #   T = i + 1
          #  break
    
    mean_nrmse = mean(nrmses[:T])
    std_nrmse = std(nrmses[:T])
    min_nrmse = min(nrmses[:T])
    #print 'Min NRMSE: ', min_nrmse, 'Mean NRMSE: ', mean_nrmse, 'Std: ', std_nrmse
    if (LOG):
        print 'Min NRMSE: ', min_nrmse
    
    #save_object(best_machine, 'm2')
    #save_object(best_trainer, 't2')
    #save_object(best_train_echo, 'train_echo2')
    #save_object(best_test_echo, 'test_echo2')
    
    if Plots==True:
        plt.figure(1).clear()
        plt.plot( evaluation_data, 'g' )
        plt.plot( best_evaluation_prediction, 'b' )
        plt.title('Test Performance')
        plt.legend(['Target signal', 'Free-running predicted signal'])
        plt.show()
        
        
        #plt.plot(3,1,3)
        #plt.pcolormesh(plot_echo,cmap="bone")
    
        #plt.matshow(machine.w_input.T,cmap="copper")
        plt.matshow(best_machine.w_echo,cmap="bone")
        plt.show()
        
        #plt.matshow(best_trainer.w_out,cmap="bone")
        hist=np.histogram(best_trainer.w_out,bins=np.linspace(0,6,num=61))
        plt.hist(best_trainer.w_out)
        plt.show()
        
    return best_nrmse, best_machine
    """        

def run_task_for_grid(params_list):
    if (params_list == None or len(params_list)==0):
        params_list = [{"output_dim":100, "leak_rate":0.7, "conn_input":0.4, 
                                         "conn_recurrent":0.2, "input_scaling":1, "bias_scaling":1, 
                                         "spectral_radius":0.95, "reset_state":False, "start_in_equilibrium": True}]
    output = io.BytesIO()
    fieldnames = params_list[0].keys()
    drone_tasks.remove_unnecessary_params(fieldnames)
    fieldnames.append("NRMSE")
    writer = csv.DictWriter(output, fieldnames)
    writer.writerow(dict((fn,fn) for fn in fieldnames))
    for machine_params in params_list:
        
        #best_nrmse, best_esn = mso_task(**machine_params)
        #best_nrmse, best_esn = mackey_glass_task(**machine_params)
        best_nrmse, best_esn = NARMA_task(**machine_params)
        
        drone_tasks.remove_unnecessary_params(machine_params)
        machine_params["NRMSE"] = best_nrmse
        writer.writerow(machine_params)
    
    result = output.getvalue()
    print result
    """
    best_nrmse = mso_task(**machine_params)
    
    machine_params["NRMSE"] = best_nrmse
    str(machine_params)
    #unwichtig
    del machine_params["reset_state"]
    del machine_params["start_in_equilibrium"]
    del machine_params["Plots"]
    del machine_params["LOG"]
    
    output = io.BytesIO()
    output.getvalue()
    fieldnames = machine_params.keys()
    writer = csv.DictWriter(output, fieldnames)
    writer.writerow(dict((fn,fn) for fn in fieldnames))
    #writer.writerows(machine_params)
    writer.writerow(machine_params)
    
    result = output.getvalue()
    print result
    """

def mso_task_regression_analysis():
    print 'MSO Task Regression Analysis'
    shelf_file_name = 'data/mso_shelved.txt'
    washout_time = 100
    training_time = 1000
    testing_time = 600
    evaluation_time = 500 #only last X steps evaluated
    
    input_range = np.arange(0, 10000, 1) #np.array([range(2000)])
    data1 = np.sin(0.2*input_range) + np.sin(0.0311*input_range)
    data2 = np.sin(2.92*input_range) + np.sin(1.074*input_range)
    data3 = data1 + data2;
    data1 = data1[:, None]
    data2 = data2[:, None]
    data3 = data3[:, None];
    
    train_target1 = data1[washout_time+1:washout_time+training_time]
    train_target2 = data2[washout_time+1:washout_time+training_time]
    train_target3 = data3[washout_time+1:washout_time+training_time]
    
    evaluation_data1 = data1[washout_time+training_time+(testing_time-evaluation_time):washout_time+training_time+testing_time]
    evaluation_data2 = data2[washout_time+training_time+(testing_time-evaluation_time):washout_time+training_time+testing_time]
    evaluation_data3 = data3[washout_time+training_time+(testing_time-evaluation_time):washout_time+training_time+testing_time]
        
    train_echo1 = load_object('train_echo1', shelf_file_name)
    train_echo2 = load_object('train_echo2', shelf_file_name)
    test_echo1 = load_object('test_echo1', shelf_file_name)
    test_echo2 = load_object('test_echo2', shelf_file_name)
    
    (w_out1, train_prediction1) = lin_regression_train(train_echo1, train_target1, ridge=1e-8)
    prediction1 = lin_regression_predict(test_echo1, w_out1)
    evaluaton_prediction1 = prediction1[-evaluation_time:]
    nrmse1 = error_metrics.nrmse(evaluaton_prediction1,evaluation_data1)        
    print 'NRMSE1 sin(0.2) + sin(0.0311): ', nrmse1
    
    (w_out2, train_prediction2) = lin_regression_train(train_echo2, train_target2, ridge=1e-8)
    prediction2 = lin_regression_predict(test_echo2, w_out2)
    evaluaton_prediction2 = prediction2[-evaluation_time:]
    nrmse2 = error_metrics.nrmse(evaluaton_prediction2,evaluation_data2)        
    print 'NRMSE2: sin(2.92) + sin(1.074) ', nrmse2
    
    train_X = np.append(train_echo1, train_echo2, 1)
    test_X = np.append(test_echo1, test_echo2, 1)

    (w_out3, train_prediction3) = lin_regression_train(train_X, train_target3, ridge=1e-8)
    prediction3 = lin_regression_predict(test_X, w_out3)
    evaluaton_prediction3 = prediction3[-evaluation_time:]
    nrmse3 = error_metrics.nrmse(evaluaton_prediction3,evaluation_data3)        
    print 'NRMSE3 sin(0.2) + sin(0.0311) + sin(2.92) + sin(1.074) from regression on combined states of 1 and 2: ', nrmse3
    
    plt.figure(1).clear()
    plt.plot( evaluation_data3, 'g' )
    plt.plot( evaluaton_prediction3, 'b' )
    plt.title('Test Performance')
    plt.legend(['Target signal', 'Free-running predicted signal'])
    plt.show()
        
def mackey_glass_task(LOG=True, Plots=False, **machine_params):
    #from http://minds.jacobs-university.de/mantas/code
    if LOG:
        print 'Mackey-Glass t17 - Task'
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {"output_dim":300, "conn_input":1, "conn_recurrent":1, "leak_rate":0.3,
                          "input_scaling":0.5, "bias_scaling":0.5, "spectral_radius":1.25,
                          "reset_state":False, "start_in_equilibrium": False
                      #,'ip_learning_rate':0.0005, 'ip_std':0.1
                      }


    data = np.loadtxt('data/MackeyGlass_t17.txt') 
    task = ESNTask()
    nrmse, machine = task.run(data, training_time=2001, testing_time=500, washout_time=100, 
                    target_columns=[0], fb=True, T=10, LOG=LOG, **machine_params)
    
    
    if Plots:
        plt.figure(1).clear()
        #plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
        #plt.plot( prediction, 'b' )
        plt.plot( task.evaluation_target, 'g' )
        plt.plot( task.best_evaluation_prediction, 'b' )
        plt.title('Test Performance')
        plt.legend(['Target signal', 'Free-running predicted signal'])
        #plt.show()
        
        plt.figure(2).clear()
        N = machine_params["output_dim"]
        plt.bar( range(1+N), task.best_trainer.w_out)
        plt.title('Output weights $\mathbf{W}^{out}$')
        plt.show()
        
    return nrmse, machine
    
    """ Peformance on Training Data. Hier schlechtere Ergebnisse - wahrscheinlich wegen washout time
    machine.reset()
    machine.current_feedback = 0
    echo, prediction = trainer.generate(trainLen)
    testData = data[1:trainLen+1]
    mse = error_metrics.mse(prediction,testData)
    nrmse = error_metrics.nrmse(prediction,testData)
    print 'Training MSE: ', mse, 'NRMSE: ', nrmse
    
    if Plots:
        plt.figure(3).clear()
        #plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
        #plt.plot( prediction, 'b' )
        plt.plot( testData, 'g' )
        plt.plot( prediction, 'b' )
        plt.title('Training Performance')
        plt.legend(['Target signal', 'Free-running predicted signal'])
        plt.show()
    
    """

if __name__ == "__main__":
    if 1:
        if (len(sys.argv)==1):
            #astring = "{start_in_equilibrium: False, Plots: False, bias_scaling: 1, LOG: False, spectral_radius: 0.94999999999999996, task_type: 1, leak_rate: 0.3, output_dim: 100, input_scaling: 0.59999999999999998, reset_state: False, conn_input: 0.4, input_dim: 1, conn_recurrent: 0.2}"
            #dic = correct_dictionary_arg(astring)
            #one_two_a_x_task()
            mso_task()
            #NARMA_task()
            #mackey_glass_task()
            
            """
            machine_params_ip = {"output_dim":150, "leak_rate":0.5, "conn_input":0.3, "conn_recurrent":0.2, 
                      "input_scaling":0.4, "bias_scaling":0.2, "spectral_radius":1.3, 'recurrent_weight_dist':0, 
                      'ridge':1e-8, 'fb_noise_var':0, 'ip_learning_rate':0.001, 'ip_std':0.2,
                      "reset_state":False, "start_in_equilibrium": True}
            
            machine_params_standard = {"output_dim":100, "leak_rate":0.5, "conn_input":0.3, "conn_recurrent":0.2, 
                      "input_scaling":0.1, "bias_scaling":0.2, "spectral_radius":1.1, 'recurrent_weight_dist':0, 
                      'ridge':1e-8, 'fb_noise_var':0, 'ip_learning_rate':0, 'ip_std':0.01,
                      "reset_state":False, "start_in_equilibrium": True}
            nrmse, esn = mso_task(Plots=False, **machine_params_standard)
            nrmse_ip, esn_ip = mso_task(Plots=False, **machine_params_ip)
            
            input_range = np.arange(0, 10000, 1)
            data = np.sin(0.2*input_range) + np.sin(0.311*input_range) + np.sin(0.42*input_range) + sin(0.51*input_range) + sin(0.74*input_range)
            data = data[:,None]
            
            #perturbed_state = esn2.current_state + np.random.randn()
            #perturbed_state = perturbed_state/max(perturbed_state)
            #esn2.current_state = perturbed_state
            esn_plotting.run_perturbation(esn, data)
            esn_plotting.plot_diff2()
            
            esn_plotting.run_perturbation(esn_ip)
            esn_plotting.plot_diff2()
            """      
        else:
            #"{LOG: False, start_in_equilibrium: False, Plots: False, bias_scaling: 1, spectral_radius: 1.2, task_type: 1, leak_rate: 0.3, output_dim: 100, input_scaling: 0.80000000000000004, reset_state: False, conn_input: 0.4, input_dim: 1, conn_recurrent: 0.2}"
            args = sys.argv[1]
            dic_list = correct_dictionary_arg(args)
            run_task_for_grid(dic_list)
    elif raw_input("mso-task_regression_analysis?[ja/nein] ").startswith('j'): 
        mso_task_regression_analysis()  
    elif raw_input("mso-task?[ja/nein] ").startswith('j'): 
        mso_task()
    elif raw_input("mackey glass?[ja/nein] ").startswith('j'): 
        mackey_glass_task()
    elif  raw_input("multiple superimposed oscillators separation task?[ja/nein] ").startswith('j'): 
        mso_separation_task()
    elif raw_input("memory task?[ja/nein] ").startswith('j'): 
        memory_task()
    elif raw_input("1_2_A_X task?[ja/nein] ").startswith('j'): 
        one_two_a_x_task()
    elif raw_input("NARMA 30[ja/nein] ").startswith('j'): 
        NARMA_task()
    else:
        print "do nothing"
