from flight_data import *
from reservoir import *
from esn_readout import *
from py_utils import *
from data_normalizer import *
from py_utils import *

import csv
import io
import sys
import ast

import numpy as np
import time
import error_metrics
import esn_plotting
import activations

import matplotlib
import matplotlib.pyplot as plt


def predict_xyz_task(Plots=False):
    #flight_data = FlightData('flight_data/a_to_b_constantYaw/flight_Sun_03_Feb_2013_12_27_26_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    #flight_data = FlightData('flight_data/HausVomNikolaus/flight_Sun_03_Feb_2013_18_22_19_AllData')
    
    #no norm
    #flight_data = FlightData('flight_data/rectangle/flight_Sun_03_Feb_2013_17_36_54_AllData')
    
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_45_56_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_07_34_AllData')
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_23_03_AllData')
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_37_34_AllData')
    flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData')
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_37_34_AllData')
    
    
    data = flight_data.data
    all_dims = data.shape[1]
    nr_rows = data.shape[0]  - 500 #Am Schluss sind gelegentlich schlechte Daten
    
    washout_length = 100
    test_length = 1000
    train_length = nr_rows - test_length
    
    T = 10
    k = 10
    washout_data = data[:washout_length,:]
    train_input = data[washout_length:train_length,:]
    train_target = data[washout_length+k:train_length+k,flight_data.xyz_columns] #x, y, z
    test_input = data[train_length:nr_rows-k,:]
    test_target = data[train_length+k:nr_rows,flight_data.xyz_columns]
    
    #plt.plot(train_input[:,0], '.')
    #plt.show()
    
    
    best_nrmse = float('Inf')
    for i in range(T):
        machine = ESN(input_dim=all_dims, output_dim=200, leak_rate=0.1, input_scaling=1, reset_state=False, start_in_equilibrium=True)
        machine.run_batch(washout_data)
        
        trainer = LinearRegressionReadout(machine, ridge=1e-8)
        
        trainer.train(train_input, train_target)
            
        #print "predict..."
        #machine.reset()
        echo, prediction = trainer.predict(test_input)
        nrmse = error_metrics.nrmse(prediction,test_target)
        
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_prediction = prediction
        printf("%d NRMSE: %f\n", i+1, nrmse)
        
    print 'Min NRMSE: ', best_nrmse
    
    if Plots:
        esn_plotting.plot_predictions_targets(best_prediction, test_target, ('X', 'Y', 'Z'))
        
    return best_nrmse  

def control_task(Plots=True, LOG=True, **machine_params):
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'input_dim': 15, 'output_dim':150, 'input_scaling':0.2, 'conn_input':0.3, 'output_dim':100, 'leak_rate':0.7, 'reset_state':False, 
                      'start_in_equilibrium':True }
    #flight_data = FlightData('flight_data/a_to_b_constantYaw/flight_Sun_03_Feb_2013_12_27_26_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    #flight_data = FlightData('flight_data/HausVomNikolaus/flight_Sun_03_Feb_2013_18_22_19_AllData')
    
    #no norm
    #flight_data = FlightData('flight_data/rectangle/flight_Sun_03_Feb_2013_17_36_54_AllData')
    
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_45_56_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData')
    
    data = flight_data.data

    data = data[:-100,:] #die letzten 100 fuer alle faelle ignoriren
            
    all_dims = data.shape[1]
    nr_rows = data.shape[0]
    
    test_length = 1000
    train_length = nr_rows - test_length
    
    #Normalisierung - macht alles noch schlimmer!
    #normalizer = DataNormalizer()
    #data = normalizer.init_with_data(data)
    train_data = data[:train_length,:]
    test_data = data[train_length:nr_rows,:]
    #train_data = normalizer.init_with_data(data[:train_length,:])
    #test_data = normalizer.normalize(data[train_length:nr_rows,:])
    
    T = 30
    target_columns = flight_data.w_columns
    input_columns = np.arange(0,all_dims-4) #mit Position

    #washout_data = data[:washout_length,:]
    #train_input = data[washout_length:train_length,input_columns]
    #train_target = data[washout_length:train_length,target_columns] #die letzten drei sind x, y, z
    #test_input = data[train_length:nr_rows,input_columns]
    #test_target = data[train_length:nr_rows,target_columns]
    train_input = train_data[:,input_columns]
    train_target = train_data[:,target_columns]
    test_input = test_data[:,input_columns]
    test_target = test_data[:,target_columns]
    
    best_nrmse = float('Inf')
    for i in range(T):
        """
        activ_fct = activations.ip_tanh(0.00005, 0, 0.01,machine_params["output_dim"])
        activ_fct.learn = False
        machine = ESN(gamma=activ_fct, **machine_params)
        activ_fct.learn = True
        machine.run_batch(train_data)
        activ_fct.learn = False
        machine.reset()
        """
        machine = ESN(gamma=np.tanh, **machine_params)
        
        trainer = FeedbackReadout(machine, LinearRegressionReadout(machine, ridge=1e-8))
        train_echo, train_prediction = trainer.train(train_input, train_target)

        machine.current_feedback = train_target[-1]
        test_echo, prediction = trainer.generate(test_length, None, test_input)
        #testData = data[washout_time+training_time:washout_time+training_time+testing_time]
        
        #evaluation_data = data[train_length:nr_rows]
        #evaluaton_prediction = prediction[-evaluation_time:]
        nrmse = error_metrics.nrmse(test_target, prediction)
        
        
        nrmse = error_metrics.nrmse(prediction,test_target)
        
        if nrmse < best_nrmse:
            best_trainer = trainer
            best_nrmse = nrmse
            best_prediction = prediction
        if LOG:
            printf("%d NRMSE: %f\n", i+1, nrmse)
    
    if LOG:    
        print 'Min NRMSE: ', best_nrmse
    
    if Plots:
        esn_plotting.plot_predictions_targets(best_prediction, test_target, ('w1', 'w2', 'w3', 'w4'))
    
    save_object(best_trainer, 'trainer', 'drone_esn')
        
    return best_nrmse  

def control_task_for_grid(**machine_params):
    #TODO: kein copy&paste, sondern grid_runner herausfaktorieren 
    if (machine_params == None or len(machine_params)==0):
        machine_params = {"input_dim":15, "output_dim":100, "leak_rate":0.7, "conn_input":0.4, "conn_recurrent":0.2, 
                      "input_scaling":1, "bias_scaling":1, "spectral_radius":0.95, "reset_state":False, "start_in_equilibrium": True}
     
    
    if 'task_type' in machine_params: #todo script aufraumen
        del machine_params['task_type'] 
    best_nrmse = control_task(**machine_params)
    
    machine_params["NRMSE"] = best_nrmse
    str(machine_params)
    #unwichtig
    del machine_params["reset_state"]
    del machine_params["start_in_equilibrium"]
    if "Plots" in machine_params:
        del machine_params["Plots"]
    if "LOG" in machine_params:
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
    
if __name__ == '__main__':
    #control_task()
    #control_task_for_grid()
    #predict_xyz_task(Plots=True)
    if (len(sys.argv)==1):
        control_task()
    else:
        args = sys.argv[1]
        dic = correct_dictionary_arg(args)
        control_task_for_grid(**dic)
