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
from activations import *

import matplotlib
import matplotlib.pyplot as plt


def predict_xyz_task(Plots=False, **machine_params):
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
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_37_34_AllData')
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData')
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Fri_08_Feb_2013_16_06_09_AllData')
    
    
    flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_07_34_AllData')
    flight_data2 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_23_03_AllData')
    flight_data3 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_37_34_AllData')
    flight_data4 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_54_52_AllData')
    flight_data5 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData')
    flight_data6 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_18_08_AllData')
    #flight_data7 = FlightData('flight_data/flight_random_points_with_target/flight_Fri_08_Feb_2013_16_06_09_AllData')
    data = np.vstack((flight_data.data, flight_data2.data, flight_data3.data, flight_data4.data, 
                      flight_data5.data, flight_data6.data ))#, flight_data7.data))
    
    #Fehler beim Laden
    #flight_data = FlightData('flight_data/systematicwwwwchange/flight_Sun_03_Feb_2013_18_35_58_AllData')

    #flight_data = FlightData('flight_data/random_targetPoints_changingYaw/flight_Sun_03_Feb_2013_16_55_59_AllData')
    
    #Drone Data
    #flight_data = FlightData('flight_data/esn_flight/flight_Tue_12_Feb_2013_11_53_03_3s_AllData')
    #data = flight_data.data
    
    all_dims = data.shape[1]-3
    nr_rows = data.shape[0]
    
    washout_length = 100
    test_length = 1000
    train_length = nr_rows - test_length
    
    T = 5
    k = 30
    washout_data = data[:washout_length,flight_data.w_non_target_columns]
    train_input = data[washout_length:train_length,flight_data.w_non_target_columns]
    train_target = data[washout_length+k:train_length+k,flight_data.xyz_columns] #x, y, z
    test_input = data[train_length:nr_rows-k,flight_data.w_non_target_columns]
    test_target = data[train_length+k:nr_rows,flight_data.xyz_columns]
    
    #delta:
    #train_target = train_target - train_input[:,flight_data.xyz_columns]
    #test_target = test_target - test_input[:,flight_data.xyz_columns]
    
    #plt.plot(train_input[:,0], '.')
    #plt.show()
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'input_dim': all_dims, 'output_dim':150, 'input_scaling':0.2, 'conn_input':0.3, 
                          'leak_rate':0.3, 'reset_state':False, 'start_in_equilibrium':True,
                          'ridge':1e-8 }
    use_ip = True
        
    ridge=1e-8
    if 'ridge' in machine_params:
        ridge = machine_params['ridge']
        del machine_params['ridge']
            
    best_nrmse = float('Inf')
    for i in range(T):
        if use_ip:
            activ_fct = IPTanhActivation(0.00001, 0, 0.01,machine_params["output_dim"])
            activ_fct.learn = False
            machine = ESN(gamma=activ_fct, **machine_params)
            machine.run_batch(washout_data)
            activ_fct.learn = True
            machine.run_batch(train_input)
            activ_fct.learn = False
            machine.reset()
        else:
            machine = ESN(**machine_params)
        
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
    #start = time.time()
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
    
    flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_07_34_AllData')
    flight_data2 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_23_03_AllData')
    flight_data3 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_37_34_AllData')
    flight_data4 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_54_52_AllData')
    flight_data5 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData')
    flight_data6 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_18_08_AllData')
    #flight_data7 = FlightData('flight_data/flight_random_points_with_target/flight_Fri_08_Feb_2013_16_06_09_AllData')
    data = np.vstack((flight_data.data, flight_data2.data, flight_data3.data, flight_data4.data, 
                      flight_data5.data, flight_data6.data ))#, flight_data7.data))
    
    #flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Fri_08_Feb_2013_16_06_09_AllData')

    #flight_data = FlightData('flight_data/random_targetPoints_changingYaw/flight_Sun_03_Feb_2013_16_55_59_AllData')
    #flight_data = FlightData('flight_data/random_targetPoints_changingYaw/flight_Sun_03_Feb_2013_17_22_37_AllData')
    
    #data = flight_data.data
    
            
    all_dims = data.shape[1]
    nr_rows = data.shape[0]
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'input_dim': all_dims, 'output_dim':150, 'input_scaling':0.1, 'bias_scaling':0.2, 
                          'conn_input':0.4, 'leak_rate':0.3, 'conn_recurrent':0.2, 'reset_state':False, 'start_in_equilibrium':True,
                          'ridge':1e-8, 'ip_learning_rate':0.00005, 'ip_std':0.01 }
        
    test_length = 1000
    train_length = nr_rows - test_length
    
    #Normalisierung - macht alles noch schlimmer!
    #normalizer = DataNormalizer()
    #data = normalizer.init_with_data(data)
    train_data = data[:train_length,:]
    test_data = data[train_length:nr_rows,:]
    #train_data = normalizer.init_with_data(data[:train_length,:])
    #test_data = normalizer.normalize(data[train_length:nr_rows,:])
    
    T = 20
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
        
    best_nrmse = float('Inf')
    for i in range(T):
        if use_ip:
            activ_fct = IPTanhActivation(ip_learning_rate, 0, ip_std,machine_params["output_dim"])
            activ_fct.learn = False
            machine = ESN(gamma=activ_fct, **machine_params)
            activ_fct.learn = True
            machine.run_batch(train_data)
            activ_fct.learn = False
            machine.reset()
        else:
            machine = ESN(**machine_params)
        
        trainer = FeedbackReadout(machine, LinearRegressionReadout(machine, ridge))
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
    if LOG:
        print 'esn saved'
    #print 'Time: ', time.time() - start, 's' 
        
    return best_nrmse  

def remove_unnecessary_params(list_or_dic):
    unnecessary_params = ['LOG', 'Plots', 'reset_state', 'start_in_equilibrium']
    for param in unnecessary_params:
        if param in list_or_dic:
            if isinstance(list_or_dic, list):
                list_or_dic.remove(param)
            else:
                del list_or_dic[param]
    
def control_task_for_grid(params_list):
    #TODO: kein copy&paste, sondern grid_runner herausfaktorieren 
    if (params_list == None or len(params_list)==0):
        params_list = [{"input_dim":15, "output_dim":100, "leak_rate":0.7, "conn_input":0.4, 
                                         "conn_recurrent":0.2, "input_scaling":1, "bias_scaling":1, 
                                         "spectral_radius":0.95, "reset_state":False, "start_in_equilibrium": True}]
    output = io.BytesIO()
    fieldnames = params_list[0].keys()
    remove_unnecessary_params(fieldnames)
    fieldnames.append("NRMSE")
    writer = csv.DictWriter(output, fieldnames)
    writer.writerow(dict((fn,fn) for fn in fieldnames))
    for machine_params in params_list:
        best_nrmse = control_task(**machine_params)
        remove_unnecessary_params(machine_params)
        machine_params["NRMSE"] = best_nrmse
        writer.writerow(machine_params)
    
    result = output.getvalue()
    print result
    
if __name__ == '__main__':
    #control_task_for_grid()
    #predict_xyz_task(Plots=True)
    if (len(sys.argv)==1):
        control_task(Plots=True)
        #predict_xyz_task(Plots=True)
    else:
        args = sys.argv[1]
        dic_list = correct_dictionary_arg(args)
        control_task_for_grid(dic_list)
