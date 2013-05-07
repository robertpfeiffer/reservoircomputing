from flight_data import *
from reservoir import *
from esn_readout import *
from py_utils import *
from data_normalizer import *
from py_utils import *

import io
import sys

import numpy as np
import time
import error_metrics
import esn_plotting
from activations import *
from tasks import *

import matplotlib
import matplotlib.pyplot as plt

def control_task_wo_position(Plots=True, LOG=True, Save=False, **machine_params):
    
    #flight_data = FlightData('flight_data/a_to_b_constantYaw/flight_Sun_03_Feb_2013_12_27_26_AllData',load_altitude=True, load_xyz=False)
    flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_45_56_AllData',load_altitude=True, load_xyz=False)
    
    #Klappt nicht
    #flight_data = FlightData('flight_data/rectangle/flight_Sun_03_Feb_2013_17_36_54_AllData',load_altitude=True, load_xyz=False)
    data = flight_data.data
    
            
    all_dims = data.shape[1]
    nr_rows = data.shape[0]
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'output_dim':150, 'input_scaling':0.1, 'bias_scaling':0.2, 
                          'conn_input':0.4, 'leak_rate':0.3, 'conn_recurrent':0.2, 'reset_state':False, 'start_in_equilibrium':True,
                          'ridge':1e-8, 'ip_learning_rate':0.00005, 'ip_std':0.001 }
        
    test_length = 500
    train_length = nr_rows - test_length
    
    task = ESNTask()
    nrmse, machine = task.run(data=data, 
            training_time=train_length, target_columns=flight_data.w_columns, fb=True, T=20, LOG=LOG, **machine_params)
    
    if Plots:
        esn_plotting.plot_predictions_targets(task.best_evaluation_prediction, task.evaluation_target, ('w1', 'w2', 'w3', 'w4'))
    
    if Save:
        save_object(task.best_trainer, 'saved_drone_esn')
        if LOG:
            print 'esn saved'
    #print 'Time: ', time.time() - start, 's' 
        
    return nrmse  

def control_task(T=3, Plots=True, LOG=True, Save=False, **machine_params):
    if LOG:
        print 'Control Ws Generation'
        
    data, flight_data = load_flight_random_target_data(k=20)
    
    nr_rows = data.shape[0]
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'output_dim':150, 'input_scaling':0.1, 'bias_scaling':0.2, 
                          'conn_input':0.4, 'leak_rate':0.3, 'conn_recurrent':0.2, 
                          'ridge':1e-8, 
                          'ip_learning_rate':0.00005, 'ip_std':0.01, 
                          'reset_state':False, 'start_in_equilibrium':True}
        
    test_length = 1000
    train_length = nr_rows - test_length
    
    task = ESNTask(machine_params, fb=True, T=T, LOG=LOG)
    best_nrmse, machine = task.run(data,
                    training_time=train_length, testing_time=test_length, washout_time=50, 
                    target_columns=flight_data.w_columns)

    if Plots:
        esn_plotting.plot_predictions_targets(task.best_evaluation_prediction, task.evaluation_target, ('w1', 'w2', 'w3', 'w4'))
    
    if Save:
        save_object(task.best_trainer, 'drone_esn')
    if Save and LOG:
        print 'esn saved'
            
    return best_nrmse, machine 

def remove_unnecessary_params(list_or_dic):
    unnecessary_params = ['LOG', 'Plots', 'reset_state', 'start_in_equilibrium']
    for param in unnecessary_params:
        if param in list_or_dic:
            if isinstance(list_or_dic, list):
                list_or_dic.remove(param)
            else:
                del list_or_dic[param]

def load_flight_random_target_data(k):
    """ returns the concatenated data and the first flight_data """
    flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_07_34_AllData',  k=k)
    flight_data2 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_23_03_AllData', k=k)
    flight_data3 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_37_34_AllData', k=k)
    flight_data4 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_54_52_AllData', k=k)
    flight_data5 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData', k=k)
    flight_data6 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_18_08_AllData', k=k)
    #flight_data7 = FlightData('flight_data/flight_random_points_with_target/flight_Fri_08_Feb_2013_16_06_09_AllData')
    data = np.vstack((flight_data.data, flight_data2.data, flight_data3.data, flight_data4.data, 
                      flight_data5.data, flight_data6.data ))#, flight_data7.data))
    data = data[:-200,:] #bei flight_random_points ist am Ende ein outlier
    return data, flight_data
    
def predict_xyz_task(T=5, LOG=True, Plots=False, **machine_params):
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
    
    data, flight_data = load_flight_random_target_data(k=30)

    #Fehler beim Laden
    #flight_data = FlightData('flight_data/systematicwwwwchange/flight_Sun_03_Feb_2013_18_35_58_AllData')

    #flight_data = FlightData('flight_data/random_targetPoints_changingYaw/flight_Sun_03_Feb_2013_16_55_59_AllData')
    
    #Drone Data
    #flight_data = FlightData('flight_data/esn_flight/flight_Tue_12_Feb_2013_11_53_03_3s_AllData')
    
    #flight_data = FlightData('flight_data/a_to_b_constantYaw/flight_Sun_03_Feb_2013_12_27_26_AllData')
    #data = flight_data.data
    if LOG:
        print 'Predict XYZ'
    nr_rows = data.shape[0]
    washout_length = 50
    test_length = 4779
    #washout_length = 100
    #test_length = 4729
    train_length = nr_rows - test_length
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'output_dim':300, 'input_scaling':0.1, 'conn_input':0.2, 'conn_recurrent':0.3,
                          'leak_rate':0.5, 'ridge':1e-8, 'bias_scaling':0.01, 'spectral_radius':0.8, 
                          #'ip_learning_rate':0.001, 'ip_std':0.01,
                          'reset_state':False, 'start_in_equilibrium':True
                          }
    
    task = ESNTask(machine_params, fb=False, T=T, LOG=LOG)
    best_nrmse, machine = task.run(data,
                    training_time=train_length, testing_time=test_length, washout_time=washout_length, 
                    target_columns=flight_data.target_xyz_columns)

    if Plots:
        esn_plotting.plot_predictions_targets(task.best_evaluation_prediction[:1000,:], task.evaluation_target[:1000,:], ('X', 'Y', 'Z'))
        
    return best_nrmse, machine 

def predict_xyz_task_sequence(T=5, LOG=True, Plots=False, **machine_params):
    k = 10
    f1FD = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_07_34_AllData',  k=k)
    f1 = f1FD.data
    f2 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_23_03_AllData', k=k).data
    f3 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_37_34_AllData', k=k).data
    f4 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_54_52_AllData', k=k).data
    f5 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData', k=k).data
    f6 = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_18_08_AllData', k=k).data
    #bei flight_random_points ist am Ende ein outlier
    f6 = f6[:-200,:]
    #flight_data7 = FlightData('flight_data/flight_random_points_with_target/flight_Fri_08_Feb_2013_16_06_09_AllData')

    inputs = [f1[:,f1FD.w_non_target_columns], f2[:,f1FD.w_non_target_columns],f3[:,f1FD.w_non_target_columns],
              f4[:,f1FD.w_non_target_columns],f5[:,f1FD.w_non_target_columns],f6[:,f1FD.w_non_target_columns]]
    
    targets = [f1[:,f1FD.target_xyz_columns],f2[:,f1FD.target_xyz_columns],f3[:,f1FD.target_xyz_columns],
               f4[:,f1FD.target_xyz_columns],f5[:,f1FD.target_xyz_columns],f6[:,f1FD.target_xyz_columns]]
    if LOG:
        print 'Predict XYZ'
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'output_dim':100, 'input_scaling':0.2, 'conn_input':0.3, 
                          'leak_rate':0.3, 'ridge':1e-8, 
                          'ip_learning_rate':0.00001, 'ip_std':0.01,
                          'reset_state':False, 'start_in_equilibrium':True
                          }
    
    task = ESNTask(machine_params, fb=False, T=T, LOG=LOG)
    best_nrmse, machine = task.run_sequence(inputs, targets, washout_time=50)

    if Plots:
        esn_plotting.plot_predictions_targets(task.best_evaluation_prediction, task.evaluation_target, ('X', 'Y', 'Z'))
        
    return best_nrmse, machine 
    
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
        control_task(Plots=True, Save=True)
        #control_task_wo_position(Plots=True, Save=True)
        #predict_xyz_task(Plots=True)
    else:
        args = sys.argv[1]
        dic_list = correct_dictionary_arg(args)
        control_task_for_grid(dic_list)
