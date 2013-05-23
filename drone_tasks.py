from reservoircomputing import *
from flight_data import *
from py_utils import *
from tasks import *
import reservoircomputing.esn_plotting_simple as eplot

import io
import sys
import os

import numpy as np
import time

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

def control_task(T=1, Plots=True, LOG=True, Save=False, **machine_params):
    if LOG:
        print 'Control Ws Generation'
        
    data, flight_data = load_new_mensa_data(k=20)
    #data, flight_data = load_flight_random_target_data(k=30)
    w_columns = flight_data.w_columns
    
    data[:,flight_data.xyz_columns] = data[:,flight_data.target_xyz_columns] - data[:,flight_data.xyz_columns]
    data = np.hstack((data[:,:flight_data.target_xyz_columns[0]], data[:,flight_data.w_columns]))
    w_columns = w_columns - 3
    
    if (machine_params == None or len(machine_params)==0):
        #Ohne feedback
#        machine_params = {'output_dim':300, 'input_scaling':0.1, 'bias_scaling':0.1, 
#                  'conn_input':0.5, 'leak_rate':0.5, 'conn_recurrent':0.2, 
#                  'ridge':1e-6, 'spectral_radius':0.9,
#                  #'ip_learning_rate':0.00005, 'ip_std':0.01, 
#                  'reset_state':False, 'start_in_equilibrium':True} 
        
        # Mit feedback:
        machine_params = {'output_dim':500, 'input_scaling':0.1, 'bias_scaling':0.3, 
                  'conn_input':0.3, 'leak_rate':0.7, 'conn_recurrent':0.2, 
                  'ridge':1e-6, 'spectral_radius':0.9,
                  'ip_learning_rate':0.00005, 'ip_std':0.01, 
                  'reset_state':False, 'start_in_equilibrium':True}
        """
        machine_params = {'output_dim':300, 'input_scaling':0.1, 'bias_scaling':0.3, 
                          'conn_input':0.3, 'leak_rate':0.7, 'conn_recurrent':0.2, 
                          'ridge':1e-6, 'spectral_radius':1,
                          'ip_learning_rate':0.00005, 'ip_std':0.01, 
                          'reset_state':False, 'start_in_equilibrium':True}
        """
          
    test_length = 6000
    train_length = data.shape[0] - test_length
        
    task = ESNTask(machine_params, fb=True, T=T, LOG=LOG)
    best_nrmse, machine = task.run(data,
                    training_time=train_length, testing_time=test_length, washout_time=50, 
                    target_columns=w_columns)

    if Plots:
        esn_plotting.plot_predictions_targets(task.best_evaluation_prediction[:2000,:3], task.evaluation_target[:2000,:3], ('w1', 'w2', 'w3'))
    
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
    return load_flight_data_in_dir(k, "flight_data/flight_random_points_with_target/")

def load_flight_data_in_dir(k, directory):
    #file_names = filter(os.path.isfile, os.listdir(directory))
    if not os.path.exists(directory):
        directory = '../'+directory
    all_file_names = os.listdir(directory)
    file_names = [ f for f in all_file_names if not f.startswith('.') ]
    data_list = list()
    for file_name in file_names:
        flight_data = FlightData(directory+file_name,k=k)
        data_list.append(flight_data.data)
    data = np.vstack(data_list)
    #data = data[:-100,:] #bei flight_random_points ist am Ende ein outlier
    return data, flight_data

def load_new_mensa_data(k):
    """ returns the concatenated data and the first flight_data """
    return load_flight_data_in_dir(k, "flight_data/mensa_random/")
    
    
def predict_xyz_task(T=10, LOG=True, Plots=False, Save=False, k=20, **machine_params):
    #data, flight_data = load_flight_random_target_data(k=30)
    data, flight_data = load_new_mensa_data(k=k)
    #delta_xyz - schlechtere Ergebnisse
    #data[:,flight_data.target_xyz_columns] = data[:,flight_data.target_xyz_columns] - data[:,flight_data.xyz_columns]
    if LOG:
        print 'Predict XYZ, k =', k
    #test_length = 4779
    test_length = 6000
    nr_rows = data.shape[0]
    train_length = nr_rows - test_length
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'output_dim':200, 'input_scaling':0.1, 'conn_input':0.5, 'conn_recurrent':0.2,
                          'leak_rate':0.3, 'ridge':1e-5, 'bias_scaling':0.1, 'spectral_radius':1, 
                          #'ip_learning_rate':0.00005, 'ip_std':0.01,
                          'reset_state':False, 'start_in_equilibrium':True
                          ,#'dummy':True, 'hist':10
                          }
        
    task = ESNTask(machine_params, fb=False, T=T, LOG=LOG)
    best_nrmse, machine = task.run(data,
                    training_time=train_length, testing_time=test_length, washout_time=50, 
                    target_columns=flight_data.target_xyz_columns)

    if Plots:
        esn_plotting.plot_predictions_targets(task.best_evaluation_prediction[:1000,:], task.evaluation_target[:1000,:], ('X', 'Y', 'Z'))
        #eplot.plot_activations(task.best_evaluation_echo[-100:,:])
    if Save:
        save_object(task.best_trainer, 'drone_esn')
    if Save and LOG:
        print 'esn saved'
            
    return best_nrmse, machine 

def predict_xyz_task_sequence(T=5, LOG=True, Plots=False, **machine_params):
    k = 20
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
        print 'Predict XYZ Sequence'
    
    if (machine_params == None or len(machine_params)==0):
        machine_params = {'output_dim':200, 'input_scaling':0.2, 'conn_input':0.3, 
                          'leak_rate':0.3, 'ridge':1e-8, 
                          'ip_learning_rate':0.00001, 'ip_std':0.01,
                          'reset_state':False, 'start_in_equilibrium':True
                          }
    
    task = ESNTask(machine_params, fb=False, T=T, LOG=LOG)
    best_nrmse, machine = task.run_sequence(inputs, targets, washout_time=50)

    if Plots:
        esn_plotting.plot_predictions_targets(task.best_evaluation_prediction, task.evaluation_target, ('X', 'Y', 'Z'))
        
    return best_nrmse, machine 
"""    
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
"""    
if __name__ == '__main__':
    #heatmap()
    control_task(LOG=True, Plots=True, Save=True)
    #predict_xyz_task(T=3, LOG=True, Plots=True, Save=True)
    #control_task_wo_position(Plots=True, Save=True)
    

#    data, flight_data = load_new_mensa_data(k=20)
#    ypr = data[:,:3]
#    plot_hist(ypr, bins=100, labels=['Yaw', 'Pitch', 'Roll'])
#    
#    #yaw
#    print 'Yaw,Pitch,Roll Min:', np.min(ypr,0), 'Max:', np.max(ypr, 0)