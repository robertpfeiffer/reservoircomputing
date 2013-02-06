from flight_data import *
from reservoir import *
from esn_readout import *
from py_utils import *


import numpy as np
import time
import error_metrics
import esn_plotting

def predict_xyz_task(Plot=False):
    #flight_data = FlightData('flight_data/a_to_b_constantYaw/flight_Sun_03_Feb_2013_12_27_26_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    #flight_data = FlightData('flight_data/HausVomNikolaus/flight_Sun_03_Feb_2013_18_22_19_AllData')
    
    #no norm
    flight_data = FlightData('flight_data/rectangle/flight_Sun_03_Feb_2013_17_36_54_AllData')
    
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_45_56_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    
    data = flight_data.data
    nr_rows = data.shape[0] - 500 #Am Schluss sind gelegentlich schlechte Daten
    
    washout_length = 100
    test_length = 1000
    train_length = nr_rows - test_length
    
    T = 10
    k = 5
    washout_data = data[:washout_length,:]
    train_input = data[washout_length:train_length,:]
    train_target = data[washout_length+k:train_length+k,9:12] #die letzten drei sind x, y, z
    test_input = data[train_length+k:nr_rows-k,:]
    test_target = data[train_length+k:nr_rows-k,9:12]
    
    best_nrmse = float('Inf')
    for i in range(T):
        machine = ESN(input_dim=12, output_dim=200, reset_state=False, start_in_equilibrium=True)
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
    
    if Plot:
        esn_plotting.plot_predictions_targets(best_prediction, test_target, ('X', 'Y', 'Z'))
        
    return best_nrmse  

def control_task(Plot=False):
    #flight_data = FlightData('flight_data/a_to_b_constantYaw/flight_Sun_03_Feb_2013_12_27_26_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    #flight_data = FlightData('flight_data/HausVomNikolaus/flight_Sun_03_Feb_2013_18_22_19_AllData')
    
    #no norm
    flight_data = FlightData('flight_data/rectangle/flight_Sun_03_Feb_2013_17_36_54_AllData')
    
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_45_56_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_58_39_AllData')
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_13_11_34_AllData')
    
    
    data = flight_data.data
    nr_rows = data.shape[0] - 500
    
    washout_length = 100
    test_length = 1000
    train_length = nr_rows - test_length
    
    T = 10
    target_columns = np.arange(5,9)
    input_columns = np.hstack((np.arange(0,5), np.arange(9,12))) #mit Position
    #all_columns = np.hstack((input_columns, target_columns)) #in der richtigen Reihenfolge
    #5:9
    washout_data = data[:washout_length,:]
    train_input = data[washout_length:train_length,input_columns]
    train_target = data[washout_length:train_length,target_columns] #die letzten drei sind x, y, z
    test_input = data[train_length:nr_rows,input_columns]
    test_target = data[train_length:nr_rows,target_columns]
    
    best_nrmse = float('Inf')
    for i in range(T):
        machine = ESN(input_dim=12, output_dim=200, reset_state=False, start_in_equilibrium=True)
        machine.run_batch(washout_data)
        
        
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
            best_nrmse = nrmse
            best_prediction = prediction
        printf("%d NRMSE: %f\n", i+1, nrmse)
        
    print 'Min NRMSE: ', best_nrmse
    
    if Plot:
        esn_plotting.plot_predictions_targets(best_prediction, test_target, ('w1', 'w2', 'w3', 'w4'))
        
    return best_nrmse  


if __name__ == '__main__':
    control_task(Plot=True)
    #predict_xyz_task(Plot=True)

