from esn_persistence import *
from esn_plotting import *
from reservoir import *
from esn_readout import *
from flight_data import *
from py_utils import *
import numpy as np
import esn_persistence


class DroneESN(object):
    def __init__(self):
        self.trainer = load_object('trainer', 'drone_esn');
        self.last_time = None
        self.echos = None
        
    def compute(self, data):
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)
            
        #time, yaw, pitch, roll, x, y, z, targetX, targetY, targetZ
        data[1] = float(data[1])/100.0 #Yaw_scale
         
        if float(data[0]) > 1000:
            if self.last_time is None:
                self.last_time = timestamp_to_date(float(data[0])-0.1)
            data[0] = float(compute_time_diff_in_ms(self.last_time, data[0]))/100.0
            self.last_time = datetime.fromtimestamp(float(data[0]))
       
        #relativer target-vector
        #data[8:] = data[8:] - data[5:8]
        
        #1d -> 2d
        if len(data.shape) == 1:
            data = data[None,:]
            
        echo, prediction = self.trainer.predict(data)
        if self.echos is None:
            self.echos = echo
        else:
            self.echos = np.vstack((self.echos, echo))
        return prediction
        
    def reset(self):
        self.tainer.machine.reset()
        self.echos = None
        
    def save_echo(self):
        save_arrays('drone_echo', self.echos)
        
if __name__ == '__main__':
    flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_17_06_52_AllData')
    """
    row_data = np.column_stack((np.asarray(flight_data.dataTimeDiffs), np.asarray(flight_data.dataYaw),
            np.asarray(flight_data.dataPitch), np.asarray(flight_data.dataRoll),
            np.asarray(flight_data.dataX), np.asarray(flight_data.dataY), np.asarray(flight_data.dataZ),
            np.asarray(flight_data.dataTargetX), np.asarray(flight_data.dataTargetY), np.asarray(flight_data.dataTargetZ),
            np.asarray(flight_data.dataw1), np.asarray(flight_data.dataw2), np.asarray(flight_data.dataw3), np.asarray(flight_data.dataw4)))
   """
    row_data = flight_data.data
    row_data[:,1] *= 100.0
    input_data = row_data[:,:-4]
    drone_esn = DroneESN()
    results = drone_esn.compute(input_data[0,:])
    for i in range(1,1000):
        result = drone_esn.compute(input_data[i,:])
        results = np.vstack((results, result))

    targets = row_data[:1000,-4:]
    plot_predictions_targets(results, targets, ('w1', 'w2', 'w3', 'w4'))
    #arrays = esn_persistence.load_arrays('drone_echo')
    
    #drone_esn.save_echo()
