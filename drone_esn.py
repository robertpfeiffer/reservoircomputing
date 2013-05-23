from reservoircomputing.esn_persistence import *
from reservoircomputing.esn_plotting import *
from reservoircomputing.reservoir import *
from reservoircomputing.esn_readout import *
import reservoircomputing.esn_plotting_simple as eplot
from flight_data import *
from py_utils import *
import numpy as np
import os.path


class DroneESN(object):
    def __init__(self):
        if os.path.isfile('drone_esn'):
            self.trainer = load_object('drone_esn')
        else:
            self.trainer = load_object('../drone_esn')
        #self.trainer = load_object('drone_esn')
        self.last_time = None
        self.echos = None
        self.counter = 0
        
    def compute(self, data, LOG=True):
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)
        
        self.counter += 1    
        #yaw, pitch, roll, x, y, z, targetX, targetY, targetZ
        #yaw, x, y, z, targetX, targetY, targetZ
        data[0] = float(data[0])/FlightData.Yaw_scale
        #data[1] = float(data[1])/FlightData.Pitch_scale
        #data[2] = float(data[2])/FlightData.Roll_scale
         
#        if float(data[0]) > 1000:
#            if self.last_time is None:
#                self.last_time = timestamp_to_date(float(data[0])-0.1)
#            data[0] = float(compute_time_diff_in_ms(self.last_time, data[0]))/100.0 #Tscale
#            try:
#                self.last_time = datetime.fromtimestamp(float(data[0]))
#            except:
#                pass
#        data[0] = 0.1
        #relativer target-vector
        #data[8:] = data[8:] - data[5:8]
        if LOG:
            print self.counter, "DATA:", str(data)
        #1d -> 2d
        if len(data.shape) == 1:
            data = data[None,:]
            
            
        #if drone control
        #if self.trainer.machine.ninput == 10 or self.trainer.machine.ninput == 6: #6 ohne fb
        if self.trainer.machine.ninput == 8 or self.trainer.machine.ninput == 4: #6 ohne fb
            nr_columns = data.shape[1]
            #delta: target - position
            data[:,nr_columns-6:nr_columns-3] = data[:,nr_columns-3:] - data[:,nr_columns-6:nr_columns-3]
            data = data[:,:nr_columns-3]
            
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

def example_drone_esn(save_echo=False, Plots=True):
    print 'DroneESN-Example'   
    flight_data = FlightData('flight_data/mensa_random/flight_Tue_07_May_2013_12_27_46_AllData', k=20)
    #flight_data = FlightData('flight_data/a_to_b_changingYaw/flight_Sun_03_Feb_2013_12_45_56_AllData',load_altitude=True, load_xyz=False)
    
    """
    row_data = np.column_stack((np.asarray(flight_data.dataTimeDiffs), np.asarray(flight_data.dataYaw),
            np.asarray(flight_data.dataPitch), np.asarray(flight_data.dataRoll),
            np.asarray(flight_data.dataX), np.asarray(flight_data.dataY), np.asarray(flight_data.dataZ),
            np.asarray(flight_data.dataTargetX), np.asarray(flight_data.dataTargetY), np.asarray(flight_data.dataTargetZ),
            np.asarray(flight_data.dataw1), np.asarray(flight_data.dataw2), np.asarray(flight_data.dataw3), np.asarray(flight_data.dataw4)))
   """
    row_data = flight_data.data
    row_data[:,0] *= FlightData.Yaw_scale
    #row_data[:,1] *= FlightData.Pitch_scale
    #row_data[:,2] *= FlightData.Roll_scale
    input_data = row_data[:,:-4]
    drone_esn = DroneESN()
    results = drone_esn.compute(input_data[0,:])
    for i in range(1,1000):
        result = drone_esn.compute(input_data[i,:])
        results = np.vstack((results, result))

    targets = row_data[:1000,-4:]
    if Plots:
        #eplot.plot_activations(drone_esn.echos[900:,:])
        eplot.plot_predictions_targets(results, targets, ('w1', 'w2', 'w3', 'w4'))
    #arrays = esn_persistence.load_arrays('drone_echo')
    
    if save_echo:
        drone_esn.save_echo()
        
        
    return results
    
if __name__ == '__main__':
    example_drone_esn()