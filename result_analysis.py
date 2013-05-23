import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


from reservoircomputing.esn_persistence import *
from drone_tasks import *
from drone_esn import *
from flight_data import *

def analyse_drone_results():
#    directory = 'results/mensa_results_2s_500N_062/'
#    flight_data_name = 'flight_Thu_16_May_2013_16_31_32_AllData'
    
    directory = 'results/mensa_results_2s_300N_066/'
    flight_data_name = 'flight_Thu_16_May_2013_15_30_31_AllData'
    
    directory = 'results/mensa_results_schlecht/'
    flight_data_name = 'flight_Thu_23_May_2013_13_38_16_AllData'
#    
    echo = np.squeeze(load_arrays(directory+'drone_echo.npz')[0])
    flight_data = FlightData(directory+flight_data_name)
    data = flight_data.data
    
    print 'echo.shape', echo.shape, 'flight-data', data.shape
    

def heatmap():
    data, flight_data = load_new_mensa_data(k=20)
    x = data[:,3]
    z = data[:,5]
    # missing for some reason 
    #plt.hist2d(pylab.hist2d(x,z,bins=100)
    
    hist,xedges,yedges = numpy.histogram2d(x,z,bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
    plt.imshow(hist.T,extent=extent,interpolation='nearest',origin='lower')
    plt.colorbar()
    plt.title('Space exploration by the drone')
    plt.xlabel('x in meters')
    plt.ylabel('z in meters')
    plt.show()
    
def analyse_input_data():
    data, flight_data = load_new_mensa_data(k=20)
    ypr = data[:,:3]
    w4 = data[:,-4:]
    
    #plot_hist(ypr, bins=100, labels=['Yaw', 'Pitch', 'Roll'])
    #plot_hist(w4, bins=100, labels=['w1', 'w2', 'w3', 'w4'])
    
    #pitch/forward-backward-tilt
    pitch_data = ypr[:,1]
    roll_data = ypr[:,2]
    lr_tilt_data = w4[:,0]
    fb_tilt_data = w4[:,1]
    #pitch_corr = np.correlate(pitch_data, fb_tilt_data, 'same')
    #lag_max = len(pitch_data)/2
    #plt.plot(arange(-lag_max,lag_max+1), pitch_corr)
    #plt.xcorr(fb_tilt_data, pitch_data, maxlags=20)
    plt.xcorr(lr_tilt_data, roll_data, maxlags=20)
    plt.xlabel('lags')
    plt.ylabel('correlation')
    plt.show()
    
    #yaw
    #print 'Yaw,Pitch,Roll Min:', np.min(ypr,0), 'Max:', np.max(ypr, 0)

    
    #np.correlate(a, v, mode, old_behavior)
        
def analyse_grid_results():

    #data = genfromtxt('results/mso5_13_02.csv', delimiter=',', names=True)
    #data = numpy.loadtxt('results/mso5_13_02.csv', delimiter=',')
    data = np.loadtxt(open("results/mso5_13_02.csv","rb"), delimiter=",", skiprows=1)
    #0,1,2:    input_scaling,ip_learning_rate,output_dim,
    #3,4,5:    leak_rate,ip_std,conn_input,
    #6,7:    fb_noise_var,bias_scaling,
    #8,9,10: recurrent_weight_dist,spectral_radius,conn_recurrent, 
    #11: NRMSE
    #print my_data[0], my_data.shape[0]
    
    best_percentile = 1 #1%
    RWD_COL = 8 #recurr. weight dist.
    
    data_wo_ip_ind = data[:,1]==0
    data_wo_ip = data[data_wo_ip_ind,:]
    data_with_ip = data[~data_wo_ip_ind,:]
    #print len(data_wo_ip), len(data_with_ip)
    
    with_ip_nrmse_min = data_with_ip[:,-1].min()
    without_ip_nrmse_min = data_wo_ip[:,-1].min()
    #print min(data_with_ip[:,1])
    
    best_1percentile_with_ip = np.percentile(data_with_ip[:,-1], best_percentile)
    best_1percentile_wo_ip = np.percentile(data_wo_ip[:,-1], best_percentile)
    print 'best_1percentile_with_ip: ', best_1percentile_with_ip, 'best_1percentile_without_ip: ',  best_1percentile_wo_ip
    
    best_ip_data = data_with_ip[data_with_ip[:,-1] < best_1percentile_with_ip,:]
    
    plt.hist(best_ip_data[:,7])
    #print best_ip_data[:,-1].min()
    plt.show()
    
def analyse_drone_predict():
    data, flight_data = load_new_mensa_data(k=20)
    trainer = load_object('drone_esn_predict_2s')
    input_columns = exclude_columns(data.shape[1], flight_data.target_xyz_columns)
    echo, prediction = trainer.predict(data[-2000:,input_columns])
    #eplot.plot_predictions_targets(prediction[-1000:,:], data[-1000:,flight_data.target_xyz_columns], ['x', 'y', 'z'])
    eplot.plot_activations(echo[-100:,:])
    #Unterschied in Aktivierungen
    #diff = np.diff(echo, axis=0)
    #eplot.plot_activations(echo[-100:,:])
    
if __name__ == '__main__':
    #analyse_drone_predict()
#    analyse_drone_results()
    #heatmap()
    analyse_input_data()