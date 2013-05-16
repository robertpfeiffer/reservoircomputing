import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def analyse_drone_results():
    pass


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
    
if __name__ == '__main__':