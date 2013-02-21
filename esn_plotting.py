# Plotting of ESN response

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

d={}

def run_perturbation(esn,data1=None):
    T = 400
    if data1 is None:
        data1=np.zeros((T,esn.ninput))
    else:
        data1=data1[:T,:]
    data2=np.copy(data1)
    data2[100,:]+=0.1
    esn2 = copy.deepcopy(esn)
    s1=esn.run_batch_feedback(data1)
    s2=esn2.run_batch_feedback(data2)
    d["input"] = data1
    d["echo_states"] = s1
    d["net"]=esn
    d["difference"] = abs(s2-s1)
    d["sum_difference"] = np.sum(abs(s2-s1),axis=1)

def run_ESN(esn,input):
    d["input"]=input
    d["echo_states"] = esn.run_batch_feedback(input)

def set_weights(weights):
    d["weights"]=weights

def run_task(readout,input,target):
    d["output"]=readout.predict(input)[1]
    d["target"]=target
    d["input"]=input

def show_task():
    for dim in range(d["target"].shape[1]): 
        plt.plot(d["target"][dim,:1000].ravel())
        plt.plot(d["output"][dim,:1000].ravel())
    plt.show()

def plot_activations(max_duration=1000,max_nodes=300):
    echo_states = d["echo_states"]
    x,y = echo_states.shape
    if x > max_duration:
        echo_states = echo_states[0:max_duration,:]
    if y > max_nodes:
        echo_states = echo_states[:,0:max_nodes]
    echo_states = echo_states.T
    plt.pcolormesh(echo_states,cmap="bone")
    plt.show()

def plot_diff(max_duration=1000,max_nodes=300):
    echo_states = d["difference"]
    x,y = echo_states.shape
    if x > max_duration:
        echo_states = echo_states[0:max_duration,:]
    if y > max_nodes:
        echo_states = echo_states[:,0:max_nodes]
    echo_states = echo_states.T
    plt.pcolormesh(echo_states,cmap="bone")
    plt.show()

def plot_diff2():
    plt.plot(d["sum_difference"])
    esn=d["net"]
    if hasattr(esn,"comment"):
        comment=esn.comment
    else:
        comment=""
    plt.title(str(esn.nnodes)+" neurons "+comment)
    plt.annotate('perturbation', xy=(100, 10), xytext=(150,15), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

def plot_input_spectrum(max_freq=1000,max_nodes=300):
    echo_states = d["input"]
    x,y = echo_states.shape
    spectrum=np.fft.rfft(echo_states[:max_freq,:], axis = 0)
    spectrum[0,:]=0 # we do not care about constants
    spectrum=np.abs(spectrum)
    plt.pcolormesh(spectrum[:max_freq,:].T)
    plt.show()

def plot_spectrum(max_freq=1000,max_nodes=300):
    echo_states = d["echo_states"]
    x,y = echo_states.shape
    spectrum=np.fft.rfft(echo_states[:max_freq,:], axis = 0)
    spectrum[0,:]=0 # we do not care about constants
    spectrum=np.abs(spectrum)
    plt.pcolormesh(spectrum[:max_freq,:].T)
    plt.show()

def plot_activations_weighted(max_duration=1000,max_nodes=300,out_unit=0):
    echo_states = np.copy(d["echo_states"])
    x,y = echo_states.shape
    w_out=d["weights"]
    for i in range(echo_states.shape[0]):
        echo_states[i,:] = echo_states[i,:]*w_out[1:,out_unit]
    if x > max_duration:
        echo_states = echo_states[0:max_duration,:]
    if y > max_nodes:
        echo_states = echo_states[:,0:max_nodes]
    echo_states = echo_states.T
    plt.pcolormesh(echo_states,cmap="bone")
    plt.show()

def plot_spectrum_weighted(max_freq=1000,max_nodes=300,out_unit=0):
    echo_states = d["echo_states"]
    x,y = echo_states.shape
    w_out=d["weights"]
    spectrum=np.fft.rfft(echo_states[:max_freq,:], axis = 0)
    spectrum[0,:]=0 # we do not care about constants
    spectrum=np.abs(spectrum)
    for i in range(spectrum.shape[0]):
        spectrum[i,:] = spectrum[i,:]*w_out[1:,out_unit]
    #spectrum = spectrum*w_out[1:,out_unit]
    plt.pcolormesh(spectrum_weighted[:max_freq,:].T)
    plt.show()
    
def plot_predictions_targets(predictions, targets, labels):
    nr_plots = predictions.shape[1]
    for i in range(nr_plots):
        plt.subplot(nr_plots,1,i+1)
        plt.plot(predictions[:,i])
        plt.plot(targets[:,i])
        plt.title(labels[i])
        
    plt.show()
    
def plot_output_distribution(echos, labels):
    nr_plots = len(echos)
    for i in range(nr_plots):
        echo = echos[i]    
        N = echo.shape[1]
        hist, bin_edges = np.histogram(echo,bins=np.linspace(-1,1,num=201))
        hist = hist.astype(float)
        #y: percentage of (neurons*timesteps)
        hist = (hist*100)/(N*len(echo))
        plt.subplot(nr_plots, 1, i+1)
        plt.plot(bin_edges[1:], hist, '.')
        plt.title(labels[i])
        plt.xlabel('neuron output')

    plt.show()
