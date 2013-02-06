# Plotting of ESN response

import numpy
import matplotlib
import matplotlib.pyplot as plt

d={}

def run_perturbation(esn,data1=None):
    if data1 is None:
        data1=numpy.zeros((500,esn.ninput))
    else:
        data1=data1[:500,:]
    data2=numpy.copy(data1)
    data2[90:100,:]=1
    data1[90:100,:]=0
    s1=esn.run_batch_feedback(data1)
    s2=esn.run_batch_feedback(data2)
    d["input"] = data1
    d["echo_states"] = s1
    d["difference"] = abs(s2-s1)
    d["sum_difference"] = numpy.sum(abs(s2-s1),axis=1)

def run_ESN(esn,input):
    d["input"]=input
    d["echo_states"] = esn.run_batch_feedback(input)

def set_weights(weights):
    d["weights"]=weights

def run_task(readout,input,target):
    d["output"]=readout.predict(input)
    d["target"]=target
    d["input"]=input

def show_task():
    plt.plot(d["target"][0,:1000].ravel())
    plt.plot(d["output"][0,:1000].ravel())
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
    plt.show()

def plot_input_spectrum(max_freq=1000,max_nodes=300):
    echo_states = d["input"]
    x,y = echo_states.shape
    spectrum=numpy.fft.rfft(echo_states[:max_freq,:], axis = 0)
    spectrum[0,:]=0 # we do not care about constants
    spectrum=numpy.abs(spectrum)
    plt.pcolormesh(spectrum[:max_freq,:].T)
    plt.show()

def plot_spectrum(max_freq=1000,max_nodes=300):
    echo_states = d["echo_states"]
    x,y = echo_states.shape
    spectrum=numpy.fft.rfft(echo_states[:max_freq,:], axis = 0)
    spectrum[0,:]=0 # we do not care about constants
    spectrum=numpy.abs(spectrum)
    plt.pcolormesh(spectrum[:max_freq,:].T)
    plt.show()

def plot_activations_weighted(max_duration=1000,max_nodes=300,out_unit=0):
    echo_states = numpy.copy(d["echo_states"])
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
    spectrum=numpy.fft.rfft(echo_states[:max_freq,:], axis = 0)
    spectrum[0,:]=0 # we do not care about constants
    spectrum=numpy.abs(spectrum)
    for i in range(spectrum.shape[0]):
        spectrum[i,:] = spectrum[i,:]*w_out[1:,out_unit]
    #spectrum = spectrum*w_out[1:,out_unit]
    plt.pcolormesh(spectrum[:max_freq,:].T)
    plt.show()

def plot_activation_distribution():
    echo_states = d["echo_states"]
    x,y = echo_states.shape
    

def calc_loops(m):
    x,y = m.shape
    m1=numpy.eye(x)
    diag=numpy.diag_indices_from(m)
    for i in range(x):
        m1=numpy.dot(m1,m)
        yield len(numpy.nonzero(m1[diag])[0])
    
        
