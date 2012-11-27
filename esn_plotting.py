# Plotting of ESN response

import numpy
import matplotlib
import matplotlib.pyplot as plt

d={}

def run_ESN(esn,input):
    d["echo_states"] = esn.run_batch(input)

def set_weights(weights):
    d["weights"]=weights

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

def plot_spectrum(max_freq=1000,max_nodes=300):
    echo_states = d["echo_states"]
    x,y = echo_states.shape
    spectrum=numpy.fft.rfft(echo_states, axis = 0)
    print spectrum.shape
    #plt.pcolormesh(spectrum[:max_freq,:],cmap="bone")
    plt.matshow(spectrum)
    plt.show()

def plot_ESN_Wout(Wout):
    plt.matshow(Wout)
