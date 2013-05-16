import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(data, max_freq=1000):
    """ data-columns: dimensions """
    spectrum = _compute_spectrum(data, max_freq)
    #plt.imshow(spectrum.T)
    plt.pcolormesh(spectrum[:max_freq,:].T)
    #plt.colorbar()

def plot_spectrum_weighted(echo_states, w_out, max_freq=1000,out_unit=0):
    spectrum = _compute_spectrum(echo_states, max_freq)
    #for i in range(spectrum.shape[0]):
    #    spectrum[i,:] = spectrum[i,:]*w_out[1:,out_unit]
    spectrum = spectrum*w_out[1:,out_unit]
    #todo normalisieren
    plt.pcolormesh(spectrum[:max_freq,:].T)
    
def _compute_spectrum(data, max_freq):
    spectrum=np.fft.rfft(data[:max_freq,:], axis = 0)
    spectrum = spectrum[1:,:] # we do not care about constants
    spectrum=np.abs(spectrum)
    return spectrum
    
"""
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
    plt.pcolormesh(spectrum[:max_freq,:].T)
    plt.show()
"""
    
def plot_predictions_targets(predictions, targets, labels):
    nr_plots = predictions.shape[1]
    for i in range(nr_plots):
        plt.subplot(nr_plots,1,i+1)
        plt.plot(predictions[:,i])
        plt.plot(targets[:,i])
        plt.title(labels[i])
        
    plt.show()