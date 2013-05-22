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

def plot_hist(data, bins=10, labels=None):
    
    nr_plots = data.shape[1]
    for i in range(nr_plots):
        plt.subplot(nr_plots,1,i+1)
        plt.hist(data[:,i], bins)
        if labels is not None:
            plt.title(labels[i])
        
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
    
def plot_activations(echo):
    echo = echo.T
    plt.pcolormesh(echo,cmap="bone")
    plt.colorbar()
    plt.show()