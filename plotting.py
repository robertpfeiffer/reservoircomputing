import itertools
import shelve
import numpy
import matplotlib
import matplotlib.pyplot as plt

def plot_ESN_run(machine,input,target,w_out=None,n_plot=1000,n_throwaway=0,jumpstart=0):
        if n_plot == 0:
            plot_in=list(input)
            plot_target=list(target)
            n_plot=len(plot_in)
        else:
            plot_in=list(itertools.islice(input,n_plot+n_throwaway))
            plot_target=list(itertools.islice(target,n_plot+n_throwaway))

        if jumpstart > 0:
            plot_run=list(machine.predict_with_echo(plot_in,w_out,plot_target[:jumpstart]))
        else:
            plot_run=list(machine.predict_with_echo(plot_in,w_out))
        plot_echo=[echo.ravel() for val,echo in plot_run]
        plot_out=[val for val,echo in plot_run]

        plot_in=numpy.array(plot_in[n_throwaway:]).T
        plot_echo=numpy.array(plot_echo[n_throwaway:]).T
        plot_out=numpy.array(plot_out[n_throwaway:]).T
        plot_target=numpy.array(plot_target[n_throwaway:]).T

        plt.subplot(3,1,1)
	if plot_in.shape[0]>50:
         plt.pcolormesh(plot_in,cmap="bone")
        else:
         for d in range(plot_in.shape[0]):
            plt.plot(plot_in[d])

        plt.subplot(3,1,2)
        for d in range(plot_out.shape[0]):
            plt.plot(plot_target[d])
            plt.plot(plot_out[d])
        if jumpstart > 0 and jumpstart-n_throwaway > 0:
            plt.axvline(jumpstart-n_throwaway)

        plt.subplot(3,1,3)
        plt.pcolormesh(plot_echo,cmap="bone")

        plt.matshow(machine.w_input.T,cmap="copper")
        plt.matshow(machine.w_echo,cmap="bone")
        plt.show()

def plot_ESN_response(machine,input,n_throwaway=0):
        plot_in=list(input)
        n_plot=len(plot_in)
        w_out=numpy.ones((1,machine.nnodes+1))
        plot_run=list(machine.predict_with_echo(plot_in,w_out))
        plot_echo=[echo.ravel() for val,echo in plot_run]
        plot_out=[val for val,echo in plot_run]

        plot_in=numpy.array(plot_in[n_throwaway:]).T
        plot_echo=numpy.array(plot_echo[n_throwaway:]).T
        plot_out=numpy.array(plot_out[n_throwaway:]).T

        plt.subplot(3,1,1)
	if plot_in.shape[0]>50:
         plt.pcolormesh(plot_in,cmap="bone")
        else:
         for d in range(plot_in.shape[0]):
            plt.plot(plot_in[d])

        plt.subplot(3,1,2)
        for d in range(plot_out.shape[0]):
            plt.plot(plot_out[d])

        plt.subplot(3,1,3)
        plt.pcolormesh(plot_echo,cmap="bone")

        plt.matshow(machine.w_echo,cmap="bone")
        plt.show()
