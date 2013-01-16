import reservoir
import shelve
import esn_plotting
import esn_readout
import Tkinter
import numpy as np


class ESN_Generator(Tkinter.Frame):
    def generate_ESN(self):
        i=int(self.input_nodes.get())
        e=int(self.echo_nodes.get())
        l=float(self.leak_rate.get())

        self.reservoir=reservoir.ESN(i,e,leak_rate=l)
        print self.reservoir

    def setup_task(self):
        task_name=self.task_listbox.curselection()[0]
        if task_name=="0":
            print "mso separation"
            self.input_nodes.delete(0,"end")
            self.output_nodes.delete(0,"end")
            self.input_nodes.insert(0,1)
            self.output_nodes.insert(0,3)

            input_range = np.arange(30000)
            timescale=10.0
            osc1 = np.sin(input_range/timescale)
            osc2 = np.sin(2.1*input_range/timescale)
            osc3 = np.sin(3.4*input_range/timescale)
            train_target = np.column_stack((osc1, osc2, osc3))
            train_input = osc1*np.cos(osc2+2.345*osc3)
            train_input = train_input[:, None] #1d->2d
            self.task={}
            self.task["train_in"]=train_input
            self.task["train_out"]=train_target

            input_range = np.arange(3000)
            osc1 = np.sin(1+input_range/timescale)
            osc2 = np.sin(2+2.1*input_range/timescale)
            osc3 = np.sin(3+3.4*input_range/timescale)
            test_target = np.column_stack((osc1, osc2, osc3))
            test_input = osc1*np.cos(osc2+2.345*osc3)
            test_input = test_input[:, None] #1d->2d
            self.task["test_in"] =test_input
            self.task["test_out"]=test_target

    def run_ESN(self):
        esn_plotting.run_ESN(self.reservoir,self.task["train_in"])
        esn_plotting.run_perturbation(self.reservoir,self.task["train_in"])

    def train_ESN(self):
        self.readout=esn_readout.LinearRegressionReadout(self.reservoir,ridge=1)
        self.readout.train(self.task["train_in"],self.task["train_out"])
        esn_plotting.set_weights(self.readout.w_out)

    def plot_spectrum(self):
        esn_plotting.plot_spectrum_weighted()

    def plot_spectrum(self):
        esn_plotting.plot_spectrum()

    def run_task_ESN(self):
        esn_plotting.run_task(self.readout,self.task["test_in"],self.task["test_out"])

    def show_task(self):
        esn_plotting.show_task()
        
    def createWidgets(self):
	l=Tkinter.Label(self,text="Task")
        l.pack()        
	self.task_listbox = Tkinter.Listbox(self)
        self.task_listbox.pack()

        for item in ["MSO Separation"]:
            self.task_listbox.insert(Tkinter.END, item)

        self.task_picker = Tkinter.Button(self)
        self.task_picker["text"] = "pick task"
        self.task_picker["command"] =  self.setup_task
        self.task_picker.pack()
        
        l=Tkinter.Label(self,text="Input Nodes")
        l.pack()
        self.input_nodes = Tkinter.Spinbox(self,from_=1,to=100)
        self.input_nodes.pack()

	l=Tkinter.Label(self,text="Echo Nodes")
        l.pack()
        self.echo_nodes = Tkinter.Spinbox(self,from_=10,to=1000)
        self.echo_nodes.pack()

        l=Tkinter.Label(self,text="Leak Rate")
        l.pack()
        self.leak_rate = Tkinter.Entry(self)
        self.leak_rate.pack()

        l=Tkinter.Label(self,text="Readout Nodes")
        l.pack()
        self.output_nodes = Tkinter.Spinbox(self,from_=10,to=1000)
        self.output_nodes.pack()

        self.generate = Tkinter.Button(self)
        self.generate["text"] = "generate ESN"
        self.generate["command"] =  self.generate_ESN
        self.generate.pack()

        self.run = Tkinter.Button(self)
        self.run["text"] = "run ESN"
        self.run["command"] =  self.run_ESN
        self.run.pack()

        self.train = Tkinter.Button(self)
        self.train["text"] = "train ESN"
        self.train["command"] =  self.train_ESN
        self.train.pack()
        
        self.run_task = Tkinter.Button(self)
        self.run_task["text"] = "run ESN on task"
        self.run_task["command"] =  self.run_task_ESN
        self.run_task.pack()

        self.readout_spectrum = Tkinter.Button(self)
        self.readout_spectrum["text"] = "show spectrum of readout"
        self.readout_spectrum["command"] =  self.plot_spectrum
        self.readout_spectrum.pack()

        self.performance = Tkinter.Button(self)
        self.performance["text"] = "compare performance"
        self.performance["command"] =  self.show_task
        self.performance.pack()


        
        
    def __init__(self, master=None):
        Tkinter.Frame.__init__(self, master)
        self.task=None
        self.reservoir=None
        self.readout=None
        self.pack()
        self.createWidgets()

root = Tkinter.Tk()
app = ESN_Generator(master=root)
app.mainloop()
root.destroy()
