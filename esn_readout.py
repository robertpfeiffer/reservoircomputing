from esn_persistence import *
import numpy as np
import reservoir
import scipy.linalg
import itertools
from activations import *
from itertools import chain

def add_noise(data, var):
    if var==0:
        return data
    return data + np.random.normal(0, var, data.shape)
    #return data

def lin_regression_predict (inputs, weights):
    X = np.append(np.ones((inputs.shape[0],1)), inputs, 1)
    Y = np.dot(X, weights)
    return Y;

def lin_regression_train(inputs, targets, ridge):
    X = np.append(np.ones((inputs.shape[0],1)), inputs, 1)
    if ridge==0:
        w_out = np.linalg.lstsq(inputs, targets)[0]
    else:
        X_T = X.T
        beta = np.dot( np.dot(targets.T,X), scipy.linalg.inv(np.dot(X_T,X) + ridge*np.eye(X.shape[1]) ) )
        w_out = beta.T
    Y = np.dot(X, w_out)
    return (w_out, Y) # weights and training prediction

def create_sequence_data(inputs, targets, washout_time=0):
    train_input = np.vstack((inputs))
    washed_indizes = []
    current_index = 0
    for inp in inputs:
        washed_indizes.append(range(current_index+washout_time,current_index+len(inp)))
        current_index += len(inp)
    #flatten
    washed_indizes = np.asarray(list(chain.from_iterable(washed_indizes)))
                             
    washed_targets = []
    for target in targets:
        washed_targets.append(target[washout_time:,:])
    washed_targets = np.vstack((washed_targets))
    return train_input, washed_indizes, washed_targets

class IPTrainer(object):
    def __init__(self, machine, learning_rate, std, mean=0):
        self.machine = machine
        self.ip_learning_rate = learning_rate
        self.ip_mean = mean
        self.ip_std = std
    
    def train(self, data, washout_time, training_time, pre_train_input_columns):
        if washout_time > 0:
            washout_input = data[:washout_time,pre_train_input_columns]
            self.machine.run_batch(washout_input)
        activ_fct = IPTanhActivation(self.ip_learning_rate, 0, self.ip_std, self.machine.nnodes, init_learn=True)
        self.machine.gamma = activ_fct
        ip_pre_train_input = data[washout_time:training_time,pre_train_input_columns]
        self.machine.run_batch(ip_pre_train_input)
        activ_fct.learn = False
        new_spectral_radius = self.rescale_after_ip(self.machine, activ_fct)
        self.machine.reset()
        return new_spectral_radius 
    
    def train_sequence(self, train_inputs, washout_time):
        activ_fct = IPTanhActivation(self.ip_learning_rate, 0, self.ip_std, self.machine.nnodes, init_learn=False)
        self.machine.gamma = activ_fct
        for train_input in train_inputs:
            if washout_time > 0:
                self.machine.run_batch(train_input[:washout_time,:])
            activ_fct.learn = True
            self.machine.run_batch(train_input[:washout_time,:])
            activ_fct.learn = False
        
        new_spectral_radius = self.rescale_after_ip(self.machine, activ_fct)
        self.machine.reset()
        return new_spectral_radius
       
    def rescale_after_ip(self, machine, activ_fct):
        """ returns the new spectral radius """
        machine.w_echo = (machine.w_echo.T*activ_fct.a).T
        machine.w_input = (machine.w_input.T *activ_fct.a).T
        machine.w_add = machine.w_add*activ_fct.a + activ_fct.b
        machine.gamma = TanhActivation()
        return machine.get_spectral_radius()
     
class LinearRegressionReadout(object):
    """ Trains an esn and uses it for prediction """

    def __init__(self, machine, ridge=0):
        self.machine = machine
        self.w_out = None
        self.ridge = ridge

    def regression(self, X, target):
        if self.ridge==0:
            self.w_out = np.linalg.lstsq(X, target)[0]
        else:
            X_T = X.T
            beta = np.dot( np.dot(target.T,X), scipy.linalg.inv( np.dot(X_T,X) + self.ridge*np.eye(X.shape[1]) ) )
            self.w_out = beta.T
        Y = np.dot(X, self.w_out)
        return (X[:, 1:],Y) #echos without constant
    
    def train(self, train_input, train_target):
        """ returns (echos, predictions) """
        if train_input is None:
            train_input=np.zeros((train_target.shape[0],0))
        X = self._createX(train_input)
        return self.regression(X, train_target)


    def train_sequence(self, train_inputs, train_targets, washout_time):
        """ washout_time: for each sample. returns (echos, predictions) """
        train_input, washed_indizes, washed_targets = create_sequence_data(train_inputs, train_targets, washout_time)
        X = self._createX(train_input) 
        X = X[washed_indizes,:] 
        return self.regression(X, washed_targets) 

    def predict(self, test_input,state=None):
        """ returns (echos, predictions) """
        X = self._createX(test_input,state=state)
        Y = np.dot(X, self.w_out)
        return (X[:, 1:],Y)

    def _createX(self, data, state=None):
        if len(data.shape) == 1:
            data = data[:,None] #1d->2d
        state_echo = self.machine.run_batch_feedback(data, state=state)
        X = np.append(np.ones((state_echo.shape[0],1)), state_echo, 1)
        return X
    """
    def predict_old(self, test_input):
        X = self._createX_old(test_input)
        Y = np.dot(X, self.w_out)
        return Y

    def _createX_old(self, data):
        if len(data.shape) == 1:
            data = data[:,None] #1d->2d
        state_echo = self.machine.run_batch(data)
        X = np.append(np.ones((state_echo.shape[0],1)), state_echo, 1)
        return X
    """

class FeedbackReadout(object):
    """ Trains an ESN in generative mode and uses it for time series creation """
    #TODO: Zwischen fortgesetzten Folgen und wiederholten Sequenzen unterscheiden
    def __init__(self, machine, trainer):
        self.machine = machine
        self.trainer = trainer

    def train(self, train_input, train_target, noise_var=0):
        """ train_target is taken as feedback.
        returns (states, prediction) """
        feedback = add_noise(train_target, noise_var)[:-1]; #all except the last
        self.feedback_dim=feedback.shape[1]
        if train_input is not None:
            train_input=np.hstack((train_input[1:],feedback)) #train_input[1:] so that input and feedback match
            self.input_dim=train_input.shape[1] #TODO: Muesste die Zeile vor der oberen Zeile stehen?
        else:
            train_input=feedback
            self.input_dim=0
        X, Y = self.trainer.train(train_input, train_target[1:])
        self.machine.w_feedback=self.trainer.w_out.T
        return (X,Y)

    def generate(self, length=1000, initial_feedback=None, state=None, inputs=None):
        """ returns (states, prediction) """
        if inputs==None or np.size(inputs)==0: #size returns 0 also on empty arrays/lists
            inputs=np.zeros((length,0))
        if initial_feedback is not None:
            echo1=self.machine.run_batch(initial_feedback)
            echo1=echo1[-1,:]
            return self.trainer.predict(inputs,echo1)
        else:
            #Feedback might not be equal to readouts,
            # so we need to compute them separately
            return self.trainer.predict(inputs, state)

    def predict(self, *args, **kwarks):
        return self.trainer.predict(*args, **kwarks)

    @property
    def w_out(self):
        return self.trainer.w_out

    def train_old(self, train_input):
        feedback = add_noise(train_input, 0.01)[:-1];
        self.trainer.train(feedback, train_input[1:])
        self.gen_dim = train_input.shape[1]
        self.initial_input = np.empty((1, self.gen_dim))
        self.initial_input[0] = train_input[-1]

    def generate_old(self, length):
        feedback = self.initial_input
        output = np.zeros((length, feedback.shape[1]))
        for i in range(length):
            output[i] = self.trainer.predict(feedback)
            feedback = output[i]
        return output
