import numpy as np
import liquid
import mdp.utils

def add_noise(data, var):
    #return data + np.random.normal(0, var, data.shape)
    return data
    
class LinearRegressionReadout(object):
    """ Trains an esn and uses it for prediction """

    def __init__(self, machine, ridge=0):
        self.machine = machine
        self.w_out = None
        self.ridge = ridge

    def train(self, train_input, train_target):
        X = self._createX(train_input)
        if self.ridge==0:
            self.w_out = np.linalg.lstsq(X, train_target)[0]
        else:
            X_T = X.T
            beta = np.dot( np.dot(train_target.T,X), mdp.utils.inv(np.dot(X_T,X) + self.ridge*np.eye(X.shape[1]) ) )
            self.w_out = beta.T
            
    def predict(self, test_input):
        X = self._createX(test_input)
        Y = np.dot(X, self.w_out) 
        return Y
    
    def _createX(self, data):
        if len(data.shape) == 1:
            data = data[:,None] #1d->2d
        state_echo = self.machine.run_batch(data)
        X = np.append(np.ones((state_echo.shape[0],1)), state_echo, 1)
        return X



class FeedbackReadout(object):
    """ Trains an ESN in generative mode and uses it for time series creation """
    #TODO: Zwischen fortgesetzten Folgen und wiederholten Sequenzen unterscheiden
    
    def __init__(self, machine, trainer):
        self.machine = machine
        self.trainer = trainer
        #self.w_out = None
        
    def train(self, train_input):
        self.gen_dim = train_input.shape[1]
        feedback = add_noise(train_input, 0.0001)[:-1];
        self.trainer.train(feedback, train_input[1:])
        self.initial_input = np.empty((1, self.gen_dim))
        self.initial_input[0] = train_input[-1]

    def generate(self, length):
        feedback = self.initial_input
        output = np.zeros((length, feedback.shape[1]))
        for i in range(length):
            output[i] = self.trainer.predict(feedback)
            feedback = output[i]
        return output
    
    @property
    def w_out(self):
        return self.trainer.w_out
            