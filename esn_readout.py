import numpy as np
import reservoir
import mdp.utils

def add_noise(data, var):
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
        beta = np.dot( np.dot(targets.T,X), mdp.utils.inv(np.dot(X_T,X) + ridge*np.eye(X.shape[1]) ) )
        w_out = beta.T
    Y = np.dot(X, w_out)
    return (w_out, Y) # weights and training prediction
    
class LinearRegressionReadout(object):
    """ Trains an esn and uses it for prediction """

    def __init__(self, machine, ridge=0):
        self.machine = machine
        self.w_out = None
        self.ridge = ridge

    def train(self, train_input, train_target):
        """ returns (echos, predictions) """
        X = self._createX(train_input)
        if self.ridge==0:
            self.w_out = np.linalg.lstsq(X, train_target)[0]
        else:
            X_T = X.T
            beta = np.dot( np.dot(train_target.T,X), mdp.utils.inv(np.dot(X_T,X) + self.ridge*np.eye(X.shape[1]) ) )
            self.w_out = beta.T
        Y = np.dot(X, self.w_out)
        return (X[:, 1:],Y) #echos without constant
            
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
        
    def train(self, train_input, train_target):
        """ train_target is taken as feedback. 
        returns (states, prediction) """
        feedback = add_noise(train_target, 0.01)[:-1]; #all except the last
        self.feedback_dim=feedback.shape[1]
        if train_input is not None:
            train_input=np.hstack((train_input[1:],feedback)) #train_input[1:] so that input and feedback match
            self.input_dim=train_input.shape[1] #TODO: Muesste die Zeile vor der oberen Zeile stehen?
        else:
            train_input=feedback
            self.input_dim=0
        X, Y = self.trainer.train(train_input, train_target[1:])
        self.machine.w_feedback=self.trainer.w_out
        return (X,Y)

    def generate(self, length, initial_feedback=None):
        """ returns (states, prediction) """
        inputs=np.zeros((length,0))
        if initial_feedback is not None:
            echo1=self.machine.run_batch(initial_feedback)
            echo1=echo1[-1,:]
            return self.trainer.predict(inputs,echo1)
        else:
            #Feedback might not be equal to readouts,
            # so we need to compute them separately
            return self.trainer.predict(inputs, None)
    
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
