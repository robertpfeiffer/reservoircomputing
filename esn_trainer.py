import numpy as np

class LinearRegressionTrainer(object):
    """ Trains an esn and uses it for prediction """

    def __init__(self, machine):
        self.machine = machine
        self.w_out = None

    def train(self, train_input, train_target):
        X = self._createX(train_input)
        self.w_out = np.linalg.lstsq(X, train_target)[0]

    def predict(self, test_input):
        X = self._createX(test_input)
        Y = np.dot(X, self.w_out) 
        return Y
    
    def _createX(self, data):
        state_echo = self.machine.run_batch(data)
        X = np.append(np.ones((state_echo.shape[0],1)), state_echo, 1)
        return X



