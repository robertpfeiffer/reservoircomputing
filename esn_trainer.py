import numpy as np

""" Trains an esn and uses it for prediction """
class LinearRegressionTrainer(object):

    def __init__(self, machine):
        self.machine = machine
        self.w_out = None

    def train(self, train_input, train_target):
        X = self._createX(train_input)
        self.w_out = np.dot(np.linalg.pinv(X), train_target)

    def predict(self, test_input):
        X = self._createX(test_input)
        Y = np.dot(X, self.w_out) 
        return Y
    
    def _createX(self, data):
        self.machine.run2(data)
        X = np.append(np.ones((self.machine.state_echo.shape[0],1)), self.machine.state_echo, 1)
        return X



