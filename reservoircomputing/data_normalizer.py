class DataNormalizer(object):
    """ Z-Score-normalization """
    def init_with_data(self, data):
        self.means = data.mean(0)
        self.stds = data.std(0)
        self.stds[self.stds == 0] = 1 #um NaN zu verhindern
        
        data -= self.means
        data /= self.stds
        return data
    
    def normalize(self, data):
        data -= self.means
        data /= self.stds
        return data
    
    def denormalize(self, data):
        data *= self.stds
        data += self.means
        return data