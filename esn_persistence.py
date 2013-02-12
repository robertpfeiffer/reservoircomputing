import shelve
import numpy as np

def save_object(obj, name, file_name):
    shelf = shelve.open(file_name)
    shelf[name] = obj
    
def load_object(name, file_name):
    shelf = shelve.open(file_name)
    obj = shelf[name]
    return obj

def save_arrays(file_name, *arrays):
    np.savez(file_name, arrays)
    
def load_arrays(file_name):
    if (not (file_name.endswith('.npy') or file_name.endswith('.npz'))):
        file_name = file_name+str('.npz')
    npzfile = np.load(file_name)
    arrays = []
    for i in range(len(npzfile.files)):
        arrays.append(npzfile['arr_'+str(i)])
    return tuple(arrays)

#Puffern von files
loaded_files = {}
def load_file(filename):
    if filename in loaded_files:
        return loaded_files[filename]
    f = open(filename, 'r')
    lines = f.readlines()
    loaded_files[filename] = lines
    return lines