import shelve

def save_object(obj, name, file_name):

    shelf = shelve.open(file_name)
    shelf[name] = obj
    
def load_object(name, file_name):
    
    shelf = shelve.open(file_name)
    obj = shelf[name]
    return obj