import shelve

def save_object(obj, name):

    shelf = shelve.open('shelved_esns.txt')
    shelf[name] = obj
    
def load_object(name):
    
    shelf = shelve.open('shelved_esns.txt')
    obj = shelf[name]
    return obj