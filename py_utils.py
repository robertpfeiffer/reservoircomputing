import sys
import numpy as np

def printf(format_string, *args):
    sys.stdout.write(format_string % args)  
    
def frange(start, stop, step):
    """ Uses linspace to avoid rounding problems for float ranges """
    num = np.ceil((stop - start)/step + 1)
    values = np.linspace(start, stop, num)
    return values
    
def correct_dictionary_arg(astring):
    #astring = "{start_in_equilibrium: False, plots: False, bias_scaling: 1, spectral_radius: 0.94999999999999996}"
    astring = str(astring)
    
    #if there are quotes around the string we have to remove them
    if (astring[-1] == '\''  or astring[-1] ==  '\"'):
        astring = astring[1:-1]
    
    #if there are (still) quotes inside, this should be a normal str(dic)    
    if '\'' in astring or '\"' in astring:
        print "NO STRING CORRECTION"
        return eval(astring)
    
    astring = astring.replace('{', '')
    astring = astring.replace('}', '')
    key_value_pairs = astring.split(',')
    corrected_string = '{'
    for key_value in key_value_pairs:
        parts = key_value.split(':')
        key = parts[0].strip()
        value = parts[1].strip()
        corrected_string+='\"'+key+'\"'+': '+value+','
    corrected_string = corrected_string[:-1]
    corrected_string += '}'
    #print 'CORRECTED: ', corrected_string
    dic = eval(corrected_string)
    return dic