import sys
import numpy as np
from datetime import datetime

def printf(format_string, *args):
    sys.stdout.write(format_string % args)  
    
def frange(start, stop, step):
    """ Uses linspace to avoid rounding problems for float ranges """
    num = np.ceil((stop - start)/step + 1)
    values = np.linspace(start, stop, num)
    return values
    
def correct_dictionary_arg(arg_string):
    """ returns a list with dictionaries from the arg_string. Separator: '#' """
    #astring = "{start_in_equilibrium: False, plots: False, bias_scaling: 1, spectral_radius: 0.94999999999999996}#{start_in_equilibrium: False, plots: False, bias_scaling: 1, spectral_radius: 0.94999999999999996}"
    dic_list = []
    parts = str(arg_string).split('#')
    for astring in parts:
        #if there are quotes around the string we have to remove them
        if (astring[0] == '\''  or astring[0] ==  '\"'):
            astring = astring[1:]
        if (astring[-1] == '\''  or astring[-1] ==  '\"'):
            astring = astring[:-1]
        
        #if there are (still) quotes inside, this should be a normal str(dic)    
        if '\'' in astring or '\"' in astring:
            #print "NO STRING CORRECTION"
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
        dic_list.append(dic)
    return dic_list

def timestamp_to_date(timestamp):
    if isinstance(timestamp, basestring):
        timestamp = datetime.fromtimestamp(float(timestamp))
    elif isinstance(timestamp, float):
        timestamp = datetime.fromtimestamp(timestamp)
    return timestamp
    
def compute_time_diff_in_ms(time1, time2):
    time1 = timestamp_to_date(time1)
    time2 = timestamp_to_date(time2)
            
    timediff = time2 - time1
    milliseconds_diff = ((timediff.days * 24 * 60 * 60 + timediff.seconds) * 1000 + timediff.microseconds / 1000)
    return milliseconds_diff