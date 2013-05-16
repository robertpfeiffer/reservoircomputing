import numpy as np
import random
import collections

lastNum = None
lastLetter = None
count_AX = 0
cont_AX = 0
 
lastNum = ""
lastLetter = ""
 
def _nextOutput(nextInput):
    global lastNum, lastLetter
    if nextInput in [0, 1]: #0:1  1:2
        lastNum = nextInput
        lastLetter = []
        return 0 #0: L
    elif nextInput in [3,4]: #3:A  4:B
        lastLetter = nextInput
        return 0
    elif nextInput in [6,7]: #6:X 7:Y
        if (lastNum==0 and lastLetter==3 and nextInput==6) or \
            (lastNum==1 and lastLetter==4 and nextInput==7): #"1AX" or #2BY"
            return 1 #1: R
        return 0
    return 0    
    
def one_two_ax_task(length):
    """ This is a continuous performance task. 
    The input symbols are coded by population coding (input dimension for symbol)
    Input: Random Sequence of numbers and letters: 1,2,A,X,B,Y, 3,C,Z (last three are called distractors)
    Target: 0 or 1. 
    1 and 2 mark the beginning of an outer loop
    When 1 is encountered: target is the sequence AX -> at X a 1 is expected.
    When 2 is encountered target is BY.
    
    http://en.wikipedia.org/wiki/1-2-AX_working_memory_task
    """
    input_1 = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    input_2 = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    input_3 = [0, 0, 1, 0, 0, 0, 0, 0, 0]  
    input_A = [0, 0, 0, 1, 0, 0, 0, 0, 0]  
    input_B = [0, 0, 0, 0, 1, 0, 0, 0, 0]  
    input_C = [0, 0, 0, 0, 0, 1, 0, 0, 0]  
    input_X = [0, 0, 0, 0, 0, 0, 1, 0, 0] 
    input_Y = [0, 0, 0, 0, 0, 0, 0, 1, 0] 
    input_Z = [0, 0, 0, 0, 0, 0, 0, 0, 1] 
    
    input_AX = np.zeros((2,9))
    input_AX[0,3] = 1
    input_AX[1,6] = 1
    
    input_BY = np.zeros((2,9))
    input_BY[0,4] = 1
    input_BY[1,7] = 1
    
    symbols = np.array([input_1, input_2, input_3, input_A, input_B, input_C, input_X, input_Y, input_Z])
    symbol_indizes = np.random.randint(0,8, length)
    symbol_indizes[0] = 0 if np.random.random()<0.5 else 1
    data = symbols[symbol_indizes]
    
    readable_symbols = np.array(['1', '2', '3', 'A', 'B', 'C', 'D', 'X', 'Y', 'Z'])
    readable_data = readable_symbols[symbol_indizes]
    
    targets = np.zeros((length, 1), dtype=int)
    for i in range(length):
        targets[i] = _nextOutput(symbol_indizes[i])
    readable_target_symbols = np.array(['_', 'L']);
    readable_targets = readable_target_symbols[targets]
    
   # print readable_data[0:100]
   #print readable_targets[0:100]
    return data, targets
    
#(data, targets) = one_two_ax_task(10000)
#print sum(targets)