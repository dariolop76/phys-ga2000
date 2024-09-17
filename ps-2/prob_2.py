# PS2 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 2

# imported packages
import numpy as np

# 32-bit precision
def info_32_bit():

    # smallest increment
    '''
    Method: we add an increment to 1, starting from 1 + 1. For each step, we check if the sum is different
              from 1. If it is, we divde the increment by 2 and repeat. When the sum is not different from 1
              then we have reached the smallest increment we can represent in 32-bit. 
    '''
    x = np.float32(1.0)
    step = np.float32(1.0)
    while True:
        x_new = x + step
        if x_new != x:
            smallest_32 = step
        else:   
            break 
    
        step /= np.float32(2.0)      

    # minimum number
    '''
    Method: we start from 1 and we divide this number by 2 for each step, in order to reach the minimum value 
              we can represent in 32-bit. When this number becomes 0, it means underflow has occurred, so we stop
              the loop.
    '''
    min = np.float32(1.0)
    while True:
        min /= np.float32(2)
        if min > 0:
            minimum_32 = min
        else:   
            break 
    
    # maximum number
    '''
    Method: we start from 1 and we multiply this number by 2 for each step, in order to reach the maximum value 
              we can represent in 32-bit. When this number becomes infinity, it means overflow has occurred, so we stop
              the loop.
    '''
    max = np.float32(1.0)
    while True:
        max_new = max * np.float32(2)
        if ~np.isfinite(max_new):
            maximum_32 = max
            break
    
        max = max_new
    
    print("Smallest value you can add up to 1 in 32-bit: ", smallest_32) 
    print("Minimum positive number we can represent in 32-bit: ", minimum_32)
    print("Maximum positive number we can represent in 32-bit: ", maximum_32)

# 64-bit precision
'''
Method: it is the same as the one used for 32-bit.
'''
def info_64_bit():

    # smallest increment
    x = np.float64(1.0)
    step = np.float64(1.0)
    while True:
        x_new = x + step
        if x_new != x:
            smallest_64 = step
        else:   
            break 
    
        step /= np.float64(2.0)

    # minimum number
    min = np.float64(1.0)
    while True:
        min /= np.float64(2)
        if min > 0:
            minimum_64 = min
        else:   
            break 
    
    # maximum number
    max = np.float64(1.0)
    while True:
        max_new = max * np.float64(2)
        if ~np.isfinite(max_new):
            maximum_64 = max  
            break 
    
        max = max_new

    print("Smallest value you can add up to 1 in 64-bit: ", smallest_64)   
    print("Minimum positive number we can represent in 64-bit: ", minimum_64)
    print("Maximum positive number we can represent in 64-bit: ", maximum_64)


# printing of results
info_32_bit()
info_64_bit()

# printing of exact info
print(np.finfo(np.float32))
print(np.finfo(np.float64))