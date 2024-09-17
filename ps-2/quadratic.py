# PS2 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 5 - part c)

# imported packages
import numpy as np


# definition of function that gives the solutions for a quadratic equation with method b)
def quadratic(a, b, c):
    
    delta = np.sqrt(b**2 - 4*a*c)

    if (np.abs(b - delta) < 0.001):                 # if this difference is smaller than the threshold 0.001 then the solutions
       x_plus = (2*c)/ (-b - delta)                 # can be computed as follows: x_plus from method b, x_minus from method_a
       x_minus = (1./(2*a)) * (-b - delta)          # this way, we obtain the accurate solutions avoiding evaluating -b + delta
    
    else:                                           # if the difference is not small, we can apply method a) as standard method
       x_plus = (1./(2*a)) * (-b + delta)
       x_minus = (1./(2*a)) * (-b - delta)

    return (x_plus, x_minus)