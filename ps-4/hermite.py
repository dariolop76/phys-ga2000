# PS4 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 3 
# hermite.py

# imported packages
import numpy as np
import math


# Hermite polynomial function
# arguments: n -> order of the polynomial, x -> point of evaluation (in our case a numpy array)
def H(n,x):
    # Case n = 0 -> H0(x) = 1
    if (n == 0):
        return np.ones(len(x))
    
    # Case n = 1 -> H1(x) = 2x
    elif (n == 1):
        return 2*x
    
    # General case: we apply the recursive formula
    else:
        return 2*x*H(n-1,x) - 2*(n-1)*H(n-2,x)

# Wavefunction
def psi(n,x):
    return 1./np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi)) * np.exp(-0.5*x**2) * H(n,x)
