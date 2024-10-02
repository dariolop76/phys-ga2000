# PS4 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 3 
# Part c)

# imported packages
import numpy as np
import math
import matplotlib.pyplot as plt

import hermite 

# Constants -------------------------------------------
N = 100              # sample points for integration
# -----------------------------------------------------

# Part c) ----------------------------------------------------------------------
# Function that returns the integrand
def integrand(n, x):
    return x**2 * hermite.psi(n, x)**2

# Function that performs the integral using the method numpy.polynomial.legendre.leggauss
def rms_squared(n):

    # Evaluation of nodes and weights for the given number of sample points N
    nodes, weights = np.polynomial.legendre.leggauss(N)

    # Change of variable (to perform the integral between -inf and +inf)
    # x = tan (pi/2 y)-> dx = pi/2 1/cos^2(pi/2 y) dy; x = -inf -> y = -1, x = +inf -> y = +1 
    new_x = np.tan(0.5*np.pi*nodes)
    jacobian = 0.5*np.pi/np.cos(0.5*np.pi*nodes)**2

    # Performing the integral
    integral = sum(weights* integrand(n, new_x)* jacobian)

    return integral

# Check n = 5 
num_res = np.sqrt(rms_squared(5))
res = np.sqrt(0.5*11)
print("rms for n = 5: ", num_res)
print("numerical/analytical: ", num_res/res)
print("num - analyt: ", num_res - res)