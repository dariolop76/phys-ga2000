# PS4 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 3 
# Part d)

# imported packages
import numpy as np
import math
import matplotlib.pyplot as plt

import scipy
import scipy.special 

import hermite 

# Constants -------------------------------------------
N = 9              # sample points for integration
# -----------------------------------------------------

# Part c) ----------------------------------------------------------------------
# Function that returns the integrand
def integrand(n, x):
    return x**2 * (np.exp(0.5*x**2)*hermite.psi(n, x))**2

# Function that performs the integral using the method numpy.polynomial.legendre.leggauss
def rms_squared(n):

    # Evaluation of nodes and weights for the given number of sample points N
    nodes, weights = np.polynomial.hermite.hermgauss(N)

    # Performing the integral
    integral = sum(weights* integrand(n, nodes))

    return integral

# Check n = 5 
num_res = np.sqrt(rms_squared(5))
res = np.sqrt(0.5*11)
print("rms for n = 5: ", num_res)
print("numerical/analytical: ", num_res/res)
print("num - analyt: ", num_res - res)

