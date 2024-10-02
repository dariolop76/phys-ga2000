# PS4 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 2

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# Constants -------------------------------------------
m = 1               # mass
N = 20              # sample points for integration
# -----------------------------------------------------

# Potential V(x) = x^4
def potential(x):
    return x**4

# Function that returns the integrand
def integrand(x, A):
    return 1./np.sqrt(potential(A) - potential(x))

# Function that performs the integral using the method numpy.polynomial.legendre.leggauss
# arguments: A -> amplitude, N -> sample points
def period(A, N):

    # Evaluation of nodes and weights for the given number of sample points N
    nodes, weights = np.polynomial.legendre.leggauss(N)

    # Mapping to the new integration domain: [0, A]
    a = 0
    b = A

    nodes_new_interval = 0.5*(b-a)*nodes + 0.5*(b+a)   
    weights_new_interval = 0.5*(b-a)* weights

    # Performing the integral
    integral = np.sqrt(8*m) * sum(weights_new_interval * integrand(nodes_new_interval, A))

    return integral

# Part b) ----------------------------------------------------------------------
# Range of amplitudes
amp_range = np.arange(0.1,2.01,0.01)

# Generating data
periods = [period(A, N) for A in amp_range]

# Plotting of results
plt.figure(figsize=(8, 6))
plt.plot(amp_range, periods, color= 'purple')

plt.xlabel("Amplitude (a.u.)")
plt.ylabel("Period (a.u.)")

plt.grid(True)

plt.savefig("prob_2.png")
plt.close()
