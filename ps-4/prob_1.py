# PS4 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 1

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# Constants -------------------------------------------
k = 1.38e-23        # Boltzmann constant (J K^-1)
T_debye = 428       # Debye temperature (K)
rho = 6.022e+22     # number density (cm^-3)
vol = 1000          # volume of solid (cm^-3)

V_rho_k = vol*rho*k

N = 50              # sample points for integration
# -----------------------------------------------------

# Function that returns the integrand
def integrand(x):
    return (x**4 * np.exp(x))/((np.exp(x) -1)**2)

# Function that performs the integral using the method numpy.polynomial.legendre.leggauss
def cv(T, N):

    # Evaluation of nodes and weights for the given number of sample points N
    nodes, weights = np.polynomial.legendre.leggauss(N)

    # Mapping to the new integration domain: [0, T_debye/T]
    a = 0
    b = T_debye/T

    nodes_new_interval = 0.5*(b-a)*nodes + 0.5*(b+a)   
    weights_new_interval = 0.5*(b-a)*weights

    # Performing the integral
    integral = 9*V_rho_k* (T/T_debye)**3 * sum(weights_new_interval * integrand(nodes_new_interval))

    return integral

# Part a) ----------------------------------------------------------------------
T_a = 50 # K
print("Heat capacity fot T = {0} K: {1} J/K".format(T_a, cv(T_a, N)))

# Part b) ----------------------------------------------------------------------
# Range of temperatures
T_range = np.arange(5,500)

# Generating data
heat_capacities = [cv(T, N) for T in T_range]

# Plotting of results
plt.figure(figsize=(8, 6))
plt.plot(T_range, heat_capacities, color= 'red')

plt.xlabel("Temperature (K)")
plt.ylabel("Heat capacity (J/K)")

plt.grid(True)

plt.savefig("prob_1_b.png")
plt.close()

# Part c) ----------------------------------------------------------------------
# Testing convergence
N_range = np.arange(10,71,10)

T_test = 50 # K
heat_capacities_test = [cv(T_test, N) for N in N_range]

# Plotting of results
plt.figure(figsize=(8, 6))
plt.plot(N_range, heat_capacities_test, marker = 'o', linestyle = '--',  color= 'green')

plt.xlabel("N")
plt.ylabel("Heat capacity (J/K)")

plt.grid(True)

plt.savefig("prob_1_c.png")
    