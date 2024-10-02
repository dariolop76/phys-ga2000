# PS4 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 3 
# Parts a) and b)

# imported packages
import numpy as np
import math
import matplotlib.pyplot as plt

import hermite 


# Part a) ----------------------------------------------------------------------
# Generating data
x_points = np.linspace(-4,4,100) 

psi_functions = [hermite.psi(n,x_points) for n in range(4)]

# Plotting of results
plt.figure(figsize=(8, 5))
for i in range(4):
    plt.plot(x_points, psi_functions[i], label =f'n = {i}')

plt.xlabel("x")
plt.ylabel("Wavefunction")

plt.legend()
plt.grid(True, alpha = 0.3)

plt.savefig("prob_3a.png")
plt.close()

# Part b) ----------------------------------------------------------------------
# Generating data
x_points_b = np.linspace(-10, 10, 500) 
psi_30 = hermite.psi(30, x_points_b)

# Plotting of results
plt.figure(figsize=(8, 5))
plt.plot(x_points_b, psi_30, label ='n = 30')

plt.xlabel("x")
plt.ylabel("Wavefunction")

plt.legend()
plt.grid(True, alpha = 0.3)

plt.savefig("prob_3b.png")
plt.close()