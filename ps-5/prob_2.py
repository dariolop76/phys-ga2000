# PS5 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 2

# imported packages
import numpy as np
import matplotlib.pyplot as plt
import math

# Integrand of Gamma Fuction
def integrand_gamma(x, a):
    return x**(a-1) * np.exp(-x)

# Integrand of Gamma Fuction part d)
def integrand_gamma_part_d(x, a):
    # Change of variable z = x/(a-1 + x)
    new_x = (a-1) * x/(1-x) 
    jacobian = (a-1)/(1-x)**2
    return jacobian*np.exp((a-1)*(np.log(new_x) -new_x/(a-1)))

# Generating data
x_points = np.linspace(0,5, 100)
curve_2 = integrand_gamma(x_points, 2)
curve_3 = integrand_gamma(x_points, 3)
curve_4 = integrand_gamma(x_points, 4)


# Integral for gamma function
def gamma(a):
    # Evaluation of nodes and weights for the given number of sample points N
    N = 50 
    nodes, weights = np.polynomial.legendre.leggauss(N)

    # Mapping to the new integration domain: [0, 1]
    c = 0
    b = 1

    nodes_new_interval = 0.5*(b-c)*nodes + 0.5*(b+c)   
    weights_new_interval = 0.5*(b-c)*weights

    # Performing the integral
    integral = sum(weights_new_interval* integrand_gamma_part_d(nodes_new_interval, a))

    return integral

# Results
a_values = [1.5, 3, 6 , 10]
results = [gamma(1.5), gamma(3), gamma(6), gamma(10)]
expected_values = [0.5*np.sqrt(np.pi), math.factorial(2), math.factorial(5), math.factorial(9)]


for i in range(len(a_values)):
    diff = results[i]-expected_values[i]
    print("Gamma of {0}: {1}".format(a_values[i], results[i]))
    print("Difference with expected values: {0} \n".format(diff))

# Plotting of results ----------------------------------------------------------------------------
plt.figure(figsize=(9,6))

plt.plot(x_points, curve_2, color = 'red', label = "a = 2")
plt.plot(x_points, curve_3, color = 'blue', label = "a = 3")
plt.plot(x_points, curve_4, color = 'orange', label = "a = 4")


plt.xlabel("x")
plt.ylabel("Integrand")

plt.legend()
plt.grid(True)
plt.savefig("prob_2.png")

plt.close()
#--------------------------------------------------------------------------------------------------
