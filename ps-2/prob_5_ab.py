# PS2 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 5 - part a) and b)

# imported packages
import numpy as np

# definition of function that gives the solutions for a quadratic equation with method a)
def quadratic_a(a, b, c):
    
    delta = np.sqrt(b**2 - 4*a*c)
    x_plus = (1./(2*a)) * (-b + delta)
    x_minus = (1./(2*a)) * (-b - delta)
    
    return (x_plus, x_minus)

# definition of function that gives the solutions for a quadratic equation with method b)
def quadratic_b(a, b, c):
    
    delta = np.sqrt(b**2 - 4*a*c)
    x_plus = (2*c)/ (-b - delta)
    x_minus = (2*c)/ (-b + delta)
    
    return (x_plus, x_minus)


# given input parameters
a = 0.001
b = 1000.
c = 0.001

# printing of solutions
solutions_a = quadratic_a(a,b,c)
solutions_b = quadratic_b(a,b,c)

print("Solutions with method a:\n x+ = {0} \n x- = {1}".format(solutions_a[0], solutions_a[1]))
print("Solutions with method b:\n x+ = {0} \n x- = {1}".format(solutions_b[0], solutions_b[1]))