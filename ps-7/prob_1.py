# PS7 -- Dario Loprete -- dl5521@nyu.edu
# Problem 1

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# Parameters
M_e = 5.974e24      # Earth mass (kg)
M_s = 1.989e30      # Sun mass (kg)
M_m = 7.348e22      # Moon mass (kg)
M_j = 1.898e27      # Jupiter mass (kg)

R_e_m = 3.844e8     # distance Earth-Moon (m)
R_e_s = 1.496e11    # distance Earth-Sun (m)

# Implementation of Newton's method
# Function we want to find the root of
def f(r,m):
    return (r**5 -2*r**4 + r**3 - (1-m)*r**2 + 2*r -1)

# Derivative of f 
def df_dr(r,m):
    return(5*r**4 - 8*r**3 + 3*r**2 - 2*r*(1-m) +2)

# Newton's method
def newton (f,df_dr, r_i, m, tol=1e-4):

    root = r_i
    new_root = R_e_m

    condition = True
    while (condition):
        delta = - (f(root,m))/(df_dr(root,m))
        new_root = root + delta 
        root = new_root

        condition = np.abs(delta) > tol

    return root


# Computing roots
root_e_m = R_e_m * newton(f,df_dr, 0.5, M_m/M_e)
root_e_s = R_e_s * newton(f,df_dr, 0.5, M_e/M_s)
root_j_s = R_e_m * newton(f,df_dr, 0.5, M_j/M_s)

print("Lagrange point Earth - Moon: {:.3e} m".format(root_e_m))
print("Lagrange point Earth - Sun: {:.3e} m".format(root_e_s))
print("Lagrange point Jupiter-mass planet - Sun: {:.3e} m".format(root_j_s))
