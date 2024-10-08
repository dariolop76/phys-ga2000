# PS5 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 1

# imported packages
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Function f(x) = 1 + 1/2 tanh(2x)
def f(x):
    return 1+0.5*np.tanh(2.*x)

# Derivative -- analytic formula -- df(x)/dx = 1-tanh^2(2x)
def df_dx(x):
    return 1- (np.tanh(2.*x))**2

# Derivative -- central difference
def df_dx_cd(x, dx):
    return 1./dx * (f(x+0.5*dx) - f(x-0.5*dx))

# Functions for jax
def f_jax(x):
    return 1+0.5*jnp.tanh(2.*x)


# Generating data
dx = 1e-5
x_points = np.linspace(-2,2,100)

analytic_derivatives = df_dx(x_points)
numerical_derivatives = df_dx_cd(x_points, dx)

errors = analytic_derivatives - numerical_derivatives


df_jax = jax.grad(f_jax)
jax_derivatives = jax.vmap(df_jax)(x_points)

errors_jax = analytic_derivatives - jax_derivatives

# Plotting of results ----------------------------------------------------------------------------
plt.figure(figsize=(9,6))

plt.plot(x_points, analytic_derivatives, color = 'red', label = "Analytic")
plt.plot(x_points, numerical_derivatives, '.', color = 'blue', alpha = 0.5, label = "Numerical")

plt.xlabel("x")
plt.ylabel("df/dx")

plt.legend()
plt.grid(True)
plt.savefig("prob_1_derivative.png")

plt.close()
#--------------------------------------------------------------------------------------------------
plt.figure(figsize=(9,6))

plt.plot(x_points, errors, color = 'blue')

plt.xlabel("x")
plt.ylabel("Error")


plt.grid(True)
plt.savefig("prob_1_error.png")

plt.close()
#--------------------------------------------------------------------------------------------------
plt.figure(figsize=(9,6))

plt.plot(x_points, errors_jax, color = 'green')

plt.xlabel("x")
plt.ylabel("Error (jax)")


plt.grid(True)
plt.savefig("prob_1_error_jax.png")

plt.close()