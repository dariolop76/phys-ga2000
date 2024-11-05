# PS8 -- Dario Loprete -- dl5521@nyu.edu


# imported packages
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import scipy.optimize as optimize
import jax
import jax.numpy as jnp
import csv

# Reading data ------------------------------------------------------------------------------------------------------
data_array = []

with open('survey.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader, None)  
    for row in csv_reader:
        data_array.append(row)

# Saving data 
data = np.array(data_array, dtype = float)
ages = data[:,0]
answers = data [:,1]

# Model: logistic function
def model (x, params):
    return 1./(1 + jnp.exp(-params[0] - params[1]*x))

# Likelihood function
def like(params, x, y):
    return np.prod(model(ages, params)**y *(1-model(x, params))**(1-y))


# Negative logarithm of likelihood function
def negloglike(params, x, y):
    p = model(x, params)
    nll = y*jnp.log(p) + (1-y)*jnp.log(1-p)
    return -nll.sum()
    
# Minimizing -nll, thus obtaining the parameters that maximize the likelihood -----------------------------------------------
negloglike_grad = jax.grad(negloglike) 

init = np.array([-3,0.2])  # initial guess for beta0, beta1
result = optimize.minimize(negloglike, init, args=(ages, answers), method='BFGS',jac = negloglike_grad, tol = 1e-8)


# Maximum likelihood value, formal errors and covariance matrix -------------------------------------------------------------
# Covariance matrix = inverse of hessian matrix: we compute the hessian matrix using jax
def hessian(params, x, y):
    return jax.jacfwd(jax.grad(negloglike))(params, x, y)

h = hessian(result.x, ages, answers)

hessian_matrix = hessian(result.x, ages, answers)
covariance_matrix = np.linalg.inv(hessian_matrix)

# Errors: square root of diagonal components of covariance matrix
errors = np.sqrt(np.diag(covariance_matrix))

# Maximum likelihood value: its given by the function like evaluated at the estimated paramters
likelihood = like(result.x, ages, answers)


# Printing of results ------------------------------------------------------------------------------------------------------
print("Parameters beta0 and beta1:")
print(f"beta0: {result.x[0]} \nbeta1: {result.x[1]}")
print("\nCovariance Matrix:")
print(covariance_matrix)
print(f"\nError for beta0: {errors[0]}")
print(f"Error for beta1: {errors[1]}")
print(f"\nMaximum Likelihood value: {likelihood}")
print("\nFinal estimate of parameters: ")
print(f"beta0 = {result.x[0]:.0f} pm {errors[0]:.0f}")
print(f"beta1 = {result.x[1]:.2f} pm {errors[1]:.2f}")

# Plotting of results ------------------------------------------------------------------------------------------------------
x = np.arange(np.min(ages), np.max(ages), 0.1)
model_plot= model(x,(result.x[0],result.x[1])) 

plt.figure(figsize=(9,6))

plt.plot(x,model_plot,label="Fit")
plt.plot(ages,answers,".",label="Data")

plt.xlabel("Ages (yrs)")
plt.ylabel("Answers")

plt.legend()
plt.grid(True)
plt.savefig("prob_1.png")