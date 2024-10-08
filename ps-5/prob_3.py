# PS5 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 3

# imported packages
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import math

# Loading data --------------------------------------------------------------
# Cleaning from '|' and ' '
with open('signal.dat', 'r') as file:
    lines = file.readlines()
signal_data = [line.replace('|', ' ').strip() for line in lines[1:]]

time, signal = np.loadtxt(signal_data, unpack = True)
#--------------------------------------------------------------------------------------------------

# Plotting data -----------------------------------------------------------------------------------
plt.figure(figsize=(9,6))

plt.plot(time, signal, '.')

plt.xlabel("Time (s)")
plt.ylabel("Signal (a.u.)")

plt.grid(True, alpha = 0.6)
plt.savefig("prob_3_data.png")

plt.close()
#-------------------------------------------------------------------------------------------------

# Third-order polynomial fit-----------------------------------------------------------------------
# model: f_3 (t, parameters) = para_0 + para_1 * time + para_2 * time^2 + para_3 * time^3

# Rescaling the time
time_rescaled = (time - np.mean(time))/np.std(time) 

# Design matrix
A = np.zeros((len(time_rescaled), 4))
A[:, 0] = 1.
A[:, 1] = time_rescaled 
A[:, 2] = time_rescaled**2
A[:, 3] = time_rescaled**3

# Applying SVD
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
condition_number = np.max(w)/np.min(w)


# Computing the inverse of A
Ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())

# Evaluating values for the fit
fit = A.dot(Ainv.dot(signal))

# Printing of results
print("3rd order polynomial fit --------------------------")
print("Condition number: ", condition_number, "\n")

# Plotting data + fit 
plt.figure(figsize=(9,6))

plt.plot(time, signal, '.', label = 'Data')
plt.plot(time, fit, '.',label = 'Third-order polynomial fit')

plt.xlabel("Time (s)")
plt.ylabel("Signal (a.u.)")

plt.legend()
plt.grid(True, alpha = 0.6)
plt.savefig("prob_3_fit_pol.png")

plt.close()

# Plotting residuals
res = signal - fit
plt.figure(figsize=(9,6))

plt.plot(time, res, '.')
plt.xlabel("Time (s)")
plt.ylabel("Residuals (a.u.)")

plt.grid(True, alpha = 0.6)
plt.savefig("prob_3_res.png")

plt.close()

#--------------------------------------------------------------------------------------------------

# Finding the optimal polynomial fit ----------------------------------------------------------------
'''
Approach: we fit the data with polynomials of increasing order, starting with order n = 3 (= n_plus_one -1 for
          computational purposes) and increase this order by one at each step. We do this until the condition number
          reaches a threshold, which we chose to be cn_threshold = 1e+12, which is smaller than the inverse of the
          machine precision ( = 1e+15). The optimal fit is obtained by finding for which polynomial fit the
          mean of the absolute value of the residuals is the smallest
'''

print("nth order polynomial fit --------------------------")

n_plus_one = 4          # order n for the polynomial fit: n = n_plus_one -1
cn_threshold = 1e+12    # acceptance threshold for the condition number
std_data = 2            # uncertainty on the signal values

# Arrays which will contain the data for the fit, the residuals and the condition number
fits = []
residuals = []
condition_numbers = []

# Loop over the order of the polynomial fit
while (True):
    
    # Design matrix
    A = np.zeros((len(time_rescaled), n_plus_one))

    A[:, 0] = 1.
    for i in range(1, n_plus_one):          # the for loop runs up to (n_plus_one -1) which is n
        A[:, i] = time_rescaled**(i)  
    
    # Applying SVD
    (u, w, vt) = np.linalg.svd(A, full_matrices=False)

    # Computing the inverse of A
    Ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())

    # Evaluating values for the fit
    fit = A.dot(Ainv.dot(signal))
    fits.append(fit)

    # Evaluating residuals
    residuals.append(signal - fit)

    # Evaluating condition number
    condition_number = np.max(w)/np.min(w)
    condition_numbers.append(condition_number)

    if (condition_number  > cn_threshold): 
        print("Loop stops at order n = ", n_plus_one-1)
        break
    
    n_plus_one += 1

# Finding the order such that the mean of the residuals (in abs value) is the smallest
mean_values = np.mean(np.abs(residuals), axis=1)
optimal_index = np.argmin(mean_values)

n_optimal = optimal_index + 3               # the +3 comes from the fact that index 0 corresponds to n = 3


print("Optimal order: n = ", n_optimal)
print("Condition number for n = {0}: {1:.2e}".format(n_optimal, condition_numbers[optimal_index]), "\n")


# Plotting data + fit 
plt.figure(figsize=(9,6))

plt.plot(time, signal, '.', label = 'Data')
plt.plot(time, fits[optimal_index], '.',label = f'{n_optimal}th order polynomial fit')

plt.xlabel("Time (s)")
plt.ylabel("Signal (a.u.)")

plt.legend()
plt.grid(True, alpha = 0.6)
plt.savefig("prob_3_opt_fit_pol.png")

plt.close()

# Plotting residuals
res = signal - fit
plt.figure(figsize=(9,6))

plt.plot(time, residuals[optimal_index], '.')
plt.xlabel("Time (s)")
plt.ylabel("Residuals (a.u.)")

plt.grid(True, alpha = 0.6)
plt.savefig("prob_3_res_opt_fit_pol.png")

plt.close()

# Sin/cos fit----------------------------------------------------------

n_freq_plus_one = 11
time_span = np.max(time_rescaled) - np.min(time_rescaled)


# Design matrix
A = np.zeros((len(time_rescaled), n_plus_one))

A[:, 0] = 1.
for i in range(1, n_freq_plus_one, 2):  
    mode = (i+1) // 2        
    A[:, i]   = np.sin(2*np.pi * mode/(0.5*time_span) * time_rescaled)
    A[:, i+1] = np.cos(2*np.pi * mode/(0.5*time_span) * time_rescaled)   
    
#Applying SVD
(u, w, vt) = np.linalg.svd(A, full_matrices=False)

# Computing the inverse of A
Ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())

# Evaluating values for the fit
fit = A.dot(Ainv.dot(signal))


# Evaluating residuals
residual= signal - fit

# Evaluating condition number
condition_number = np.max(w)/np.min(w)

print("Harmonic sequence fit ----------------------------------------")
print("Condition number: ", condition_number)
# Plotting data + fit 
plt.figure(figsize=(9,6))

plt.plot(time, signal, '.', label = 'Data')
plt.plot(time, fit, '.',label = 'Harmonic sequence fit')

plt.xlabel("Time (s)")
plt.ylabel("Signal (a.u.)")

plt.legend()
plt.grid(True, alpha = 0.6)
plt.savefig("prob_3_harm_seq.png")

plt.close()

# Plotting residuals
res = signal - fit
plt.figure(figsize=(9,6))

plt.plot(time, residual, '.')
plt.xlabel("Time (s)")
plt.ylabel("Residuals (a.u.)")

plt.grid(True, alpha = 0.6)
plt.savefig("prob_3_res_harm_seq.png")

plt.close()    

print("Estimate for the period----------------------------------------")
print("T: {0:.2e} s".format((np.max(time) - np.min(time))/7))



