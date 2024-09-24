# PS3 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 4 - part 2

# imported packages
import numpy as np
import matplotlib.pyplot as plt


M = 1000                                # number of samples (here we use a smaller M than that of part 1
                                        # given that more computations are involved in this part)
N_values = np.arange(1,1000,2)          # sample size values 

# Definition of function that generates the data
def data (sample_size, number_samples):
    y_set = []
    for i in range(number_samples):
        x = np.random.exponential(scale = 1., size = sample_size)
        y = np.mean(x)
        y_set.append(y)
    return y_set


# Definition of function which computes the mean, the variance, the skewness and the kurtosis from the data
def moments (data):
    m = np.mean(data)                                   # mean
    v = np.var(data)                                    # variance
    s = 1./M * np.sum((data - m)**3)/(v)**1.5           # skewness
    k = -3 + 1./M * np.sum((data - m)**4)/(v**2)        # kurtosis

    return m, v, s, k

# Arrays which will contain the data to be plotted
mean_values = []
var_values = []
skew_values = []
kurt_values = []

# Generation of the data
for N in N_values:

    histo_data = data(N,M)

    moments_values = moments(histo_data)
    mean_values.append(moments_values[0])
    var_values.append(moments_values[1])
    skew_values.append(moments_values[2])
    kurt_values.append(moments_values[3])

# Loop over values of skewness to find at which N 1% of initial value is reached
for i in range(len(N_values)):
    if (skew_values[i] <= 0.01*skew_values[0]):
        N_skew = N_values[i]
        break

# Loop over values of kurtosis to find at which N 1% of initial value is reached
for i in range(len(N_values)):
    if (kurt_values[i] <= 0.01*kurt_values[0]):
        N_kurt= N_values[i]
        break

if 'N_skew' in locals():
    print("Estimate of N such that skewness is 1% of initial value:", N_skew, "\n")

if 'N_kurt' in locals():
    print("Estimate of N such that kurtosis is 1% of initial value:", N_kurt, "\n")


# Plotting of results
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax1, ax2, ax3, ax4 = axes.flatten()

ax1.plot(N_values, mean_values)
ax2.plot(N_values, var_values)
ax3.plot(N_values, skew_values)
ax4.plot(N_values, kurt_values)

ax1.set_title("Mean")
ax2.set_title("Variance")
ax3.set_title("Skewness")
ax4.set_title("Kurtosis")

ax3.set_xlabel("N")
ax4.set_xlabel("N")


plt.savefig("prob_4_2.png")
