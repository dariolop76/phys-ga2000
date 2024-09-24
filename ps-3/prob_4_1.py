# PS3 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 4 - part 1

# imported packages
import numpy as np
import matplotlib.pyplot as plt


M = 10000                       # number of samples
N_values = [5, 10, 50, 200]     # sample size (increasing values)

# Definition of function that generates the data
def data (sample_size, number_samples):
    y_set = []
    z_set = []
    for i in range(number_samples):
        x = np.random.exponential(scale = 1., size = sample_size)
        y = np.mean(x)
        z = np.sqrt(sample_size) * (y - 1)
        y_set.append(y)
        z_set.append(z)
    return y_set, z_set

# Definition of function for the gaussian distribution
def gauss(x, mean, sigma):
    return 1./(np.sqrt(2*np.pi) * sigma) * np.exp(-0.5 * (x - mean)**2/sigma**2) 

# Arrays which will contain the data to be plotted
z_histo = []
x_gauss_plot = []
z_gauss_plot = []

# Generation of the data
for N in N_values:

    # Generation of values for the variable z 
    histo_data = data(N,M)[1]                               
     
    # Generation of values for the plotting of the gaussian distribution
    mu, sigma = 0, 1
    x_gauss = np.linspace(mu -4*sigma, mu + 4*sigma, 200)
    z_gauss = gauss(x_gauss, mu, sigma)

    # Saving the data for the plot
    z_histo.append(histo_data)
    x_gauss_plot.append(x_gauss)
    z_gauss_plot.append(z_gauss)
    
    
# Plotting of results
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

for i in range(len(N_values)):
    axes[i].hist(z_histo[i], bins = 70, density=True, color = 'darkviolet')
    axes[i].plot(x_gauss_plot[i], z_gauss_plot[i], color = 'red')
    axes[i].set_title(f'N =  {N_values[i]}',)
    
    if i == 2 or i == 3:
        axes[i].set_xlabel("z")

    if i == 0 or i == 2:
        axes[i].set_ylabel("Frequency")    

plt.savefig("prob_4_1.png")
