# PS1 -- Dario Loprete -- dl5521@nyu.edu

# imported packages 
import matplotlib.pyplot as plt
import numpy as np

# definition of the function which returns the gaussian
def gauss(x, mean, sigma):
    return 1./(np.sqrt(2*np.pi) * sigma) * np.exp(-0.5 * (x - mean)**2/sigma**2) 

# creation of the sample for plot
mean = 0
sigma = 3

x = np.linspace(-10, 10, 200, dtype="float32")
y = gauss(x, mean, sigma)

# plot settings
plt.xlabel("x")
plt.ylabel("y")

plt.xlim([-10,10])
plt.ylim([0,0.18])

plt.title("Gaussian function with mean = 0, standard deviation = 3")
plt.grid(linestyle = '--', linewidth = 0.4)

# plot
plt.plot(x,y)
plt.savefig("gaussian.png")