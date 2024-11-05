# PS8 -- Dario Loprete -- dl5521@nyu.edu


# imported packages
import numpy as np
import matplotlib.pyplot as plt


# Loading data 
data = np.loadtxt("dow.txt")

# Plotting of data and computation of FFT
# Plotting of data
x = np.arange(len(data))
plt.figure(figsize=(9,6))

plt.plot(x, data, marker = '.', markersize = 0.1, color = 'b')

plt.xlabel("Time since 2006 (days)")
plt.ylabel("Daily closing value ($)")

plt.savefig("prob_3_data.png")
plt.close()


# FFT 
coeffs = np.fft.rfft(data)

# We set to zero all the coefficients except for the first 10 % 
coeffs[int(len(coeffs) * 0.1):] = 0 

# ... and reconstruct the data
data_10 = np.fft.irfft(coeffs)

# Plotting
plt.figure(figsize=(9,6))

plt.plot(x, data, marker = '.', markersize = 0.1, color = 'b', label = 'original data')
plt.plot(x, data_10, marker = '.', markersize = 0.1, color = 'r', label = 'reconstructed data (10 per cent)')


plt.xlabel("Time since 2006 (days)")
plt.ylabel("Daily closing value ($)")

plt.legend()
plt.savefig("prob_3_reconstr_data_10.png")
plt.close()

# We set to zero all the coefficients except for the first 2 % 
coeffs = np.fft.rfft(data)
coeffs[int(len(coeffs) * 0.02):] = 0 

# ... and reconstruct the data
data_2 = np.fft.irfft(coeffs)

# Plotting
plt.figure(figsize=(9,6))

plt.plot(x, data, marker = '.', markersize = 0.1, color = 'b', label = 'original data')
plt.plot(x, data_2, marker = '.', markersize = 0.1, color = 'purple', label = 'reconstructed data (2 per cent)')


plt.xlabel("Time since 2006 (days)")
plt.ylabel("Daily closing value ($)")

plt.legend()
plt.savefig("prob_3_reconstr_data_2.png")
plt.close()

