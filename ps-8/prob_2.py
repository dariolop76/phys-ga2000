# PS8 -- Dario Loprete -- dl5521@nyu.edu


# imported packages
import numpy as np
import matplotlib.pyplot as plt


# Loading data 
piano = np.loadtxt("piano.txt")
trumpet = np.loadtxt("trumpet.txt")

# Plotting of data and computation of FFT
# Plotting of piano waveform
x = np.arange(len(piano))
plt.figure(figsize=(9,6))

plt.plot(x, piano, marker = '.', markersize = 0.1, color = 'b')

plt.title("Piano (waveform)")
plt.xlabel("t (a.u.)")
plt.ylabel("Waveform")

plt.savefig("prob_2_piano_data.png")
plt.close()

# Plotting of trumpet waveform
x = np.arange(len(trumpet))
plt.figure(figsize=(9,6))

plt.plot(x, trumpet, marker = '.', markersize = 0.1, color = 'g')

plt.title("Trumpet (waveform)")
plt.xlabel("t (a.u.)")
plt.ylabel("Waveform")

plt.savefig("prob_2_trumpet_data.png")
plt.close()

# FFT and plotting of coefficients
piano_coeff = np.fft.rfft(piano)
trumpet_coeff = np.fft.rfft(trumpet)

k = np.arange(10000)
plt.figure(figsize=(9,6))

plt.plot(k, np.abs(piano_coeff[:10000]), marker = '.', markersize = 1, color = 'b')

plt.title("Piano (spectrum)")
plt.xlabel(r'$k$')
plt.ylabel(r'$|c_k|$')

plt.savefig("prob_2_piano_spectrum.png")
plt.close()

plt.figure(figsize=(9,6))

plt.plot(k, np.abs(trumpet_coeff[:10000]), marker = '.', markersize = 1, color = 'g')

plt.title("Trumpet (spectrum)")
plt.xlabel(r'$k$')
plt.ylabel(r'$|c_k|$')

plt.savefig("prob_2_trumpet_spectrum.png")
plt.close()

# Finding the note
scale = 44100./len(piano)
k = np.arange(4000)

plt.figure(figsize=(9,6))

plt.plot(scale*k, np.abs(piano_coeff[:4000]), marker = '.', markersize = 1, color = 'b', label = 'piano')
plt.plot(scale*k, np.abs(trumpet_coeff[:4000]), marker = '.', markersize = 1, color = 'g', label = 'trumpet')

plt.title("Spectra")
plt.xlabel(r'$Frequency (Hz)$')
plt.ylabel(r'$|c_k|$')

plt.savefig("prob_2_spectra.png")
plt.close()

freq = scale*k[np.argmax(piano_coeff)]
print(f"Fundamental frequency: {freq} Hz")