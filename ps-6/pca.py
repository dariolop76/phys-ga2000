# PS6 -- Dario Loprete -- dl5521@nyu.edu
# Principal Component Analysis

# imported packages
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import math
import astropy
from astropy.io import fits 
import timeit


# Part a)-----------------------------------------------------------------------------------------
# Loading data 
hdu_list = astropy.io.fits.open("specgrid.fits")
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

print("Dim array logwave: ", logwave.shape)
print("Dim array flux: ", flux.shape)

# flux[i] -> data for the i^th galaxy

# Plotting data 
plt.figure(figsize=(9,6))

galaxies = [0,1,2,3,4]
for i in galaxies:
    plt.plot(logwave, flux[i],  label = f'Galaxy # {i+1}')

plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel(r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")


plt.legend()
plt.grid(True, alpha = 0.6)
plt.savefig("pca_data.png")

plt.close()

# Part b)-----------------------------------------------------------------------------------------
# Normalization procedure
# Performing the integration of the data over the wavelength interval for each galaxy
flux_integrals = np.array([np.trapz(flux[i], logwave) for i in range(len(flux))])

# Normalizing the flux
flux_normalized = flux / flux_integrals[:, np.newaxis]

# Plotting a the normalzed flux
plt.figure(figsize=(9,6))

galaxies = [0,1,2,3,4]
for i in galaxies:
    plt.plot(logwave, flux_normalized[i],  label = f'Galaxy # {i+1}')

plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel(r"Flux (a.u.)")


plt.legend()
plt.grid(True, alpha = 0.6)
plt.savefig("pca_data_normalized.png")

plt.close()

# Part c)-----------------------------------------------------------------------------------------
# Calculating residuals
# Evaluating the mean value of the data over the wavelength interval for each galaxy
mean_flux = np.mean(flux_normalized, axis=1)

# Evaluating the residual
residuals = flux_normalized - mean_flux[:, np.newaxis]

# Plotting the residuals
plt.figure(figsize=(9,6))

galaxies = [0,1,2,3,4]
for i in galaxies:
    plt.plot(logwave, residuals[i],  label = f'Galaxy # {i+1}')

plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel(r"Flux (a.u.)")


plt.legend()
plt.grid(True, alpha = 0.6)
plt.savefig("pca_data_residuals.png")

plt.close()


# Part d)-----------------------------------------------------------------------------------------
# Covariant matrix
C = np.dot(residuals.T,residuals)

# Computing eigenvalues and eigenvectors of C
eigenvalues, eigenvectors = np.linalg.eig(C)

# Plotting the first 5 eigenvectors
for i in range(5):
    plt.figure(figsize=(9,6))
    plt.plot(logwave, np.real(eigenvectors[:,i]), label = f'Eigenvector {i+1}')
    plt.xlabel(r"$\log_{10}(\lambda)$")
    plt.ylabel('Eigenvector value')

    plt.legend()
    plt.grid(True, alpha = 0.6)
    plt.savefig(f'pca_eigenvector_{i+1}.png')

    plt.close()

# Part e)-----------------------------------------------------------------------------------------
# Performing the SVD decomposition
(u, w, vt) = np.linalg.svd(residuals)
svd_eigenvectors = vt.T

# Comparison of the eigenvector values (for eigenvector # 1)
print(np.allclose(svd_eigenvectors[:, 0], np.real(eigenvectors[:, 0])))

# Computational cost comparison
if False:     # we set this to True if we want to perform the computation cost comparison

    def covariant_matrix_approach ():
         C = np.dot(residuals.T,residuals)
         eigenvalues, eigenvectors = np.linalg.eig(C)

    def svd_approach ():
        (u, w, vt) = np.linalg.svd(residuals)
        svd_eigenvectors = vt.T

    time_cov = timeit.timeit(covariant_matrix_approach,number=1)
    print("Execution time with the Covariant matrix method: {0} s".format(time_cov))

    time_svd = timeit.timeit(svd_approach, number=1)
    print("Execution time with the SVD method: {0} s".format(time_svd))

# Part f)-----------------------------------------------------------------------------------------
# Comparing condition numbers
cov_condition_number = np.linalg.cond(C)
svd_condition_number = np.linalg.cond(residuals)

print("Condition number of covariance matrix: ",cov_condition_number)
print("Condition number of residuals matrix: ", svd_condition_number)

# Part g)-----------------------------------------------------------------------------------------
# Recostructing spectra
# Number of principal components
Nc = 5

# Computing the coefficients
coefficients = np.dot(residuals, eigenvectors)


# Evaluting the approximate spectra
approx_spectra = flux_integrals[:, np.newaxis] * (mean_flux[:, np.newaxis] + np.dot(coefficients[:, :Nc], eigenvectors[:, :Nc].T))


# Plotting data 
plt.figure(figsize=(9,6))
for i in range(1): # we do this for galaxy # 1
    plt.plot(logwave, flux[i],  label = f'Galaxy # {i+1}')
    plt.plot(logwave, approx_spectra[i],  label = f'Galaxy # {i+1} - approx')

plt.xlabel(r"$\log_{10}(\lambda)$")
plt.ylabel(r"Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")


plt.legend()
plt.grid(True, alpha = 0.6)
plt.savefig("pca_data_approx.png")

plt.close()


# Part h)-----------------------------------------------------------------------------------------
# Plotting c0 vs c1
plt.figure(figsize=(9,6))
plt.plot(coefficients[:,1], coefficients[:,0], ".")

plt.xlabel(r"$c_1$")
plt.ylabel(r"$c_0$")


plt.grid(True, alpha = 0.6)
plt.savefig("pca_c0_c1.png")

plt.close()

# Plotting c0 vs c2
plt.figure(figsize=(9,6))
plt.plot(coefficients[:,2], coefficients[:,0], ".")

plt.xlabel(r"$c_2$")
plt.ylabel(r"$c_0$")


plt.grid(True, alpha = 0.6)
plt.savefig("pca_c0_c2.png")

plt.close()

# Part i)-----------------------------------------------------------------------------------------
# Computing rms for various Nc
rms_values=[]

for Nc in np.arange(1,21):
    sq_res=0
    
    approx_spectra = flux_integrals[:, np.newaxis] * (mean_flux[:, np.newaxis] + np.dot(coefficients[:, :Nc], eigenvectors[:, :Nc].T))
    

    for i in np.arange(len(flux[0])): # we do this for galaxy # 1
        sq_res+=np.square(flux[0][i]-approx_spectra[0][i])
    
    rms_values.append(np.sqrt(sq_res/len(flux[0])))
   

# Plotting RMS vs Nc
plt.figure(figsize=(9,6))
plt.plot(np.arange(1,21), rms_values)

plt.xlabel(r"$N_c$")
plt.ylabel(r"$RMS$")


plt.grid(True, alpha = 0.6)
plt.savefig("pca_rms.png")

plt.close()

# Value of RMS for Nc = 20
print("RMS for Nc = 20: ", rms_values[-1])