# PS10 -- Dario Loprete -- dl5521@nyu.edu
# Problem 2

# imported packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import scipy.fft


# Part a) -----------------------------------------------------------------------------------
# Parameters
L = 1e-8         # m  - Length of box
h = 1e-18        # s - time-step
Nx = 1000        # number of spatial slices
a = L/Nx         # m - space-step
Nt = 3000         # number of time slices

M = 9.109e-31    # kg - eletron mass
hbar = 1.054e-34 # J s^-1


# Space grid
x = np.arange(a,L,a)

# Initial condition: parameters
x0 = 0.5*L      # m 
sigma = 1e-10   # m 
k = 5e+10       # m^-1 


psi_init = np.exp(-(x-x0)**2/(2*sigma**2))*np.exp(1j*k*x)

psi_init_real = np.real(psi_init)
psi_init_imag = np.imag(psi_init)

b_real = scipy.fft.dst(psi_init_real)
b_imag = scipy.fft.dst(psi_init_imag)

# Part b) -----------------------------------------------------------------------------------
# Solving for psi(x,t) using expression

def psi_1 (Nx, h, Nt, b_real, b_imag):
    solution = []
    
    for t in range(Nt):

        psi_n = []
        for n in range(1,Nx):
            k_values = np.arange(1,Nx)
        
            cos_k = np.cos((np.pi**2*hbar)/(2*M*L**2) * k_values**2 * t*h)
            sin_k = np.sin((np.pi**2*hbar)/(2*M*L**2) * k_values**2 * t*h)

            sin_kn = np.sin((np.pi*k_values*n)/(Nx))

            tmp = 1/Nx * np.sum((b_real*cos_k + b_imag*sin_k)*sin_kn)

            psi_n.append(tmp)

        solution.append(psi_n)

    return solution


solution_1 = psi_1 (Nx, h, Nt, b_real, b_imag)

# Solving for psi(x,t) using inverse discrete sine transform

def psi_2 (Nx, h, Nt, b_real, b_imag):
    solution = []

    for t in range(Nt):

        k_values = np.arange(1,Nx)
        
        cos_k = np.cos((np.pi**2*hbar)/(2*M*L**2) * k_values**2 * t*h)
        sin_k = np.sin((np.pi**2*hbar)/(2*M*L**2) * k_values**2 * t*h)

        tmp = b_real*cos_k + b_imag*sin_k

        psi = scipy.fft.idst(tmp)

        solution.append(psi)

    return solution

solution_2 = psi_2 (Nx, h, Nt, b_real, b_imag)

times = [0,250,400,1200, 1550, 2100]

# Plotting solution 1
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, ax in enumerate(axes.flat):
        ax.plot(x, solution_1[times[i]], color = 'red')

        ax.set_title(f't = {times[i]*h:.1e} s')
        ax.set_xlabel('x (m)')
        ax.set_ylabel(r'Re($\psi$)')
        ax.set_ylim(-1,1)

plt.tight_layout()        
plt.savefig("prob_2_solution1.png")
plt.close()

# Plotting solution 2
fig, axes = plt.subplots(2, 3, figsize=(12, 8))


for i, ax in enumerate(axes.flat):
        ax.plot(x, solution_2[times[i]], color = 'green')

        ax.set_title(f't = {times[i]*h:.1e} s')
        ax.set_xlabel('x (m)')
        ax.set_ylabel(r'Re($\psi$)')
        ax.set_ylim(-1,1)

plt.tight_layout()            
plt.savefig("prob_2_solution2.png")
plt.close()


