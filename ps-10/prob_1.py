# PS10 -- Dario Loprete -- dl5521@nyu.edu
# Problem 1

# imported packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import banded


# Part a) -----------------------------------------------------------------------------------
# Crank-Nicolson solver
def CN_solver (x, init, a, Nt):

    solution = []
    psi = init

    # Parameters for matrix A and B
    a1 = 1 +1j*h*hbar/(2*M*a**2)
    a2 = -1j*h*hbar/(4*M*a**2)
    b1 = 1-1j*h*hbar/(2*M*a**2)
    b2 = 1j*h*hbar/(4*M*a**2)

    # Matrix A - using the representation in banded.py
    arr1 = np.full(Nx, a1)
    arr2 = np.full(Nx, a2)
    A = np.stack([arr2, arr1, arr2])

    # Solving for psi at next step
    for t in range(Nt):

        # Creating vector v = B psi
        v = np.zeros(Nx-1, dtype=complex)

        for i in range(len(v)):
            if i == 0:
                v[i] = b1*psi[i] + b2*psi[i+1]
            elif i == len(v)-1:
                v[i] = b1*psi[i] + b2*psi[i-1]
            else:
                v[i] = b1*psi[i] + b2*(psi[i-1] + psi[i+1])

        psi_new = banded.banded(A, v, 1, 1)    
        psi = psi_new    
        solution.append(psi)
    return np.array(solution)


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

# Solution
psi_init = np.exp(-(x-x0)**2/(2*sigma**2))*np.exp(1j*k*x)

solution = CN_solver(x, psi_init, a, Nt)

# Part b) -----------------------------------------------------------------------------------
# Plotting snapshots
# Plotting solution 1
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

times = [0,300,500,1000, 1500, 2000]

for i, ax in enumerate(axes.flat):
        ax.plot(x, np.real(solution[times[i]]))

        ax.set_title(f't = {times[i]*h:.1e} s')
        ax.set_xlabel('x (m)')
        ax.set_ylabel(r'Re($\psi$)')
        ax.set_ylim(-1,1)

plt.tight_layout()        
plt.savefig("prob_1_solution.png")
plt.close()

# Displaying animation
if False:
    fig, ax = plt.subplots()
    ax.set(xlabel='x (m)', ylabel=r'Re($\psi$)')
    line = ax.plot(x, np.real(psi_init))[0]

    def update(frame):

        line.set_data(x,np.real(solution[frame]))
        return line

    ani = animation.FuncAnimation(fig=fig, func=update, frames=Nt, interval=1)

    plt.show()