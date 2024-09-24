# PS3 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 3

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# Constants -----------------------------------------------------
# Numbers of atoms
N_Tl = 1000                         # number of Tl208

# Half lives
tau_Tl = 3.053*60                   # half life of Tl208 in sec

# Temporal interval
tmax = 1000                         # total time in sec
dt = 1.                             # time step in sec

#-----------------------------------------------------------------  

time = np.arange(0., tmax, dt)


# Transformation method
def non_uniform(x):
    return -tau_Tl/(np.log(2)) * np.log(1-x) 

# Generation of data
'''
 we generate 1000 decay times following a non uniform distribution and sort them (from smaller to bigger)
'''
time_decays = np.sort(np.array([non_uniform(np.random.random()) for i in range(1000)]))  
'''
 we count the number of atoms which have not yet decayed by comparing the decay time with the atom's half life:
 if the former is larger than the latter, the corresponding atom has not yet decayed
'''
atoms_not_decayed = np.array([np.sum(time_decays > t) for t in time])

# Plotting of results
plt.figure(figsize=(10, 6))
plt.plot(time, atoms_not_decayed, color= 'red',)

plt.xlabel("Time (s)")
plt.ylabel("Number of atoms not yet decayed")

plt.grid(True)

plt.savefig("prob_3.png")
