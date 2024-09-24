# PS3 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 2

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# Constants -----------------------------------------------------
# Numbers of atoms
N_Bi209 = 0                         # number of Bi209
N_Pb = 0                            # number of Pb209
N_Tl = 0                            # number of Tl209
N_Bi213 = 10000                     # number of Bi213

# Half lives
tau_Pb = 3.3*60                     # half life of Pb209 in sec
tau_Tl = 2.2*60                     # half life of Tl209 in sec
tau_Bi213 = 46*60                   # half life of Bi213 in sec

# Temporal interval
tmax = 20000                        # total time in sec
dt = 1.                             # time step in sec

# Probabilities of decay
p_Pb = 1 - 2**(-dt/tau_Pb)          # probability of decay in one step for Pb209  
p_Tl = 1 - 2**(-dt/tau_Tl)          # probability of decay in one step for Tl209  
p_Bi213 = 1 - 2**(-dt/tau_Bi213)    # probability of decay in one step for Bi213
p_Bi_Tl = 0.00209                   # probability of decay Bi213 -> Tl

#-----------------------------------------------------------------  

# Points to plot
time = np.arange(0., tmax, dt)
Bi209 = []
Pb = []
Tl = []
Bi213 = []

# Loop for generating the data
for i in time:
    Bi209.append(N_Bi209)
    Pb.append(N_Pb)
    Tl.append(N_Tl)
    Bi213.append(N_Bi213)
    
    '''
    Methodology: instructions of Exercise 10.2, which is based on Example 10.1. The only difference is in the loop
                 which simulates the decay Bi213 -> Pb OR Tl. The decay Bi213 -> Tl, which can occur with 
                 probability of 0.0209, can be simulated by using the same approach, namely by generating a random 
                 number between 0 and 1 and compare it to such value. The decay Bi213 -> Pb occurs otherwise.
    '''
    # DECAY: Pb -> Bi209 
    decay = 0
    for k in range(N_Pb):
        if np.random.random() < p_Pb:
            decay += 1

    N_Pb -= decay
    N_Bi209 += decay        

    # DECAY: Tl -> Pb 
    decay = 0
    for k in range(N_Tl):
        if np.random.random() < p_Tl:
            decay += 1

    N_Tl -= decay
    N_Pb += decay     

    # DECAY: Bi213 -> Pb OR Tl 
    decay_to_Tl = 0
    decay_to_Pb = 0
    for k in range(N_Bi213):
        if np.random.random() < p_Bi213:
            if np.random.random() < p_Bi_Tl:
                decay_to_Tl += 1
            else:
                decay_to_Pb += 1    

    N_Bi213 -= decay_to_Tl + decay_to_Pb
    N_Pb += decay_to_Pb   
    N_Tl += decay_to_Tl


# Plotting of results
plt.figure(figsize=(10, 6))
plt.plot(time, Bi209, label='Bi209', color= 'blue')
plt.plot(time, Pb, label='Pb209', color= 'green')
plt.plot(time, Tl, label='Tl209', color= 'red',)
plt.plot(time, Bi213, label='Bi213', color= 'black',)

plt.xlabel("Time (s)")
plt.ylabel("Number of atoms")

plt.legend()
plt.grid(True)

plt.savefig("prob_2.png")
