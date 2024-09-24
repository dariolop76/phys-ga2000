# PS3 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 1

# imported packages
import timeit
import numpy as np
import matplotlib.pyplot as plt

setup_code = '''

import numpy as np

# Matrix multiplication using for loop
def matrix_product_loop (N):

    # we create two NxN matrices, A and B, with random entries uniformly distributed between 0 and 1
    A = np.random.uniform(0.,1., size = (N,N)).astype(np.float32)
    B = np.random.uniform(0.,1., size = (N,N)).astype(np.float32)

    # we create an NxN matrix, C, with all entries set to zero
    C = np.zeros([N,N], dtype = np.float32)

    # we perform the matrix multiplication A x B = C using three nested for loops
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i,k]*B[k,j]

# Matrix multiplication using the NumPy dot function 
def matrix_product_dot (N):

    # we create two NxN matrices, A and B, with random entries uniformly distributed between 0 and 1
    A = np.random.uniform(0.,1., size = (N,N)).astype(np.float32)
    B = np.random.uniform(0.,1., size = (N,N)).astype(np.float32)

    # we perform the matrix multiplication A x B = C using the NumPy dot function
    C = np.dot(A, B)
'''


N_values = [10 + i for i in range(0, 110, 2)]       # Matrix size values
times_loop = []                                     # array which will contain the duration of for loop for each N
times_dot = []                                      # array which will contain the duration of Numpy dot for each N

for N in N_values:
    code_with_loop = f'matrix_product_loop({N})'
    code_with_dot = f'matrix_product_dot({N})'

    time_with_loop = timeit.timeit(code_with_loop, setup = setup_code, number = 3) 
    time_with_dot = timeit.timeit(code_with_dot, setup = setup_code, number = 3)   

    times_loop.append(time_with_loop)
    times_dot.append(time_with_dot)

# Behaviour of computation time                  
a = np.mean(times_loop) / np.mean(np.array(N_values)**3)    # a allows to compare the curve N^3 to our data
N_behaviour = np.array(N_values)**3 * a

# Plotting of results
fig ,(ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(N_values, times_loop, label='For Loop', color= 'blue')
ax1.plot(N_values, N_behaviour, label='N^3', color= 'red')
ax2.plot(N_values, times_dot, label='NumPy Dot', color= 'green')

ax1.set_ylabel('Time (s)')

ax2.set_xlabel('Matrix size N ')
ax2.set_ylabel('Time (s)')
#ax2.set_ylim(0.,0.05)

ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)

plt.savefig("prob_1.png")