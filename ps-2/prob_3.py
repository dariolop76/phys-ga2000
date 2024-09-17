# PS2 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 3

# imported packages
import timeit


setup_code = '''

# imported packages
import numpy as np

# chosen value for L
L = 100

# WITH FOR LOOP
# definition of function which returns the potential due to the atom at position (i,j,k)
# (we forget about the constants which do not appear in the computation of the Madelung constant)

def potential_loop(i,j,k):
    if (i==0 and j==0 and k==0):                            # we can avoid counting the atom at the origin in the sum 
        return 0                                            # by setting this value to zero (even though it is not zero)
    else:    
        if ((i+j+k) % 2 == 0):                              # condition for sodium atoms - positive charge (i+j+k even)
            return np.float32(1./np.sqrt(i**2+j**2+k**2))
    
        elif ((i+j+k) % 2 != 0):                            # condition for chlorine atoms - negative charge (i+j+k odd)
            return np.float32(- 1./np.sqrt(i**2+j**2+k**2))
    

# definition of function that runs a for loop
def madelung_loop(L):
    
    madelung_const= np.float32(0)

    # we use three concatenated for loops to run through each atom of the lattice
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                madelung_const += potential_loop(i,j,k)     # we evaluate the potential due to 
                                                            # the atom at position (i,j,k) and add its contribution
    
    print("Madelung constant using for loop: ", madelung_const)                                                        
    return madelung_const
           

# WITHOUT A FOR LOOP
# definition of function which returns the potential due to the atom at position (i,j,k) - vectorized version
# (we forget about the constants which do not appear in the computation of the Madelung constant)

def potential_no_loop(i,j,k):
    
    r = np.sqrt(i**2 + j**2 + k**2)
    
    r[r == 0] = np.inf                                              # this allows to avoid division by zero (namely, to exclude from the sum 
                                                                    # the potential due to the atom at the origin)

    potentials = np.where((i + j + k) % 2 == 0, 1 / r, -1 / r)      # the discrimination between sodium and chlorine atoms is summarized by 
                                                                    # one line of code using np.where, which also allows to vectorize the function
    return potentials

# definition of function without a for loop
def madelung_no_loop(L):

    indices = np.arange(-L, L + 1)                    # these are the indices that run from -L to +L
    I, J, K = np.meshgrid(indices, indices, indices)  # these are the coordinates for all the atoms in the lattice
    lattice = potential_no_loop(I,J,K)                # this is a 3d array containing the potential at the origin 
                                                      # due to each and every other atom of the lattice
    
    madelung_const = np.sum(lattice)                  # the Madelung constant is given by the sum of such values
    
    print("Madelung constant without using a for loop: ", madelung_const) 
    return madelung_const

'''
 

code_with_loop = 'madelung_loop(L)'
code_without_loop = 'madelung_no_loop(L)'

time_with_loop = timeit.timeit(code_with_loop, setup = setup_code, number = 1)
time_without_loop = timeit.timeit(code_without_loop, setup = setup_code, number = 1)
print("Duration with for loop: ", time_with_loop, " s")
print("Duration without for loop: ", time_without_loop, "s")

