# PS2 -- Dario Loprete -- dl5521@nyu.edu
# PROBLEM 4

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# dimension of the grid
N = 1500

# number of iterations
it = 120

# definition of function that performs the iterations  
def mandelbrot(N, it):

    # we create a grid of values for the complex number c = x + iy, with -2 <= x <= 2, -2 <= y <= 2
    grid_axis = np.linspace(-2,2, N, dtype=np.float32)      
    re, im = np.meshgrid(grid_axis, grid_axis)

    c = re + im*1j

    # we create a grid of values for the complex number z, which are all set to 0 
    z = np.zeros((N,N), dtype = np.complex64)

    # for the "enthusiastic" plot, we want to save the number of iterations performed right before |z| becomes greater than 2
    # so we create a grid of values, initially set all to zero, which will take into account such number (for each z)
    it_number = np.zeros((N,N), dtype = np.int32)
    
    '''
    Method: in a for loop, we implement the formula z' = z^2 + c. If the magnitude of the new z is less or equal than 2
            we update the previous z to z^2 + c, otherwise we set the new z to a conventional value (here we choose 3) 
            whose magnitude is > 2. After the for loop, we create a grid which contains 0 or 1 according to the magnitude 
            of the corresponding z. This will be then used for the plot of the Mandelbrot set. Additionally, in the loop 
            we save the number of iterations performed right before |z| becomes greater than 2. 
    
    '''
    for i in range(it):
        z = np.where(np.abs(z) <= 2, np.square(z) + c, 3)                      
        it_number[np.logical_and(it_number == 0, np.abs(z) > 2)] = i            # if |z| >2 then update it_number (if not previously
                                                                                # updated, that's what it_number == 0 makes sure of)

    z_to_plot = np.where(np.abs(z) <= 2, 0, 1)                                  # 0 -> |z| <= 2 (black in plot), 1 -> |z| > 2 (white in plot)
    
    return z_to_plot, it_number

# definition of function for the black and white plot
def plot_bw(data):
    fig, ax = plt.subplots(figsize = (3,3), dpi = 400)
    
    ax.set_axis_off()
    ax.set_xlim(0,1100)
    ax.set_ylim(300,1200)
    ax.imshow(data, cmap='grey')
    plt.savefig("mandelbrot_bw.png")

# definition of function for the colored plot
def plot_color(data):
    fig, ax = plt.subplots(figsize = (3,3), dpi = 400)
    
    ax.set_axis_off()
    ax.set_xlim(0,1100)
    ax.set_ylim(300,1200)
    ax.imshow(data, cmap='twilight_shifted')
    plt.savefig("mandelbrot_color.png")   

# generating data
data = mandelbrot(N, it)  
data_bw, data_color = data[0], data[1] 

# plotting of results
plot_bw(data_bw)
plot_color(data_color)  