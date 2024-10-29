# PS7 -- Dario Loprete -- dl5521@nyu.edu
# Problem 2

# imported packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Test function
def f(x):
    return (x-0.3)**2*np.exp(x)

# Golden Section Search (GSS)
def gss (f, a, b, c):
    gsection = (3. - np.sqrt(5)) / 2
    if((b - a) > (c - b)):
        x = b
        b = b - gsection * (b - a)
    else:
        x = b + gsection * (c - b)
            
    if(f(b) < f(x)):
        c = x
    else:
        a = b
        b = x 
    return a,b,c
    
# Brent's 1D minimization method
def brent (f, a, b, c, tol = 1e-8):

    condition = np.abs(a-c) > tol           # we stop the method when the bracketing interval is smaller than some tolerance
    step = 0                                # this is the distance moved on the step before last in the SPI (Successive Parabolic Interpolation)
    i = 0
    max = 200                               # maximum number of iteration

    while(condition) and (i < max):
        
        fa = f(a)
        fb = f(b)
        fc = f(c)
        denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
        numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)

        if denom < 1.e-15:                  # denominator is zero -> do GSS
            a,b,c = gss(f,a,b,c)
        
        else:                               # try SPI
            step_old = step
            step = - 0.5 * numer / denom
            x = b + step
      
            if (x < a or x > c) or (np.abs(step) > 0.5*np.abs(step_old)): # SPI not well behaved -> do GGS
                a,b,c = gss(f,a,b,c)    
            else:                                                         # else continue with SPI
                if x < b:
                    c = b
                else:
                    a = b   
        i += 1
        condition = np.abs(a-c) > tol

    return 0.5*(a+c)    # when tolerance is reached or number of iterations has reached maximum, returns minimum point


# Plot of function
x_values = np.linspace(-3,3)
plt.figure(figsize=(9,6))

plt.xlabel("x")
plt.ylabel("f(x)")

plt.plot(x_values, f(x_values))
plt.grid(True, alpha = 0.6)
plt.savefig("prob_2.png")

plt.close()

# from plot we see that the minimum is surely between -2 and 3, so we choose a=-2, b=-1, c=3
a = -2
b = -1
c = 3
min = brent(f,a,b,c)
min_scipy = opt.brent(f, brack=(a,b,c))

print("Min (brent): ", min)
print("Min (scipy): ", min_scipy)
print("Difference: ", np.abs(min - min_scipy))