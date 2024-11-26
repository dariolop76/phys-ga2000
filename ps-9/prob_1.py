# PS9 -- Dario Loprete -- dl5521@nyu.edu
# Problem 1

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# Function f - harmonic oscillator
def f_h(r,t, omega):
    x = r[0]
    v = r[1]
    f_x = v
    f_v = -omega**2*x
    return np.array([f_x,f_v])

# Function f - harmonic oscillator
def f_an(r,t, omega):
    x = r[0]
    v = r[1]
    f_x = v
    f_v = -omega**2*x**3
    return np.array([f_x,f_v])

# Function f - van der Pol oscillator
def f_vdp(r,t, omega, mu):
    x = r[0]
    v = r[1]
    f_x = v
    f_v = -omega**2*x + mu*(1-x**2)*v
    return np.array([f_x,f_v])

# Solver RK4 
def RK4(t_points, t_step, init, func, *args):

    r = init
    x_points = []
    v_points = []

    for t in t_points:
        x_points.append(r[0])
        v_points.append(r[1])

        k1 = t_step*func(r,t, *args)
        k2 = t_step*func(r + 0.5*k1, t + 0.5*t_step, *args)
        k3 = t_step*func(r + 0.5*k2, t + 0.5*t_step, *args)
        k4 = t_step*func(r + k3, t + t_step, *args)
        r += 1./6 * (k1 + 2*k2 + 2*k3 +k4)     
    
    return np.array(x_points), np.array(v_points)     

# Part a) ----------------------------------------------------------------------------------
# Solving the equation using RK4

# Parameters
t_i = 0
t_f = 50
N = 1000
t_step = (t_f - t_i)/N

# Points on t axis
t_points = np.arange(t_i, t_f, t_step)

# Initial conditions
x_0 = 1.
v_0 = 0.
init = np.array([x_0, v_0])

# Solution
x, v = RK4(t_points, t_step, init, f_h, 1.)

# Plotting of solution
plt.figure(figsize=(8,5))

plt.plot(t_points, x)

plt.xlabel("t (a.u.)")
plt.ylabel("x(t) (a.u.)")

plt.savefig("prob_1_a.png")
plt.close()

# Part b) ----------------------------------------------------------------------------------
# Initial conditions
x_0 = 2.
v_0 = 0.
init = np.array([x_0, v_0])

# Solution
x_b, v_b = RK4(t_points, t_step, init, f_h, 1.)

# Plotting of solution
plt.figure(figsize=(8,5))

plt.plot(t_points, x, label = "amp = 1")
plt.plot(t_points, x_b, label = "amp = 2")

plt.xlabel("t (a.u.)")
plt.ylabel("x(t) (a.u.)")

plt.legend()
plt.savefig("prob_1_b.png")
plt.close()

# Part c) ----------------------------------------------------------------------------------

# Solutions with different amplitudes
x_c1, v_c1 = RK4(t_points, t_step, np.array([1, 0.]), f_an, 1.)
x_c2, v_c2 = RK4(t_points, t_step, np.array([1.1, 0.]), f_an, 1.)
x_c3, v_c3 = RK4(t_points, t_step, np.array([1.3, 0.]), f_an, 1.)

# Plotting of solution
plt.figure(figsize=(8,5))

plt.plot(t_points, x_c1, label = "amp = 1")
plt.plot(t_points, x_c2, label = "amp = 1.1")
plt.plot(t_points, x_c3, label = "amp = 1.3")

plt.xlabel("t (a.u.)")
plt.ylabel("x(t) (a.u.)")

plt.legend()
plt.savefig("prob_1_c.png")
plt.close()

# Part d) ----------------------------------------------------------------------------------
# Plotting of phase space
plt.figure(figsize=(8,5))

plt.plot(x_c1, v_c1, label = "amp = 1")
plt.plot(x_c2, v_c2, label = "amp = 1.1")
plt.plot(x_c3, v_c3, label = "amp = 1.3")

plt.xlabel("x (a.u.)")
plt.ylabel("v (a.u.)")

plt.legend()
plt.savefig("prob_1_d.png")
plt.close()

# Part e) ----------------------------------------------------------------------------------

# Parameters
t_i = 0
t_f = 20
N = 5000
t_step = (t_f - t_i)/N

# Points on t axis
t_points = np.arange(t_i, t_f, t_step)


# Solutions  with different values of mu
# Initial conditions
x_0 = 1.
v_0 = 0.
init = np.array([x_0, v_0])
x_e1, v_e1 = RK4(t_points, t_step, init, f_vdp, 1., 1.)

init = np.array([x_0, v_0])
x_e2, v_e2 = RK4(t_points, t_step, init, f_vdp, 1., 2.)

init = np.array([x_0, v_0])
x_e3, v_e3 = RK4(t_points, t_step, init, f_vdp, 1., 4.)


# Plotting of phase space diagrams
plt.figure(figsize=(8,5))


plt.plot(x_e1, v_e1, label = "mu = 1")
plt.plot(x_e2, v_e2, label = "mu = 2")
plt.plot(x_e3, v_e3, label = "mu = 4")

plt.plot(x_0, v_0, '.', markersize = 5, color = 'red', label = "init. cond.")

plt.xlabel("x (a.u.)")
plt.ylabel("v (a.u.)")

plt.legend()
plt.savefig("prob_1_e.png")
plt.close()