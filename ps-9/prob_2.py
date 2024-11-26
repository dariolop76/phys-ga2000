# PS9 -- Dario Loprete -- dl5521@nyu.edu
# Problem 2

# imported packages
import numpy as np
import matplotlib.pyplot as plt

# Function f - trajectory w/ air resistance
def f(r,t, A):
    x = r[0]
    y = r[1]
    vx = r[2]
    vy = r[3]

    f_x = vx
    f_y = vy
    f_vx = -0.5*np.pi*A* vx *np.sqrt(vx**2 + vy**2)
    f_vy = -1 -0.5*np.pi*A* vy *np.sqrt(vx**2 + vy**2)

    return np.array([f_x, f_y, f_vx, f_vy])

# Solver RK4 
def RK4(t_points, t_step, init, func, *args):

    r = init
    x_points = []
    y_points = []
    vx_points = []
    vy_points = []

    for t in t_points:
        x_points.append(r[0])
        y_points.append(r[1])
        vx_points.append(r[2])
        vy_points.append(r[3])
        

        k1 = t_step*func(r,t, *args)
        k2 = t_step*func(r + 0.5*k1, t + 0.5*t_step, *args)
        k3 = t_step*func(r + 0.5*k2, t + 0.5*t_step, *args)
        k4 = t_step*func(r + k3, t + t_step, *args)
        r += 1./6 * (k1 + 2*k2 + 2*k3 +k4)     
    
    return np.array(x_points), np.array(y_points), np.array(vx_points) , np.array(vy_points)   

# Part b) ----------------------------------------------------------------------------------
# Solving the equation using RK4

# Parameters
t_i = 0
t_f = 20
N = 1000
t_step = (t_f - t_i)/N

T = 1       # s
R = 0.08    # m
rho = 1.22  # kg m^-3
C = 0.47    # dimensionless
v = 100     # m s^-1
g = 9.81    # m s^-2
theta = 30  # °
m = 1       # kg

A = (R**2*rho*C*g*T**2)/m  # dimensionless

# Points on t axis
t_points = np.arange(t_i, t_f, t_step)

# Initial conditions
x_0 = 0.
y_0 = 0.
vx_0 = v*np.cos(theta*(np.pi/180))
vy_0 = v*np.sin(theta*(np.pi/180))
init = np.array([x_0, y_0, vx_0/(T*g), vy_0/(T*g)])

# Solution
x, y, vx, vy = RK4(t_points, t_step, init, f, A)

# Plotting of trajectory
plt.figure(figsize=(8,5))

plt.plot(T**2*g*x, T**2*g*y)

plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.xlim(0,300)
plt.ylim(0,300)

plt.savefig("prob_2_b.png")
plt.close()

# Part b) ----------------------------------------------------------------------------------
# Analysis distance vs masses

t_i = 0
t_f = 50
N = 1000
t_step = (t_f - t_i)/N

#masses = [1.2, 2.4, 4.7] # kg
masses = np.arange(1,10,0.3)
distances = []

x_list = []
y_list = []
vx_list = []
vy_list = []

for m in masses:
    A = (R**2*rho*C*g*T**2)/m 

    init = np.array([x_0, y_0, vx_0/(T*g), vy_0/(T*g)])
    x, y, vx, vy = RK4(t_points, t_step, init, f, A)
    x_list.append(x)
    y_list.append(y)
    vx_list.append(vx)
    vy_list.append(vy)

    for i in range(len(x)-1):
        if y[i] * y[i+1] < 0:
            distances.append(0.5*(x[i] + x[i+1]))

    plt.plot(T**2*g*x, T**2*g*y, label = f'm = {m} kg')

distances = np.array(distances)
  
x_list = np.array(x_list)  
y_list = np.array(y_list)
vx_list = np.array(vx_list)
vy_list = np.array(vy_list)

# Plotting a few trajectories
plt.figure(figsize=(8,5))
plt.plot(T**2*g*x_list[4], T**2*g*y_list[4], label=f'm = {masses[4]:.1f} kg')
plt.plot(T**2*g*x_list[5], T**2*g*y_list[5], label=f'm = {masses[5]:.1f} kg')
plt.plot(T**2*g*x_list[6], T**2*g*y_list[6], label=f'm = {masses[6]:.1f} kg')

plt.xlabel('x(m)')
plt.ylabel('y(m)')

plt.xlim(0,450)
plt.ylim(0,450)

plt.legend()
plt.savefig("prob_2_c1.png")
plt.close()  

# Plotting distance vs mass
plt.figure(figsize=(8,5))

plt.plot(masses, T**2*g*distances, marker = '.')

plt.xlabel("m (kg)")
plt.ylabel("d (m)")


plt.savefig("prob_2_c2.png")
plt.close()
