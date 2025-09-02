import numpy as np
from matplotlib import pyplot as plt

# Parameters
m = 2       # mass (kg)
k = 10      # spring constant (N/m)
gamma = 1   # damping coefficient
total_sim_time = 10
dt = 0.01   # time step
n_steps = int(total_sim_time/dt)

# Initialize arrays
x = np.zeros(n_steps)  # position
v = np.zeros(n_steps)  # velocity

# Initial conditions
x[0] = 1.0   # initial displacement
v[0] = 0.0   # initial velocity

# Time loop (Euler method)
for t in range(n_steps-1):
    a = -(k/m) * x[t] - gamma * v[t]  # acceleration
    v[t+1] = v[t] + dt * a
    x[t+1] = x[t] + dt * v[t]

# Time vector for plotting
t_vector = np.linspace(0, total_sim_time, n_steps)

# Plot
plt.plot(t_vector, x)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Damped Mass-Spring System Simulation")
plt.show()
