import numpy as np
from matplotlib import pyplot

# --- Problem Setup ---
length = 10          # Length of the rod (meters, arbitrary units)
k = 0.89             # Thermal diffusivity constant
# Boundary conditions: fixed temperatures at both ends of the rod
temp_left = 100      # Left end temperature
temp_right = 200     # Right end temperature

total_sim_time = 10  # Total simulation time (seconds)

# --- Spatial and Temporal Discretization ---
dx = 0.1
x_vector = np.linspace(0, length, int(length/dx))  # Positions along the rod

dt = 0.0001
t_vector = np.linspace(0, total_sim_time, int(total_sim_time/dt))  # Time steps

# u will store the temperature field
# shape = (number of time steps, number of spatial points)
u = np.zeros([len(t_vector), len(x_vector)])

# --- Applying boundary conditions ---
u[:, 0] = temp_left     # Keep left boundary fixed at 100
u[:, -1] = temp_right   # Keep right boundary fixed at 200

# Plot the initial temperature distribution (all zeros except boundaries)
pyplot.plot(x_vector, u[0])
pyplot.ylabel("Temperature")
pyplot.xlabel("Position along the rod")
pyplot.title("Initial Temperature Distribution")
pyplot.show()

# --- Solving the 1D Heat Equation ---
# Explicit finite difference method:
# u[t+1, x] = u[t, x] + alpha * (u[t, x+1] - 2*u[t, x] + u[t, x-1])
# where alpha = k * dt / dx^2
for t in range(1, len(t_vector)-1):
    for x in range(1, len(x_vector)-1):
        u[t+1, x] = ((k * (dt / dx**2)) *
                     (u[t, x+1] - 2*u[t, x] + u[t, x-1])) + u[t, x]

# Plot again after running the simulation (still shows initial state)
pyplot.plot(x_vector, u[0])
pyplot.ylabel("Temperature")
pyplot.xlabel("Position along the rod")
pyplot.title("Temperature Distribution at t = 0")
pyplot.show()

# Check the size of the solution matrix (time x space)
print(len(u))

# Plot the temperature distribution at a later time step (close to steady-state)
pyplot.plot(x_vector, u[99999])
pyplot.ylabel("Temperature")
pyplot.xlabel("Position along the rod")
pyplot.title("Temperature Distribution at t ≈ 10s")
pyplot.show()
