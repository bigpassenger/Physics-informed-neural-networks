import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
2D Diffusion Simulation with Initial Visualization

This code sets up a grid to simulate the diffusion of two quantities (u and v) over a 2D domain.
The equations being solved are likely related to fluid dynamics or reaction-diffusion systems.

Key components:
- Spatial grid: 2x2 square with 51x51 points
- Time parameters: 500 time steps with Δt = 0.001
- Diffusion coefficient: ν = 0.1
- Initial conditions:
    - Base value of 1 everywhere
    - Central square region (from 0.75 to 1.25 in both directions) set to 5
- Visualizes initial conditions using contour plots


The arrays uf and vf are intended to store the time evolution but are currently just initialized
with the initial conditions repeated across all time steps.

Equations (intended):
    ∂u/∂t = ν(∂²u/∂x² + ∂²u/∂y²)
    ∂v/∂t = ν(∂²v/∂x² + ∂²v/∂y²)

Boundary conditions: Fixed value (1) at all boundaries (implied by initialization)
"""
nt = 500
nx = 51
ny = 51


nu = 0.1
dt = .001


dx = 2/(nx - 1)
dy = 2/(ny - 1)

x = np.linspace(0,2, nx)
y = np.linspace(0,2, ny)

comb = np.zeros((nx,ny)) # grid

# next time step
u = np.zeros((nx, ny)) # placeholder
v = np.zeros((nx, ny))

# previous time step
un = np.zeros((nx, ny))
vn = np.zeros((nx, ny))

# current time step
uf = np.zeros((nt,nx, ny))
vf = np.zeros((nt,nx, ny))


# initial condition
u = np.ones((nx, ny)) 
v = np.ones((nx, ny))
uf = np.ones((nt,nx, ny))
vf = np.ones((nt,nx, ny))

u[int(0.75/dy):int(1.25/dy+1), int(0.75/dy):int(1.25/dy+1)] = 5 # place of a high speed
v[int(0.75/dy):int(1.25/dy+1), int(0.75/dy):int(1.25/dy+1)] = 5 # place of a high speed

uf[int(0.75/dy):int(1.25/dy+1), int(0.75/dy):int(1.25/dy+1)] = 5 # place of a high speed
vf[int(0.75/dy):int(1.25/dy+1), int(0.75/dy):int(1.25/dy+1)]= 5 # place of a high speed


# plot u,v
X, Y = np.meshgrid(x,y)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X,Y,u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.xlabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("U scale")
# plt.show()
X, Y = np.meshgrid(x,y)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X,Y,v[:], cmap='jet')
plt.title("v solution")
plt.xlabel("X")
plt.xlabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("v scale")
# plt.show()



# Solving by updating every time step

for n in range(1, nt):
    un = u.copy()
    vn = v.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i,j] = (un[i, j] -(un[i, j] * dt / dx * (un[i, j] - un[i-1, j])) -vn[i, j] * dt / dy * (un[i, j] - un[i, j-1])) + (nu*dt/(dx**2))*(un[i+1,j]-2*un[i,j]+un[i-1,j])+(nu*dt/(dx**2))*(un[i,j-1]-2*un[i,j]+un[i,j+1])
            v[i,j] = (vn[i, j] -(un[i, j] * dt / dx * (vn[i, j] - vn[i-1, j]))-vn[i, j] * dt / dy * (vn[i, j] - vn[i, j-1])) + (nu*dt/(dx**2))*(vn[i+1,j]-2*vn[i,j]+vn[i-1,j])+(nu*dt/(dx**2))*(vn[i,j-1]-2*vn[i,j]+vn[i,j+1])
            uf[n,i,j] = u[i,j]
            vf[n,i,j] = v[i,j]
    u[:,0 ] = 1
    u[:,-1] = 1
    u[0,: ] = 1
    u[-1,:] = 1
    v[:,0 ] = 1
    v[:,-1] = 1
    v[0,: ] = 1
    v[-1,:] = 1  


#############################
X, Y = np.meshgrid(x,y)

plt.figure(figsize=(8,6))
contour = plt.contourf(X,Y,u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale")
plt.show()
#############################
X, Y = np.meshgrid(x,y)

plt.figure(figsize=(8,6))
contour = plt.contourf(X,Y,v[:], cmap='jet')
plt.title("v solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("v scale")
plt.show()
#############################
X, Y = np.meshgrid(x,y)

#set the time as you like
u = uf[30,:,:]

plt.figure(figsize=(8,6))
contour = plt.contourf(X,Y,u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale from uf[30,:,:]")
plt.show()