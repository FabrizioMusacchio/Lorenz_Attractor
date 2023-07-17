"""
A script to solve the Lorenz system of ODEs using the Runge-Kutta method.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: Oct 03, 2020
"""
# %% IMPORTS
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %% FUNCTIONS
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def rk1(state, t, h, f, *args):
    return state + h * f(t, state, *args)

def rk2(state, t, h, f, *args):
    k1 = h * f(t, state, *args)
    return state + h * f(t + h/2, state + k1/2, *args)

def rk3(state, t, h, f, *args):
    k1 = h * f(t, state, *args)
    k2 = h * f(t + h/2, state + k1/2, *args)
    return state + (k1 + 4*k2)/6

def rk4(state, t, h, f, *args):
    k1 = h * f(t, state, *args)
    k2 = h * f(t + h/2, state + k1/2, *args)
    k3 = h * f(t + h/2, state + k2/2, *args)
    k4 = h * f(t + h, state + k3, *args)
    return state + (k1 + 2*k2 + 2*k3 + k4)/6
# %% MAIN
# Define parameters:
t_start = 0.0
t_end = 40.0
N = 2000  # number of steps
h = (t_end - t_start) / N  # step size
t_values = np.linspace(t_start, t_end, N+1)

# Define initial conditions:
state_0 = np.array([1.0, 1.0, 1.0])  # initial x, y, z

# Define Lorenz parameters:
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Solve the ODE using different methods:
state_values_rk = dict()
state_values_solve_ivp = []

for method, name in [(rk1, 'RK1'), (rk2, 'RK2'), (rk3, 'RK3'), (rk4, 'RK4')]:
    state_values = np.zeros((N+1, 3))
    state_values[0] = state_0
    for i in range(N):
        state_values[i+1] = method(state_values[i], t_values[i], h, lorenz, sigma, rho, beta)
    state_values_rk[name] = state_values

# Solve the ODE using solve_ivp:
sol = solve_ivp(lorenz, [t_start, t_end], state_0, args=(sigma, rho, beta), t_eval=t_values)
state_values_solve_ivp = sol.y.T

# Plot the results:
fig = plt.figure(figsize=(14, 12))
# Plot the RK solutions
for idx, (name, state_values) in enumerate(state_values_rk.items(), start=1):
    ax = fig.add_subplot(3, 2, idx, projection='3d')
    ax.plot(state_values[:, 0], state_values[:, 1], state_values[:, 2])
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
# Plot the solve_ivp solution:
ax = fig.add_subplot(3, 2, 5, projection='3d')
ax.plot(state_values_solve_ivp[:, 0], state_values_solve_ivp[:, 1], state_values_solve_ivp[:, 2])
ax.set_title('solve_ivp (RK 4/5)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.savefig('runge_kutta_method_lorenz_system_3D.png', dpi=200)
plt.show()

# Check conservation laws:
fig = plt.figure(figsize=(12,8))
for idx, (name, state_values) in enumerate(state_values_rk.items(), start=1):
    E = state_values[:, 0]**2 / 2 + state_values[:, 1]**2 + state_values[:, 2]**2 - state_values[:, 2]
    ax = fig.add_subplot(3, 2, idx)
    ax.plot(t_values, E)
    ax.set_title(name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Lorenz Energy')
# Plot the energy for solve_ivp:
E_solve_ivp = state_values_solve_ivp[:, 0]**2 / 2 + state_values_solve_ivp[:, 1]**2 + state_values_solve_ivp[:, 2]**2 - state_values_solve_ivp[:, 2]
ax = fig.add_subplot(3, 2, 5)
ax.plot(t_values, E_solve_ivp, label='solve_ivp')
ax.set_title('solve_ivp (RK 4/5)')
ax.set_xlabel('Time')
ax.set_ylabel('Lorenz Energy')
plt.tight_layout()
plt.savefig('runge_kutta_method_lorenz_system_energy.png', dpi=200)
plt.show()
# %% END
