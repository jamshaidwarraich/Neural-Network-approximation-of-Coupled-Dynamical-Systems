

import numpy as np
import torch
from neurodiffeq.neurodiffeq import safe_diff as diff
from neurodiffeq.ode import solve, solve_system
from neurodiffeq.solvers import Solver1D
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.conditions import IVP

import matplotlib.pyplot as plt
%matplotlib notebook


from neurodiffeq.networks import FCNN # fully-connect neural network
import torch.nn as nn 


# specify the ODE system and its parameters

Salt_mixing = lambda u, v, t : [ diff(u, t) -0.02*v +0.02*u,
                                 diff(v, t) +0.02*v -0.02*u, ]
# specify the initial conditions
init_vals_lv = [
    IVP(t_0=0.0, u_0=0.0),  # 0.0 is the value of u at t_0 = 0.0
    IVP(t_0=0.0, u_0=150.0),  # 150.0 is the value of v at t_0 = 0.0
]

# specify the network to be used to approximate each dependent variable
# the input units and output units default to 1 for FCNN
nets_lv = [
    FCNN(n_input_units=1, n_output_units=1, hidden_units=(64, 64, 128), actv=nn.Tanh),
    FCNN(n_input_units=1, n_output_units=1, hidden_units=(64, 64, 128), actv=nn.Tanh)
]

# solve the ODE system
solution_lv, _ = solve_system(
    ode_system=Salt_mixing, conditions=init_vals_lv, t_min=0.0, t_max=120,
    nets=nets_lv, max_epochs=12000,
    monitor=Monitor1D(t_min=0.0, t_max=120, check_every=100)
)


ts = np.linspace(0, 120, 10)
u_net, v_net = solution_lv(ts, to_numpy=True)
u_ana, v_ana = 75-75*np.exp(-0.04*ts), 75+75*np.exp(-0.04*ts)

plt.figure()
plt.plot(ts, u_net, label='ANN-based solution of $u$')
plt.plot(ts, u_ana, '.', label='Analytical solution of $u$')
plt.plot(ts, v_net, label='ANN-based solution of $v$')
plt.plot(ts, v_ana, '.', label='Analytical solution of $v$')
plt.ylabel('Salt concentration')
plt.xlabel('time')
plt.title('comparing solutions')
plt.legend()
plt.show()