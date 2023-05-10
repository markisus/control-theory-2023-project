from sindy import eval_theta, eval_theta_single, eval_elements
import numpy as np
from scipy.sparse import csr_matrix
import time

class SindyDynamics:
  def __init__(self, sindy_model):
    self.clf = sindy_model['clf']
    self.element_ids = sindy_model['elements']
    self.feature_names = sindy_model['feature_names']

  def get_dstate_dt(self, state, action):
    # return np.zeros(state.shape)
    # start = time.monotonic()
    elements = eval_elements(self.element_ids, state.T, action.T)
    # print(f"eval elements time {time.monotonic() - start}")

    start = time.monotonic()
    W = self.clf.coef_
    Ws = self.clf.sparse_coef_

    nzrows, nzcols = Ws.nonzero()
    col_activations = np.zeros((Ws.shape[1],))
    for col in nzcols:
        col_activations[col] = 1
    # print(np.count_nonzero(col_activations))

    start = time.monotonic()
    basis = eval_theta_single(self.feature_names, elements, col_activations).T
    # print(f"basis time {time.monotonic() - start}")

    # print("ws type", type(Ws))
    # print(f"Sparse coeffs {Ws}")
    result = Ws @ csr_matrix(basis)
    # print(f"matmul time {time.monotonic() - start}")

    return result.toarray()

class SindySimulation:
  def __init__(self, dt, sindy_dynamics, init_state):
    self.dynamics = sindy_dynamics
    self.state = init_state.copy()
    self.action = np.zeros((2,1))
    self.dt = dt

  def reset(self, init_state):
    self.state = init_state.copy()

  def step(self):
    dynamics = self.dynamics
    state = self.state
    action = self.action
    dt = self.dt

    use_rk = True
    if use_rk:
      # 4th order Runge Kutta
      k1 = dynamics.get_dstate_dt(state, action)
      k2 = dynamics.get_dstate_dt(state + dt/2 * k1, action)
      k3 = dynamics.get_dstate_dt(state + dt/2 * k2, action)
      k4 = dynamics.get_dstate_dt(state + dt * k3, action)
      state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    else:
      state += dt * dynamics.get_dstate_dt(state, action)

    self.state = state
