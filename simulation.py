# Local Variables:
# python-indent: 2
# End:

from dynamics import *

class Simulation:
  def __init__(self, dt, dynamics, init_state, tau_sliding, tau_static):
    self.dt = dt
    self.dynamics = dynamics
    self.state = init_state.copy()
    self.prev_state = init_state.copy()
    self.tau_sliding = tau_sliding
    self.tau_static = tau_static
    self.action = np.zeros((2,1))
    self.use_sliding_friction = False
    self.use_static_friction = False

  def reset(self, init_state):
    self.state = init_state.copy()
    self.prev_state = init_state.copy()

  def step(self):
    dt = self.dt
    dynamics = self.dynamics
    tau_sliding = self.tau_sliding
    tau_static = self.tau_static
    prev_state = self.prev_state
    state = self.state
    action = self.action

    # compute sliding frictions
    sliding_friction = np.zeros((2,1))
    if self.use_sliding_friction:
      sliding_friction = get_sliding_friction(tau_sliding, state)

    # compute stiction hints
    stictions = [0, 0]
    if self.use_static_friction:
      for i in range(2):
        omega = state.flatten()[2+i]
        prev_omega = prev_state.flatten()[2+i]
        changed_signs = (omega * prev_omega < 0)
        stationary = (abs(omega) < 1e-6) or changed_signs
        if stictions[i] == 0 and stationary:
          stictions[i] = 1

    self.prev_state = state.copy()

    # correct for stictions
    for i in range(2):
      if stictions[i] == 1:
        state[2+i, 0] = 0

    # forward propagate the state, preferrably with Runge Kutta
    use_rk = True
    if use_rk:
      # 4th order Runge Kutta
      k1 = dynamics.get_dstate_dt(state, action + sliding_friction, stictions, tau_static)
      k2 = dynamics.get_dstate_dt(state + dt/2 * k1, action + sliding_friction, stictions, tau_static)
      k3 = dynamics.get_dstate_dt(state + dt/2 * k2, action + sliding_friction, stictions, tau_static)
      k4 = dynamics.get_dstate_dt(state + dt * k3, action + sliding_friction, stictions, tau_static)
      state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    else:
      state += dt * dynamics.get_dstate_dt(state, action + sliding_friction, stictions, tau_static)

    self.state = state
