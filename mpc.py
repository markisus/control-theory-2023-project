# Local Variables:
# python-indent: 2
# End:

dt = 0.01
MAX_TORQUE = 0.5
SPEED_PENALTY = 0.036
BUMP_SHARPNESS = 27.2

from sindy_simulation import SindySimulation
import numpy as np

def plan_to_torques(plan):
  #torques = signal.sawtooth(plan, 0.5)*max_torque
  #torques = np.sin(plan)*MAX_TORQUE
  torques = np.clip(plan, -MAX_TORQUE, MAX_TORQUE)
  return torques

def torques_to_plan(torques):
  #plan = np.arcsin(torques/MAX_TORQUE)
  plan = torques
  return plan

def rollout(dynamics, sindy_dynamics, plan, initial_state):
  """
  returns: the final state, rolling out the plan starting from initial_state
  """
  simulator = SindySimulation(dt, sindy_dynamics, initial_state)
  
  horizon = plan.shape[1]
  for i in range(horizon):
    simulator.action = plan[:,i:i+1]
    simulator.step()

  return simulator.state

def rollout_constant(dynamics, sindy_dynamics, action, initial_state, num_steps):
  simulator = SindySimulation(dt, sindy_dynamics, initial_state)
  simulator.action = action
  for i in range(num_steps):
    simulator.step()
  return simulator.state

def get_final_cost(dynamics, state, goal):
  final_ee = dynamics.get_mass_positions(state)[-2:].reshape((2,1))
  final_vee = dynamics.get_mass_velocities(state)[-2:].reshape((2,1))
  num_links = 2

  final_speed = np.linalg.norm(final_vee)
  goal_distance = np.linalg.norm(final_ee - goal)

  # a bump around goal_distance = 0
  # activates the speed penatly only when we are close to goal
  goal_bump = np.exp(-BUMP_SHARPNESS*abs(goal_distance))
  cost = (1-0.5*goal_bump)*goal_distance + goal_bump*final_speed*SPEED_PENALTY
  return cost

class MPC:
  def __init__(self, planning_horizon = 2):
    self.planning_horizion = planning_horizon
    self.rng = np.random.default_rng(0)

  def compute_action(self, dynamics, sindy_dynamics, state, goal, action):
    ee_pos = dynamics.get_mass_positions(state)[-2:].reshape((2,1))
    vel_ee = dynamics.get_mass_velocities(state)[-2:].reshape((2,1))
    dist = np.linalg.norm(goal-ee_pos)
    num_links = 2

    # for multi-link arms,
    # instead of heading towards the goal, we try to head towards a subgoal
    # which is on the path from the current position to the goal
    # this helps us get out of singular configurations where no local movements
    # get us towards the goal (eg goal is at origin and the arm is vertical)
    p = 0.9
    if np.linalg.norm(goal - ee_pos) < (dynamics.l1 + dynamics.l2)/2:
        p = 0.0

    subgoal = (1-p)*(goal) + (p)*ee_pos

    plan = action
    final_state = rollout_constant(
      dynamics, sindy_dynamics, plan_to_torques(plan), state, self.planning_horizion)
    cost = get_final_cost(dynamics, final_state, subgoal)

    epsilon = 0.1

    self.rng = np.random.default_rng(0)
    completely_stuck = True
    for _ in range(10):
      improved = False
      delta = epsilon * self.rng.uniform(-1.0, 1.0)
      for i in range(num_links):
        for pm in (-1, 1):
          perturb = np.zeros((num_links,1))
          perturb[i] = self.rng.uniform(-1.0, 1.0) * pm
          proposal = plan + perturb
          proposal_state = rollout_constant(
            dynamics, sindy_dynamics, plan_to_torques(proposal),
            state, self.planning_horizion)
          proposal_cost = get_final_cost(dynamics, proposal_state, subgoal)
          if proposal_cost < cost:
            cost = proposal_cost
            plan = proposal
            improved = True
            completely_stuck = False
        if not improved:
          epsilon *= 0.5
          if epsilon < 1e-3:
            break
        else:
          epsilon *= 1.5

    # Edge case:
    # since the algorithm is fully deterministic
    # it is possible to start in a bad initialization state where
    # the pseudorandom search does not find any improvement candidates
    # in this case, add some energy into the system to get us away
    # from this state
    if completely_stuck and np.linalg.norm(vel_ee) < 1e-3 and dist > (dynamics.l1 + dynamics.l2)/2:
      kick = 2.0 * self.rng.uniform(-1.0, 1.0, (2, 1))
      return kick
  
    return plan
