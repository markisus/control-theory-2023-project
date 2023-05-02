# Local Variables:
# python-indent: 2
# End:

import argparse
import pickle
import os
import numpy as np
from dynamics import *
from simulation import *

parser = argparse.ArgumentParser(description="Data Generation for Control Theory Final Project")
parser.add_argument("path", type=str, help="Where to save the data")
parser.add_argument("--static-friction", action="store_true", default=False, help="Use static friction")
parser.add_argument("--sliding-friction", action="store_true", default=False, help="Use sliding friction")
args = parser.parse_args()

if os.path.exists(args.path):
  print(f"Path {args.path} already exists. Overwrite? (y/n)")
  yn = input()
  if yn.strip() != 'y':
    exit()

dynamics_config = {
  'm1': 0.5,
  'm2': 0.1,
  'l1': 0.5,
  'l2': 0.4
}

dynamics = Dynamics(**dynamics_config)

init_state = np.array([0.0, 0.0, 0.0, 0.0]).reshape((4,1))

tau_sliding = 0.1
tau_static = 0.2
dt = 0.01

max_t = 10.0
drive_timer = None
t = 0
simulation = Simulation(dt, dynamics, init_state, tau_sliding, tau_static)
simulation.action = np.random.normal(size=(2,1))
simulation.use_static_friction = args.static_friction
simulation.use_sliding_friction = args.sliding_friction

states = []
actions = []

while t < max_t:
  states.append(np.copy(simulation.state.flatten()))

  if drive_timer is None or drive_timer >= 0.1:
    drive_timer = 0.0
    simulation.action = np.random.normal(size=(2,1))
    simulation.action[0] *= 0.6
    simulation.action[1] *= 0.2

  actions.append(np.copy(simulation.action.flatten()))
  simulation.step()
  t += dt
  drive_timer += dt

states = np.array(states)
actions = np.array(actions)

data = {
  'dt': dt,
  'states': states,
  'actions': actions,
  'use_sliding_friction': args.sliding_friction,
  'use_static_friction': args.static_friction,
  'sliding_coeff': tau_sliding,
  'static_coeff': tau_static,
  'dynamics_config': dynamics_config
}

with open(args.path, 'wb') as f:
  pickle.dump(data, f)
