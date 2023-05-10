# Local Variables:
# python-indent: 2
# End:

import os
import argparse
import math
import numpy as np
import time
from dynamics import *
from simulation import Simulation
from sindy_simulation import SindyDynamics, SindySimulation
from sindy import eval_theta, eval_elements
import pickle

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sindy Model Comparison with Ground Truth for Control Theory Final Project")
  parser.add_argument("path", type=str, help="Path to sindy model")
  parser.add_argument("out_path", type=str, help="Output path to save comparison data")
  parser.add_argument("--use-torques", action="store_true", default=False, help="Drive the system with random torques")

  args = parser.parse_args()

  if os.path.exists(args.out_path):
    print("Output path already exists. Overwrite? (y/n)")
    yn = input().strip()
    if yn != "y":
      exit()

  with open(args.path, "rb") as f:
    sindy_model = pickle.load(f)

  dt = 0.01
  tau_sliding = sindy_model['data']['sliding_coeff']
  tau_static = sindy_model['data']['static_coeff']
  dynamics = Dynamics(**sindy_model['data']['dynamics_config'])
  init_state = np.array([0.6, 0.0, -0.01, 0.0]).reshape((4,1))
  simulation = Simulation(dt, dynamics, init_state, tau_sliding, tau_static)
  simulation.use_sliding_friction = sindy_model['data']['use_sliding_friction']
  simulation.use_static_friction = sindy_model['data']['use_static_friction']

  sindy_dynamics = SindyDynamics(sindy_model)
  sindy_simulation = SindySimulation(dt, sindy_dynamics, init_state)

  t = 0.0
  drive_timer = float('inf')
  simulation.reset(init_state)
  simulation.action = np.zeros((2, 1))
  sindy_simulation.reset(init_state)
  sindy_simulation.action = simulation.action
  mass_positions = dynamics.get_mass_positions(simulation.state)
  sindy_mass_positions = dynamics.get_mass_positions(sindy_simulation.state)

  ts = []

  states = []
  pes = []
  kes = []

  sindy_states = []
  spes = []
  skes = []

  while t < 2.0:
    sindy_simulation.action = simulation.action
    simulation.step()
    sindy_simulation.step()
    mass_positions = dynamics.get_mass_positions(simulation.state)
    sindy_mass_positions = dynamics.get_mass_positions(sindy_simulation.state)

    t += dt
    drive_timer += dt

    pe = dynamics.get_pe(simulation.state)
    ke = dynamics.get_ke(simulation.state)

    spe = dynamics.get_pe(sindy_simulation.state)
    ske = dynamics.get_ke(sindy_simulation.state)

    ts.append(t)
    states.append(simulation.state.flatten())
    pes.append(pe)
    kes.append(ke)

    sindy_states.append(sindy_simulation.state.flatten())
    spes.append(spe)
    skes.append(ske)

    if args.use_torques:
      if drive_timer >= 0.1:
        drive_timer = 0.0

        action = np.random.normal(size=(2,1))
        action[0] *= 0.6
        action[1] *= 0.2

        simulation.action = action
        sindy_simulation.action = action

  with open(args.out_path, 'wb') as f:
    data = {
      'ts': np.array(ts),

      'states': np.array(states),
      'pes': np.array(pes),
      'kes': np.array(kes),

      'sindy_states': np.array(sindy_states),
      'sindy_pes': np.array(spes),
      'sindy_kes': np.array(skes),
    }

    pickle.dump(data, f)

  print(f"Saved analysis file to {args.out_path}")

        



