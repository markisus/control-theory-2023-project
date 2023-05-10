# Local Variables:
# python-indent: 2
# End:

import os
import argparse
import math
import numpy as np
import pickle
from matplotlib import pyplot as plt

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Comparison Plots for Control Theory Final Project")
  parser.add_argument("path", type=str, help="Path to comparison data from sindy_analyze.py")
  parser.add_argument("plot_prefix", type=str, help="Identifier to use for saved plots")

  args = parser.parse_args()

  # if os.path.exists(args.out_path):
  #   print("Output path already exists. Overwrite? (y/n)")
  #   yn = input().strip()
  #   if yn != "y":
  #     exit()

  with open(args.path, "rb") as f:
    comparison_data = pickle.load(f)

  ts = comparison_data['ts']

  states = comparison_data['states']
  pes = comparison_data['pes']
  kes = comparison_data['kes']

  sindy_states = comparison_data['sindy_states']
  sindy_pes = comparison_data['sindy_pes']
  sindy_kes = comparison_data['sindy_kes']

  fig, axs = plt.subplots(2, 2, figsize=(15, 5))

  for i, ax in enumerate(axs[0]):
    ax.plot(ts, states[:, i], c='b', linestyle='--')
    ax.plot(ts, sindy_states[:, i], c='b')
    ax.legend(["Ground Truth", "SINDy"])
    linfinity = np.max(np.abs(states[:,i] - sindy_states[:,i]))
    sub = fr"$L_\infty$ = {linfinity:0.3}"
    ax.set_xlabel("time (s)\n" + sub)
    ax.set_ylabel(fr"$\theta_{i+1}$ (rad)")

  for i, ax in enumerate(axs[1]):
    ax.plot(ts, states[:, 2+i], c='b', linestyle='--')
    ax.plot(ts, sindy_states[:, 2+i], c='b')
    ax.legend(["Ground Truth", "SINDy"])
    linfinity = np.max(np.abs(states[:,2 + i] - sindy_states[:,2 + i]))
    sub = fr"$L_\infty$ = {linfinity:0.3}"
    ax.set_xlabel("time (s)\n" + sub)
    ax.set_ylabel(fr"$\omega_{i+1}$ (rad)")
    

  fig.tight_layout()    
  plt.savefig(f"{args.plot_prefix}.state_plots.png")

  fig, axs = plt.subplots(1, 3, figsize=(15, 5))
  ax1, ax2, ax3 = axs
  ax1.plot(ts, kes, c='b', linestyle='--')
  ax1.plot(ts, sindy_kes, c='b')
  ax1.legend(["Ground Truth", "SINDy"])
  ax1.set_xlabel("time (s)")
  ax1.set_ylabel(fr"Kinetic Energy (J)")

  ax2.plot(ts, pes, c='b', linestyle='--')
  ax2.plot(ts, sindy_pes, c='b')
  ax2.legend(["Ground Truth", "SINDy"])
  ax2.set_xlabel("time (s)")
  ax2.set_ylabel(fr"Potential Energy (J)")

  ax3.plot(ts, pes + kes, c='b', linestyle='--')
  ax3.plot(ts, sindy_pes + sindy_kes, c='b')
  ax3.legend(["Ground Truth", "SINDy"])
  ax3.set_xlabel("time (s)")
  ax3.set_ylabel(fr"Total Energy (J)")

  fig.tight_layout()
  plt.savefig(f"{args.plot_prefix}.energy_plots.png")

  # plt.plot(ts, states[:, 0], c='b', linestyle='--')
  # plt.plot(ts, sindy_states[:, 0], c='b')
  # plt.legend(["Ground Truth", "SINDy"])
  # plt.xlabel("time (s)")
  # plt.ylabel(r"$\theta_1$ (rad)")
  # plt.show()


  # dt = 0.01
  # tau_sliding = sindy_model['data']['sliding_coeff']
  # tau_static = sindy_model['data']['static_coeff']
  # dynamics = Dynamics(**sindy_model['data']['dynamics_config'])
  # init_state = np.array([0.6, 0.0, -0.01, 0.0]).reshape((4,1))
  # simulation = Simulation(dt, dynamics, init_state, tau_sliding, tau_static)
  # simulation.use_sliding_friction = sindy_model['data']['use_sliding_friction']
  # simulation.use_static_friction = sindy_model['data']['use_static_friction']

  # sindy_dynamics = SindyDynamics(sindy_model)
  # sindy_simulation = SindySimulation(dt, sindy_dynamics, init_state)

  # t = 0.0
  # drive_timer = float('inf')
  # simulation.reset(init_state)
  # simulation.action = np.zeros((2, 1))
  # sindy_simulation.reset(init_state)
  # sindy_simulation.action = simulation.action
  # mass_positions = dynamics.get_mass_positions(simulation.state)
  # sindy_mass_positions = dynamics.get_mass_positions(sindy_simulation.state)

  # ts = []

  # states = []
  # pes = []
  # kes = []

  # sindy_states = []
  # spes = []
  # skes = []

  # while t < 2.0:
  #   sindy_simulation.action = simulation.action
  #   simulation.step()
  #   sindy_simulation.step()
  #   mass_positions = dynamics.get_mass_positions(simulation.state)
  #   sindy_mass_positions = dynamics.get_mass_positions(sindy_simulation.state)

  #   t += dt
  #   drive_timer += dt

  #   pe = dynamics.get_pe(simulation.state)
  #   ke = dynamics.get_ke(simulation.state)

  #   spe = dynamics.get_pe(sindy_simulation.state)
  #   ske = dynamics.get_ke(sindy_simulation.state)

  #   ts.append(t)
  #   states.append(simulation.state.copy())
  #   pes.append(pe)
  #   kes.append(ke)

  #   sindy_states.append(sindy_simulation.state.copy())
  #   spes.append(spe)
  #   skes.append(ske)

  #   if args.use_torques:
  #     if drive_timer >= 0.1:
  #       drive_timer = 0.0

  #       action = np.random.normal(size=(2,1))
  #       action[0] *= 0.6
  #       action[1] *= 0.2

  #       simulation.action = action
  #       sindy_simulation.action = action

  # with open(args.out_path, 'wb') as f:
  #   data = {
  #     'ts': np.array(ts),

  #     'states': np.array(states),
  #     'pes': np.array(pes),
  #     'kes': np.array(kes),

  #     'sindy_states': np.array(sindy_states),
  #     'sindy_pes': np.array(spes),
  #     'sindy_kes': np.array(skes),
  #   }

  #   pickle.dump(data, f)

  # print(f"Saved analysis file to {args.out_path}")

        



