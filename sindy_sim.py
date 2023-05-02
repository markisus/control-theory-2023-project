# Local Variables:
# python-indent: 2
# End:

import argparse
from overlayable import *
from imgui_sdl_wrapper import *
import imgui
import math
import numpy as np
import time
import pandas as pd
from dynamics import *
from simulation import Simulation
from sindy_simulation import SindyDynamics, SindySimulation
from sindy import eval_theta, eval_elements
from mpc import MPC
import pickle

def to_ndc(vec2):
  out = vec2.flatten() * 0.2
  out[1] *= -1
  out += np.array([0.5, 0.3])
  return out

def plot_joints(overlayable, mass_positions, color):
  prev = np.zeros((2,))
  prev_plot = to_ndc(prev).flatten()

  for i in range(len(mass_positions)//2):
    p = mass_positions[2*i:2*i+2]
    p_plot = to_ndc(p).flatten()

    overlay_line(overlayable, p_plot[0], p_plot[1], prev_plot[0], prev_plot[1],
                 color, 1)
    overlay_circle(overlayable, p_plot[0], p_plot[1], 0.01, color, 1)

    prev = p
    prev_plot = p_plot

def plot_circle(overlayable, goal_pos, color):
  plot_goal = to_ndc(goal_pos).flatten()
  overlay_circle(overlayable, plot_goal[0], plot_goal[1], 0.01, color, 1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sindy Model Simulator for Control Theory Final Project")
  parser.add_argument("path", type=str, help="Path to sindy model")

  args = parser.parse_args()

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

  mpc = MPC()

  run = False
  drive = False
  need_restart = True
  drive_timer = 0.0

  goal = np.zeros((2,1))

  app = ImguiSdlWrapper("Visualizer", 1280, 720)

  while app.running:
      if need_restart:
        t = 0.0
        simulation.reset(init_state)
        simulation.action = np.zeros((2, 1))
        sindy_simulation.reset(init_state)
        sindy_simulation.action = simulation.action
        mass_positions = dynamics.get_mass_positions(simulation.state)
        sindy_mass_positions = dynamics.get_mass_positions(sindy_simulation.state)
        need_restart = False

      app.main_loop_begin()

      if run:
        sindy_simulation.action = simulation.action
        simulation.step()
        sindy_simulation.step()
        mass_positions = dynamics.get_mass_positions(simulation.state)
        sindy_mass_positions = dynamics.get_mass_positions(sindy_simulation.state)
        t += dt

        if drive:
          drive_timer += dt

      imgui.begin("Gui", True)

      pe = dynamics.get_pe(simulation.state)
      ke = dynamics.get_ke(simulation.state)

      spe = dynamics.get_pe(sindy_simulation.state)
      ske = dynamics.get_ke(sindy_simulation.state)

      imgui.text(f"state {simulation.state.T}")
      imgui.text(f"potential {pe:<10.3} kinetic {ke:<10.3} total {pe + ke:<10.3}")
      imgui.text(f"sindy_potential {spe:<10.3} sindy_kinetic {ske:<10.3} sindy_total {spe + ske:<10.3}")

      tau1, tau2 = simulation.action.flatten()
      _, tau1 = imgui.slider_float("torque1", tau1, -5, 5)
      _, tau2 = imgui.slider_float("torque2", tau2, -5, 5)

      g1, g2 = goal.flatten()
      _, g1 = imgui.slider_float("goal x", g1, -0.5, 0.5)
      _, g2 = imgui.slider_float("goal y", g2, -0.5, 0.5)
      goal = np.array([[g1, g2]]).T

      if imgui.button("zero torques"):
        tau1 = 0
        tau2 = 0

      simulation.action[0] = tau1
      simulation.action[1] = tau2

      need_restart = imgui.button("restart") or need_restart

      _, run = imgui.checkbox("Run", run)
      drive_changed, drive = imgui.checkbox("Drive", drive)

      display_width = imgui.get_window_width() - 10
      overlayable = draw_overlayable_rectangle(1, 1, display_width)
      plot_joints(overlayable, mass_positions, imgui.get_color_u32_rgba(0,1,0,1))
      plot_joints(overlayable, sindy_mass_positions, imgui.get_color_u32_rgba(0,1,1,1))
      plot_circle(overlayable, goal, imgui.get_color_u32_rgba(1,1,1,1))

      if drive_changed:
        drive_timer = 0.0

      if drive:
        simulation.action = mpc.compute_action(dynamics, sindy_dynamics, simulation.state, goal, simulation.action.copy())
        # if drive_timer >= 0.1:
        #   drive_timer = 0.0
        #   simulation.action = np.random.normal(size=(2,1))
        #   simulation.action[0] *= 0.6
        #   simulation.action[1] *= 0.2

      imgui.end()

      app.main_loop_end()
  app.destroy()    


