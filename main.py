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
from dynamics import *
from simulation import Simulation

dt = 0.01
tau_sliding = 0.1
tau_static = 0.2
dynamics = Dynamics(m1=0.5, m2=0.1, l1=0.5, l2=0.4)
init_state = np.array([0.6, 0.0, -0.01, 0.0]).reshape((4,1))
simulation = Simulation(dt, dynamics, init_state, tau_sliding, tau_static)

def to_ndc(vec2):
  out = vec2 * 0.2
  out[1] *= -1
  out += np.array([0.5, 0.3]).T
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


parser = argparse.ArgumentParser(description="Visualizer for Control Theory Final Project")
args = parser.parse_args()

need_restart = True
run = False
use_friction = True
drive = False
drive_timer = 0.0


app = ImguiSdlWrapper("Visualizer", 1280, 720)

while app.running:
    if need_restart:
      t = 0.0
      simulation.reset(init_state)
      simulation.action = np.zeros((2, 1))
      mass_positions = dynamics.get_mass_positions(simulation.state)
      prev_mass_positions = mass_positions.copy()
      need_restart = False
  
    app.main_loop_begin()

    if run:
      if use_friction:
        simulation.tau_sliding = tau_sliding
        simulation.tau_static = tau_static
      else:
        simulation.tau_sliding = 0.0
        simulation.tau_static = 0.0

      simulation.step()
      mass_positions = dynamics.get_mass_positions(simulation.state)
      t += dt

      if drive:
        drive_timer += dt
    
    imgui.begin("Gui", True)

    pe = dynamics.get_pe(simulation.state)
    ke = dynamics.get_ke(simulation.state)

    imgui.text(f"state {simulation.state.T}")
    imgui.text(f"potential {pe:<10.3} kinetic {ke:<10.3} total {pe + ke:<10.3}")

    tau1, tau2 = simulation.action.flatten()
    _, tau1 = imgui.slider_float("torque1", tau1, -5, 5)
    _, tau2 = imgui.slider_float("torque2", tau2, -5, 5)
    _, tau_sliding = imgui.slider_float("sliding torque", tau_sliding, 0.0, 10.0)
    _, tau_static = imgui.slider_float("static torque", tau_static, 0.0, 10.0)

    dynamics.static_torque = tau_static

    if imgui.button("zero torques"):
      tau1 = 0
      tau2 = 0
    
    simulation.action[0] = tau1
    simulation.action[1] = tau2

    need_restart = imgui.button("restart") or need_restart

    _, run = imgui.checkbox("Run", run)
    drive_changed, drive = imgui.checkbox("Drive", drive)
    _, use_friction = imgui.checkbox("Use Friction", use_friction)

    display_width = imgui.get_window_width() - 10
    overlayable = draw_overlayable_rectangle(1, 1, display_width)
    plot_joints(overlayable, mass_positions, imgui.get_color_u32_rgba(0,1,0,1))

    if drive_changed:
      drive_timer = 0.0

    if drive:
      if drive_timer >= 0.1:
        drive_timer = 0.0
        simulation.action = np.random.normal(size=(2,1))
        simulation.action[0] *= 0.6
        simulation.action[1] *= 0.2

    imgui.end()

    app.main_loop_end()
app.destroy()    


