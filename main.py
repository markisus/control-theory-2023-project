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


dynamics = Dynamics(m1=0.5, m2=0.1, l1=0.5, l2=0.4)

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

init_state = np.array([0.6, 0.0, -0.01, 0.0]).reshape((4,1))

need_restart = True
run = False
use_friction = True
drive = False
drive_timer = 0.0

tau_sliding = 0.1
tau_static = 0.2

dt = 0.01
app = ImguiSdlWrapper("Visualizer", 1280, 720)

while app.running:
    if need_restart:
      t = 0.0
      state = init_state.copy()
      action = np.zeros((2, 1))
      mass_positions = dynamics.get_mass_positions(state)
      prev_mass_positions = mass_positions.copy()
      prev_state = state.copy()
      need_restart = False
  
    app.main_loop_begin()

    if run:
      # see if any stictions need to be turned on
      stictions = [0, 0]
      if use_friction:
        stictions_activated = False
        for i in range(2):
          omega = state.flatten()[2+i]
          prev_omega = prev_state.flatten()[2+i]

          # stationary if vel close to 0 or omega changed signs
          stationary = (abs(omega) < 1e-6) or (omega * prev_omega < 0)

          if stictions[i] == 0 and stationary:
            stictions[i] = 1
            # print(f"0 detected on state {i}, {state.T}")
            state[2+i, 0] = 0
            stictions_activated = True

      prev_mass_positions = mass_positions.copy()
      prev_state = state.copy()

      sliding_friction = np.zeros((2,1))
      if use_friction:
        sliding_friction = get_sliding_friction(tau_sliding, state)
        # print(f"state {state.T}, sliding friction {sliding_friction.T}")

      use_rk = True
      if use_rk:
        # 4th order Runge Kutta
        k1 = dynamics.get_dstate_dt(state, action + sliding_friction, stictions)
        k2 = dynamics.get_dstate_dt(state + dt/2 * k1, action + sliding_friction, stictions)
        k3 = dynamics.get_dstate_dt(state + dt/2 * k2, action + sliding_friction, stictions)
        k4 = dynamics.get_dstate_dt(state + dt * k3, action + sliding_friction, stictions)
        state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
      else:
        state += dt * dynamics.get_dstate_dt(state, action + sliding_friction, stictions)

      mass_positions = dynamics.get_mass_positions(state)
      t += dt

      if drive:
        drive_timer += dt
    
    vels = (mass_positions - prev_mass_positions)/dt

    imgui.begin("Gui", True)

    pe = dynamics.get_pe(state)
    ke = dynamics.get_ke(state)

    imgui.text(f"state {state.T}")
    imgui.text(f"potential {pe:<10.3} kinetic {ke:<10.3} total {pe + ke:<10.3}")

    # builtin_taus = dynamics.get_builtin_torques(state)
    # imgui.text(f"builtin torques {builtin_taus}")
    
    tau1, tau2 = action.flatten()
    _, tau1 = imgui.slider_float("torque1", tau1, -5, 5)
    _, tau2 = imgui.slider_float("torque2", tau2, -5, 5)
    _, tau_sliding = imgui.slider_float("sliding torque", tau_sliding, 0.0, 10.0)
    _, tau_static = imgui.slider_float("static torque", tau_static, 0.0, 10.0)

    dynamics.static_torque = tau_static

    if imgui.button("zero torques"):
      tau1 = 0
      tau2 = 0
    
    action[0] = tau1
    action[1] = tau2

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
        action = np.random.normal(size=(2,1))
        action[0] *= 0.6
        action[1] *= 0.2
        # print(f"drive action changed to {action.T}")

    imgui.end()

    app.main_loop_end()
app.destroy()    


