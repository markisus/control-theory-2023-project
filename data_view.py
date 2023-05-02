# Local Variables:
# python-indent: 2
# End:

import argparse
from overlayable import *
from imgui_sdl_wrapper import *
from dynamics import Dynamics
import imgui
import math
import numpy as np
import time
import pickle

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


parser = argparse.ArgumentParser(description="Data Viewer for Control Theory Final Project")
parser.add_argument("path", type=str, help="Path to collected data")
args = parser.parse_args()

with open(args.path, "rb") as f:
  data = pickle.load(f)


states = data["states"]
actions = data["actions"]
dynamics = Dynamics(**data["dynamics_config"])

num_data_points = states.shape[0]
data_idx = 0
  
app = ImguiSdlWrapper("Data Viewer", 1280, 720)

play = False
last_frame_time = time.monotonic()

while app.running:
    app.main_loop_begin()
    
    imgui.begin("Gui", True)
    _, data_idx = imgui.slider_int("Data Idx", data_idx, 0, num_data_points-1)
    _, play = imgui.checkbox("Play", play)

    if data["use_sliding_friction"]:
      imgui.text(f"Sliding Coeff {data['sliding_coeff']}")

    if data["use_static_friction"]:
      imgui.text(f"Static Coeff {data['static_coeff']}")

    action = actions[data_idx,:]
    state = states[data_idx,:]
    imgui.text(f"action {action.flatten()}")
    imgui.text(f"state {state.flatten()}")

    mass_positions = dynamics.get_mass_positions(state)

    display_width = imgui.get_window_width() - 10
    overlayable = draw_overlayable_rectangle(1, 1, display_width)
    plot_joints(overlayable, mass_positions, imgui.get_color_u32_rgba(0,1,0,1))

    if play and (time.monotonic() - last_frame_time) > data["dt"]:
      data_idx = (data_idx + 1) % num_data_points
      last_frame_time = time.monotonic()

    imgui.end()

    app.main_loop_end()
app.destroy()    


