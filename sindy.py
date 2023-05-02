# Local Variables:
# python-indent: 2
# End:

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import pickle
import itertools
from collections import defaultdict
from sklearn import linear_model
from matplotlib import pyplot as plt

def make_derivs(states, dt):
  num_samples = states.shape[0]
  x0 = states[:-1,:]
  x1 = states[1:,:]
  return (x1 - x0)/dt

def eval_theta_single(feature_names, elements, col_activations):
  assert elements.shape[0] == 1

  # elements is (num_samples, num_elements)
  elements = elements.flatten()
  result = np.zeros((len(feature_names),1))

  for i, feature_name in enumerate(feature_names):
    if col_activations[i] < 1:
      continue

    # print("feature name", feature_name)
    acc = 1.0
    tokens = feature_name.split(".")
    for token in tokens:
      base, exp = token.split("^")
      base = int(base)
      exp = int(exp)
      # print("\t{base}^{exp}")
      acc *= elements[base]**exp
    result[i] = acc

  return result.T
  

def eval_theta(feature_names, elements):
  # elements is (num_samples, num_elements)
  num_samples = elements.shape[0]
  if num_samples == 1:
    return eval_theta_single(feature_names, elements)
  
  result = np.zeros((num_samples, len(feature_names)))
  for i, feature_name in enumerate(feature_names):
    acc = 1.0
    tokens = feature_name.split(".")
    for token in tokens:
      base, exp = token.split("^")
      base = int(base)
      exp = int(exp)
      acc *= elements[:,base]**exp
    result[:,i] = acc

  return result

def get_feature_name(tup):
  d = defaultdict(lambda: 0)
  for t in tup:
    d[t] += 1
  return ".".join(f"{k}^{v}" for k,v in d.items())

def eval_elements(element_ids, states, actions):
  # states is (num_samples, state_dim)
  # actions is (num_samples, action_dim)
  
  num_samples = states.shape[0]
  
  # build element data
  element_data = np.empty((num_samples, len(element_ids)))

  for i in range(len(element_ids)):
    func = element_ids[i]
    funcname = func[0]
    
    if funcname == "1":
      element_data[:,i] = 1
    else:
      # sin, cos, id
      argname = func[1]
      idx = func[2]

      if argname == "action":
        arg = actions[:,idx]
      elif argname == "state":
        arg = states[:,idx]
      else:
        raise ValueError(f"Unknown argname {argname}")

      if funcname == "cos":
        element_data[:,i] = np.cos(arg)
      elif funcname == "sin":
        element_data[:,i] = np.sin(arg)
      elif funcname == "id":
        element_data[:,i] = arg
      elif funcname == "is_small":
        element_data[:,i] = np.exp(-arg**2)
      else:
        raise ValueError(f"Unknown funcname {funcname}")
  
  return element_data

def make_library(states, actions):
  num_samples = states.shape[0]

  element_ids = [
    ("1",),
    ("id", "action", 0),
    ("id", "action", 1),
    # ("is_small", "action", 0),
    # ("is_small", "action", 1),
    ("cos", "state", 0),
    ("sin", "state", 0),
    ("cos", "state", 1),
    ("sin", "state", 1),
    ("id", "state", 2),
    ("id", "state", 3),
    # ("is_small", "state", 2),
    # ("is_small", "state", 3),
  ]

  elements = eval_elements(element_ids, states, actions)
  feature_names = []

  i = 0
  for m in itertools.combinations_with_replacement(range(len(element_ids)), 4):
    m = tuple(m)
    feat_name = get_feature_name(m)
    feature_names.append(feat_name)

  lib = eval_theta(feature_names, elements)
  return element_ids, feature_names, lib


def make_sindy(states, actions, dt, alpha=0.01, tol=0.05):
  dotX = make_derivs(states, dt)
  elements, feature_names, Theta = make_library(states[:-1,:], actions[:-1,:])
  # print("Theta shape", Theta.shape)
  # print("feature_names", len(feature_names))

  clf = linear_model.Lasso(alpha=0.01, tol=0.05, fit_intercept=False)

  Theta_pd = pd.DataFrame(Theta, columns=feature_names)
  estimator = clf.fit(Theta_pd, dotX)

  return {
    'clf': clf,
    'elements': elements,
    'feature_names': feature_names
  }

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sindy Model Creator for Control Theory Final Project")
  parser.add_argument("path", type=str, help="Path to collected data")
  parser.add_argument("out_path", type=str, help="Path to save the resulting SINDy model")

  args = parser.parse_args()

  if os.path.exists(args.out_path):
    print("Output path already exists. Overwrite? (y/n)")
    yn = input().strip()
    if yn != "y":
      exit()

  with open(args.path, "rb") as f:
    data = pickle.load(f)

  sindy_model = make_sindy(data["states"], data["actions"], data["dt"])
  sindy_model["data"] = data

  with open(args.out_path, "wb") as f:
    pickle.dump(sindy_model, f)

