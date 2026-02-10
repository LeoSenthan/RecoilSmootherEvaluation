import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

# Add project root to path
current_file = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file)
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from src.simulator import RecoilSimulator
from src.controllers import RHEAController

# Supported weapons
WEAPONS = [
    "alternator","car","devotion_tc","flatline","havoc_tc","lstar",
    "prowler","r99","r301","rampage","re45","spitfire","volt"
]

parser = argparse.ArgumentParser(description="Run RHEA recoil controller")

parser.add_argument(
    "weapon",
    nargs="?",
    default="alternator",
    choices=WEAPONS,
    help="Weapon name"
)

parser.add_argument("--horizon", type=int, default=6)
parser.add_argument("--population", type=int, default=50)
parser.add_argument("--generations", type=int, default=10)
parser.add_argument("--mutation-rate", type=float, default=0.1)
parser.add_argument("--mutation-std", type=float, default=0.05)
parser.add_argument("--noise-std", type=float, default=0.5)
parser.add_argument("--rollouts", type=int, default=3)
parser.add_argument("--smoothness", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()

print(f"Selected weapon: {args.weapon}")

# Initialize simulator
sim = RecoilSimulator(
    args.weapon,
    noise_std=args.noise_std
)

# Initialize controller
controller = RHEAController(
    weapon_name=args.weapon,
    horizon=args.horizon,
    population=args.population,
    generations=args.generations,
    mutation_rate=args.mutation_rate,
    mutation_std=args.mutation_std,
    noise_std=args.noise_std,
    rollouts=args.rollouts,
    smoothness_weight=args.smoothness,
    seed=args.seed
)

sim.reset()
controller.reset()
trajectory = []

# Step through each shot
for shot in range(sim.num_shots):
    state = sim.step()
    if state is None:
        break

    action = controller.get_action(shot, sim.pos)
    sim.pos += action
    trajectory.append(sim.pos.copy())

trajectory = np.array(trajectory)

# Plot trajectory
plt.figure(figsize=(6, 8))
plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    "g-o",
    label="RHEA compensated"
)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"{args.weapon} RHEA Compensated Trajectory")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()
plt.show()
