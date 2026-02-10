# scripts/run_ClosedLoop.py 
import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file)
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from src.simulator import RecoilSimulator
from src.controllers import ClosedLoopController

WEAPONS = [
    "alternator","car","devotion_tc","flatline","havoc_tc","lstar",
    "prowler","r99","r301","rampage","re45","spitfire","volt"
]

parser = argparse.ArgumentParser(description="Run GA recoil controller")

parser.add_argument(
    "weapon",
    nargs="?",
    default="alternator",
    choices=WEAPONS,
    help="Weapon name"
)
parser.add_argument("--noise-std", type=float, default=0.5)
parser.add_argument("--proportionalK", type = float, default = 0.8)
parser.add_argument("--noisynessK", type = float, default = 0.2)

args = parser.parse_args()
weapon_name = args.weapon

print(f"Selected weapon: {weapon_name}")

sim = RecoilSimulator(
    weapon_name,
    noise_std=args.noise_std
)

controller = ClosedLoopController (args.proportionalK, args.noisynessK)

sim.reset()
controller.reset()
trajectory = []

for i in range(sim.num_shots):
    state = sim.step()  
    if state is None:
        break

    action = controller.get_action(i, state["pos"])
    sim.pos += action
    trajectory.append(sim.pos.copy())

trajectory = np.array(trajectory)

# Plot Result After Compensation Movements
plt.figure(figsize=(6, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-o', label="Closed Loop Compensated")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"{weapon_name} Closed Loop Compensated Trajectory")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()
plt.show()
