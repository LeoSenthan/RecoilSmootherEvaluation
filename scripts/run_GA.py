# scripts/run_ga.py 
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
from src.controllers import GAController

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
parser.add_argument("--population", type=int, default=50)
parser.add_argument("--generations", type=int, default=10)
parser.add_argument("--mutation-rate", type=float, default=0.1)
parser.add_argument("--elite-ratio", type=float, default=0.2)
parser.add_argument("--mutation-std-factor", type=float, default=0.05)
parser.add_argument("--smoothness-weight", type=float, default=0.1)
parser.add_argument("--noise-std", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()
weapon_name = args.weapon

print(f"Selected weapon: {weapon_name}")

sim = RecoilSimulator(
    weapon_name,
    noise_std=args.noise_std
)

controller = GAController(
    weapon_name=weapon_name,
    population=args.population,
    generations=args.generations,
    mutation_rate=args.mutation_rate,
    elite_ratio=args.elite_ratio,
    mutation_std_factor=args.mutation_std_factor,
    smoothness_weight=args.smoothness_weight,
    seed=args.seed
)

print("Running GA evolution...")
controller.evolve()
print("Best GA sequence computed.")


sim.reset()
controller.reset()
trajectory = []

for i in range(sim.num_shots):
    state = sim.step()  # Cumulative recoil before GA
    if state is None:
        break

    # Apply GA compensation
    action = controller.get_action(i, state["pos"])
    sim.pos += action
    trajectory.append(sim.pos.copy())

trajectory = np.array(trajectory)

# Plot Result After Compensation Movements
plt.figure(figsize=(6, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-o', label="GA Compensated")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"{weapon_name} GA Compensated Trajectory")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()
plt.show()
