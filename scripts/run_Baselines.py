# scripts/run_Baselines.py
import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import json 

current_file = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file)
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from src.simulator import RecoilSimulator
from src.controllers import ClosedLoopController, GAController, RHEAController

WEAPONS = [
    "alternator","car","devotion_tc","flatline","havoc_tc","lstar",
    "prowler","r99","r301","rampage","re45","spitfire","volt"
]

episodes = 5
seed = 0
random.seed(seed)
np.random.seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def calculate_metrics(trajectories: list) -> dict:
    """
    Compute metrics for a list of trajectories (np.ndarray of shape [num_shots, 2])
    """
    trajectories = np.array(trajectories)  # [episodes, num_shots, 2]
    mean_traj = trajectories.mean(axis=0)

    mse = np.mean(np.linalg.norm(mean_traj, axis=1)**2)
    max_dev = np.max(np.linalg.norm(mean_traj, axis=1))
    smoothness = np.sum(np.linalg.norm(np.diff(mean_traj, axis=0), axis=1)**2)

    return {
        "mse": mse,
        "max_dev": max_dev,
        "smoothness": smoothness,
        "num_episodes": len(trajectories),
        "mean_trajectory": mean_traj.tolist()
    }


def save_metrics(metrics: dict, controller_name: str, weapon_name: str, noise: float, seed: int):
    log_dir = f"results/logs/{controller_name}/{weapon_name}"
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"noise{noise}_seed{seed}.json")
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)


for weapon in WEAPONS:
    weapon_name = weapon
    for noise in [0.0, 0.5, 1.0]:
        sim = RecoilSimulator(weapon_name, noise_std=noise)

        # Initialize all three controllers
        controllers = [
            ClosedLoopController(),
            GAController(weapon_name, seed=seed),
            RHEAController(
                weapon_name,
                horizon=6,
                population=8,
                generations=6,
                mutation_rate=0.1,
                mutation_std=0.05,
                noise_std=noise,
                rollouts=3,
                smoothness_weight=0.1,
                seed=seed
            )
        ]

        # Store mean trajectories for combined plot
        combined_mean_trajectories = {}

        for controller in controllers:
            # Precompute GA only once
            if isinstance(controller, GAController):
                controller.reset()
                controller.evolve()

            trajectories = []

            for ep in range(episodes):
                sim.reset()
                controller.reset()
                trajectory = []

                for shot in range(sim.num_shots):
                    state = sim.step()
                    if state is None:
                        break
                    pos = state["pos"]
                    action = controller.get_action(shot, pos)
                    sim.pos += action
                    trajectory.append(sim.pos.copy())

                trajectories.append(np.array(trajectory))

            mean_traj = np.mean(np.array(trajectories), axis=0)
            metrics = calculate_metrics(trajectories)
            save_metrics(metrics, controller.__class__.__name__, weapon_name, noise, seed)

            combined_mean_trajectories[controller.__class__.__name__] = mean_traj

            # Plot individual controller
            plot_dir = f"results/plots/{weapon_name}/{noise}"
            ensure_dir(plot_dir)
            plot_filename = f"{weapon_name}_noise{noise}_seed{seed}_{controller.__class__.__name__}.png"

            plt.figure(figsize=(6, 8))
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'o-', label=controller.__class__.__name__)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"{weapon_name} - Noise {noise} - {controller.__class__.__name__}")
            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, plot_filename))
            plt.close()

            print(f"Saved individual plot for {weapon_name} ({controller.__class__.__name__}, noise={noise})")

        # Plot combined overlay for this noise level
        plt.figure(figsize=(6, 8))
        for name, mean_traj in combined_mean_trajectories.items():
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'o-', label=name)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{weapon_name} - Noise {noise} - Comparison")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend()
        combined_plot_filename = f"{weapon_name}_noise{noise}_seed{seed}_comparison.png"
        plt.savefig(os.path.join(plot_dir, combined_plot_filename))
        plt.close()

        print(f"Saved combined overlay plot for {weapon_name}, noise={noise}")
