import time
import numpy as np
import os
import sys
current_file = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file)
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from src.controllers import ClosedLoopController, GAController, RHEAController
from src.simulator import RecoilSimulator


def run_full_magazine(controller, weapon_name="alternator", noise=0.0):
    sim = RecoilSimulator(weapon_name, noise_std=noise)
    sim.reset()
    controller.reset()

    while True:
        state = sim.step()
        if state is None:
            break
        action = controller.get_action(sim.shot_index - 1, sim.pos.copy())
        sim.pos += action


def time_controller(controller, runs=10, weapon="alternator", noise=0.0, evolve=False):
    times = []

    for _ in range(runs):
        start = time.perf_counter()

        # If GA, evolve once per run
        if evolve:
            controller.evolve()

        run_full_magazine(controller, weapon, noise)

        end = time.perf_counter()
        times.append((end - start) * 1000)  # convert to ms

    return np.mean(times), np.std(times)


if __name__ == "__main__":

    weapon = "alternator"
    noise = 0.0
    runs = 10

    closed = ClosedLoopController()
    ga = GAController(weapon)
    rhea = RHEAController(weapon)

    print(f"\nTiming over {runs} runs (weapon={weapon}, noise={noise})\n")

    closed_mean, closed_std = time_controller(closed, runs, weapon, noise)
    print(f"Closed Loop: {closed_mean:.3f} ms ± {closed_std:.3f}")

    ga_mean, ga_std = time_controller(ga, runs, weapon, noise, evolve=True)
    print(f"GA (including evolve): {ga_mean:.3f} ms ± {ga_std:.3f}")

    rhea_mean, rhea_std = time_controller(rhea, runs, weapon, noise)
    print(f"RHEA: {rhea_mean:.3f} ms ± {rhea_std:.3f}")
