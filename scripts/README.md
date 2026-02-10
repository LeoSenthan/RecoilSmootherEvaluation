# Scripts Folder

This folder contains executable scripts for running and evaluating different recoil-control controllers.
Each script can be run directly from the command line using Python.
All scripts assume the project root is on the Python path (handled internally).

# Common Usage

From the project root:
python scripts/<script_name>.py [arguments]

All scripts accept a weapon argument and optional hyperparameters.

Available weapons:
alternator, car, devotion_tc, flatline, havoc_tc, lstar,
prowler, r99, r301, rampage, re45, spitfire, volt

# run_closedloop.py

Runs the baseline Closed-Loop Controller, which applies a deterministic per-shot correction based on recoil data.

## Usage
python scripts/run_closedloop.py <weapon>

## Example
python scripts/run_closedloop.py r301

## Parameters
None (fully deterministic baseline).

# run_ga.py

Runs a Genetic Algorithm (GA) that evolves a full recoil compensation sequence offline, then applies it during firing.

## Usage
python scripts/run_ga.py <weapon> [options]

## Example
python scripts/run_ga.py r301 \
  --population 50 \
  --generations 10 \
  --mutation-rate 0.1 \
  --noise-std 0.5

## Parameters
  Argument	             Description
--population	         Number of genomes
--generations	         GA evolution steps
--mutation-rate	         Per-gene mutation probability
--elite-ratio	         Fraction of elite genomes
--mutation-std-factor	 Mutation magnitude
--smoothness-weight	     Penalizes jerky compensation
--noise-std	             Recoil noise during execution
--seed	                 Random seed

# run_rhea.py

Runs a Rolling Horizon Evolutionary Algorithm (RHEA) controller that replans compensation at each shot using short-horizon evolution.

## Usage
python scripts/run_rhea.py <weapon> [options]

## Example
python scripts/run_rhea.py r301 \
  --horizon 6 \
  --population 40 \
  --generations 20 \
  --noise-std 0.5

## Parameters
  Argument	Description
--horizon	        Number of shots planned ahead
--population	    Genomes per planning step
--generations	    Evolution steps per shot
--mutation-rate	    Mutation probability
--mutation-std	    Mutation scale
--noise-std	        Recoil noise
--rollouts	        Monte-Carlo rollouts per fitness eval
--seed	            Random seed

# run_baselines.py

Runs all controllers (Closed-Loop, GA, RHEA) across multiple weapons and noise levels, logs metrics, and saves plots.

This script is intended for benchmarking and comparison, not interactive tuning.

## Usage
python scripts/run_baselines.py

## Output

Saves trajectory plots to:
results/plots/
Saves metrics JSON files to:
results/logs/

## Arguments
No arguments are required.

# Output Summary
Script	            Output
run_closedloop.py	On-screen trajectory plot
run_ga.py	        On-screen trajectory plot
run_rhea.py	        On-screen trajectory plot
run_baselines.py	Saved plots + JSON logs

# Notes
All trajectories are in pixel space, centered at the initial aim point.
Y-axis is inverted to match screen coordinates.
Noise is applied inside the simulator, not the controller.