RecoilControllerComparison

# Overview
This project compares multiple control strategies for recoil smoothing using simulated Apex Legends weapon recoil patterns. Closed-loop feedback control, a Genetic Algorithm (GA), and a Rolling Horizon Evolutionary Algorithm (RHEA) are evaluated based on how effectively they keep the crosshair near the target centre (0,0) under varying noise levels.
The goal is to study control stability, accuracy, and smoothness rather than game automation and comparing the 3 controllers factoring in compute time.

## How To Run
```bash
git clone https://github.com/LeoSenthan/RecoilSmootherEvaluation cd CONTROLLER_COMPARISON_USING_RECOIL_SMOOTHING
pip install -r requirements.txt
python scripts/run_Baselines.py
```

## Problem Overview

![Problem Overview Diagram](diagrams/problem_overview.png)

# Skills Practiced

- Control systems fundamentals
- Evolutionary optimization
- Experimental fairness
- Metric-driven evaluation
- Clean engineering practices


# Weapon Recoil Simulation

- Each weapon uses a predefined recoil pattern (X/Y displacement per shot)
- Recoil is applied sequentially across a magazine
- Optional Gaussian noise models human inconsistency
- The simulator tracks cursor position in pixel space

The recoil data for the weapons was originally sourced from the Apex Legends Recoil repository:
[https://github.com/metaflow/apex-recoil](https://github.com/metaflow/apex-recoil)
Thank You Very Much

## Assumptions:
- No weapon sway or burst delays
- No recoil reset between shots
- Identical recoil patterns across episodes

# Controllers Implemented

## Closed-Loop Controller

![Closed-Loop Controller](diagrams/closedLoop.png)

- A proportional-derivative (PD-style) feedback controller that applies compensation based on the current cursor position.
- Reacts instantly to error
- No planning or learning
- Serves as a simple baseline


## Genetic Algorithm (GA) Controller

![Genetic Algorithm Controller](diagrams/GA.png)

- An offline evolutionary controller that optimizes a full compensation sequence for an entire magazine.
- Optimizes all shots simultaneously
- Uses fitness based on distance from origin and smoothness
- Evolves a population over several generations
- Uses deterministic recoil during training

## Rolling Horizon Evolutionary Algorithm (RHEA) Controller

![RHEA Controller](diagrams/RHEA.png)

- An online planner that re-optimizes compensation over a short horizon at every shot.
- Plans over a limited future window
- Executes only the first action of each plan
- Evaluates candidates using multiple noisy rollouts
- Trades computation time for adaptability


# Fairness & Evaluation Budget

To ensure fair comparison:
- GA and RHEA are constrained to similar evolutionary effort
- Population size × generations are matched
- Noise conditions are identical across controllers
- Multiple episodes are averaged per configuration

This avoids favoring online or offline methods unfairly.

# Metrics Collected

For each controller, weapon, and noise level:

- Mean Squared Error (MSE) from (0,0)
- Maximum Deviation during firing
- Trajectory Smoothness (sum of squared deltas)
- Mean Trajectory across episodes

All metrics are saved as structured JSON logs.
This allows direct comparison across controllers, weapons, and noise settings.

## Results & Logging Structure
results/
├── plots/
│   └── <weapon>/<noise>/<controller>.png
└── logs/
    └── <controller>/<weapon>/noise_<level>.json


# Visualization

- Mean recoil trajectories plotted per controller
- Y-axis inverted to match screen coordinates
- Identical scaling for fair visual comparison

# Key Findings
![Alternator-Noise-0.0](results/plots/alternator/0.0/alternator_noise0.0_seed0_comparison.png)

As you can see based on the diagram with no noise the Closed Loop controller performs the worst with data from corresponding JSON file recording "mse": 843.8315261896252, "max_dev": 40.29392499575498 and "smoothness": 1781.5122473879837. 

GA performs far better with data from corresponding JSON file recording "mse": 39.66193395023402, "max_dev": 9.237087877749532 and "smoothness": 26.18346527720866. 

The RHEA controller significantly outperforms the other controllers with data from corresponding JSON file recording "mse": 0.005858456851596085, "max_dev": 0.11226773148792975 and "smoothness": 0.0009015391284114397.

Overall the Closed Loop controller severely lags behind and the RHEA controller consistently outperforms the GA.

![Alternator-Noise-0.5](results/plots/alternator/0.5/alternator_noise0.5_seed0_comparison.png)

As you can see based on the diagram with noise of 0.5 standard deviations the Closed Loop controller performs the worst with data from corresponding JSON file recording "mse": 843.6517499534484, "max_dev": 40.32120016491354 and "smoothness": 1781.4762951328512 showing that the noise had very negligible impact on the spread compared to the results during no noise.

GA performs far better with data from corresponding JSON file recording "mse": 45.239131297834554, "max_dev": 10.066736078532912 and "smoothness": 27.70697050479335 compared to the closed loop controller and the noise had a minor impact on the spread of bullets.

The RHEA controller still consistently outperforms the other 2 controllers with data from corresponding JSON file recording "mse": 2.550424302380239, "max_dev": 3.0033756824885267, "smoothness": 3.419730275888363 showing that RHEA is robust to noise.

![Alternator-Noise-1.0](results/plots/alternator/1.0/alternator_noise1.0_seed0_comparison.png)

As you can see based on the diagram  the RHEA controller still is the strongest controller out of the 3 as the Closed Loop controller has metrics of "mse": 843.4739815484913, "max_dev": 40.34848440619182, "smoothness": 1781.5431253479148 and the GA controller has metrics of "mse": 51.789423507912176, "max_dev": 11.02611424257243,"smoothness": 34.744330249759464 while the RHEA controller has metrics of  "mse": 8.6257105459546, "max_dev": 5.585265922985584,"smoothness": 13.616415292672519 showing that the RHEA controller is still optimal when handling noise.

![High-Level-Findings](diagrams/high_level_comparison.png)

- Closed-loop control is stable but struggles with sustained recoil.
- GA produces smooth, near-optimal trajectories for deterministic recoil.
- RHEA adapts better to noise but is computationally expensive.

# Conclusion

- Offline optimization excels in predictable environments.
- Online planning improves robustness under uncertainty.
- Smoothness penalties significantly affect perceived stability
- Evaluation budget matters more than algorithm choice

# Extensions

- Shared evaluation budget framework
- Real-time constraints benchmarking
- Additional recoil models

# Requirements

Minimal dependencies:
- numpy
- matplotlib

# Disclaimer
It is designed as a learning and research project, not for gameplay automation.

# Credits
The recoil data for the weapons was originally sourced from the Apex Legends Recoil repository:
[https://github.com/metaflow/apex-recoil](https://github.com/metaflow/apex-recoil)
Thank You Very Much