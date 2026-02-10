RecoilControllerComparison

# Overview
This project compares multiple control strategies for recoil smoothing using simulated Apex Legends weapon recoil patterns. Closed-loop feedback control, a Genetic Algorithm (GA), and a Rolling Horizon Evolutionary Algorithm (RHEA) are evaluated based on how effectively they keep the crosshair near the target centre (0,0) under varying noise levels.
The goal is to study control stability, accuracy, and smoothness rather than game automation and comparing the 3 controllers factoring in compute time.

## How To Run
```bash
git clone https://github.com/LeoSenthan/RecoilSmootherEvaluation
cd RECOILSMOOTHEREVALUATION
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

### Mathematics
Current Pos = After current shot fired (x1,y1) 
Prev Pos = Before current shot fired (x2,y2)
Kp = Proportional reaction constant e.g. 0.8
Kd = Smoothness constant to combat noise e.g. 0.1
Action = Compensation returned to be applied to Current Pos (x3,y3)

Action = - ( Kp * Current Pos) - Kd * (Current Pos - Prev Pos)


## Genetic Algorithm (GA) Controller

![Genetic Algorithm Controller](diagrams/GA.png)

- An offline evolutionary controller that optimizes a full compensation sequence for an entire magazine.
- Optimizes all shots simultaneously
- Uses fitness based on distance from origin and smoothness
- Evolves a population over several generations
- Uses deterministic recoil during training

### Mathematics

## Rolling Horizon Evolutionary Algorithm (RHEA) Controller

![RHEA Controller](diagrams/RHEA.png)

- An online planner that re-optimizes compensation over a short horizon at every shot.
- Plans over a limited future window
- Executes only the first action of each plan
- Evaluates candidates using multiple noisy rollouts
- Trades computation time for adaptability

### Mathematics

# Design Decisions

- Pixel-space evaluation chosen to match raw recoil data
- Smoothness penalty added to discourage jittery compensation
- Equalized evolutionary budgets to ensure fair comparison

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

## Quantitative Comparison (Alternator)

| Controller | Noise σ | MSE ↓ | Max Deviation ↓ | Smoothness ↓ | Relative Compute Cost |
|------------|---------|-------|-----------------|--------------|-----------------------|
| Closed Loop|   0.0   | 843.83|      40.29      |    1781.51   |       Very Low        |
|     GA     |   0.0   | 39.66 |       9.24      |     26.18    |        Medium         |
|    RHEA    |   0.0   | 0.0059|       0.11      |    0.0009    |         High          |
| Closed Loop|   1.0   | 843.47|      40.35      |    1781.54   |       Very Low        |
|     GA     |   1.0   | 51.79 |      11.03      |     34.74    |        Medium         |
|    RHEA    |   1.0   | 8.63  |       5.59      |     13.62    |         High          |


![Alternator-Noise-0.0](results/plots/alternator/0.0/alternator_noise0.0_seed0_comparison.png)

With no noise (σ = 0.0), RHEA dramatically outperforms both baselines.  
Compared to closed-loop control, RHEA reduces MSE by **over 99.99%** and maximum deviation by **~350×**, while producing an almost perfectly smooth trajectory.

GA achieves strong performance under deterministic recoil, reducing MSE by **~95%** relative to closed-loop control, but remains an order of magnitude less accurate than RHEA.


![Alternator-Noise-0.5](results/plots/alternator/0.5/alternator_noise0.5_seed0_comparison.png)

At moderate noise (σ = 0.5), closed-loop control remains largely unaffected due to its reactive nature but continues to exhibit large sustained deviation.

GA shows a modest degradation in accuracy and smoothness, while RHEA maintains robust performance, achieving a **~99.7% lower MSE** than closed-loop control and a **~45% reduction** compared to GA.

![Alternator-Noise-1.0](results/plots/alternator/1.0/alternator_noise1.0_seed0_comparison.png)

Under high noise (σ = 1.0), RHEA continues to outperform both alternatives despite performance degradation.  
While GA experiences increasing deviation due to its offline nature, RHEA maintains a **~6× lower MSE** and **~2× lower maximum deviation**, demonstrating superior robustness to uncertainty.

![High-Level-Findings](diagrams/high_level_comparison.png)

- Closed-loop control is stable but struggles with sustained recoil.
- GA produces smooth, near-optimal trajectories for deterministic recoil.
- RHEA adapts better to noise but is computationally expensive.

# Limitations

- No human reaction delay
- Deterministic recoil during GA training
- Simplified physics model
- Assumes fire rate is constant.

# Conclusion

- Offline optimization excels in predictable environments.
- Online planning improves robustness under uncertainty.
- Smoothness penalties significantly affect perceived stability.
- Evaluation budget matters more than algorithm choice.

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