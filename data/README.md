# Data Folder

This folder contains the recoil datasets for various weapons. Each weapon has its own
NumPy file containing the X, Y, and T recoil data. A JSON file provides metadata for
all weapons.

# File Structure

data/
├── r99.npy         # Recoil data for R-99
├── r301.npy        # Recoil data for R-301
├── flatline.npy    # Recoil data for Flatline
...
└── weapons.json    # Metadata file listing all weapons, shot counts, and file paths

# Data Format

Each `.npy` file contains a dictionary with three keys:
- "X": NumPy array of horizontal deltas per shot
- "Y": NumPy array of vertical deltas per shot
- "T": NumPy array of time intervals between shots (ms)

`weapons.json` contains a mapping of weapon names to:
- file: corresponding .npy filename
- size: number of shots in the weapon's recoil pattern

# Example Code To Load A Weapon

    import numpy as np
    import json
    import os

    data_folder = "data"

    # Load metadata
    with open(os.path.join(data_folder, "weapons.json"), "r") as f:
        weapons = json.load(f)

    # Load a specific weapon, e.g., R-99
    weapon_name = "r99"
    weapon_file = weapons[weapon_name]["file"]
    weapon_data = np.load(os.path.join(data_folder, weapon_file), allow_pickle=True).item()

    X = weapon_data["X"]
    Y = weapon_data["Y"]
    T = weapon_data["T"]

    print(f"{weapon_name}: {len(X)} shots")

# Notes
All positions are measured in either pixels or mouse deltas.
Timestamps are in milliseconds.
Data is read-only - DO NOT MODIFY THE `.npy` FILES.
To add a new weapon update `weapons.json` and add the corresponding .npy file. 

# Credits

The recoil data for the weapons was originally sourced from the Apex Legends Recoil repository:
[https://github.com/metaflow/apex-recoil](https://github.com/metaflow/apex-recoil)
Thank You Very Much