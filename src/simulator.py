import os
import json
import numpy as np

class RecoilSimulator:
    def __init__(self, weapon_name: str, noise_std: float = 0, data_folder: str = "data"):
        # Load metadata
        with open(os.path.join(data_folder, "weapons.json"), "r") as f:
            weapons = json.load(f)

        # Load weapon data
        if weapon_name not in weapons:
            raise ValueError(f"Weapon '{weapon_name}' not found in data folder.")
        
        weapon_file = weapons[weapon_name]["file"]
        weapon_data = np.load(os.path.join(data_folder, weapon_file), allow_pickle=True).item()

        self.X = weapon_data["X"]
        self.Y = weapon_data["Y"]
        self.T = weapon_data["T"]

        self.num_shots = len(self.X)
        self.weapon_name = weapon_name

        # Simulation state
        self.shot_index = 0
        self.pos = np.array([0.0, 0.0])

        # Background Noise
        self.noise_std = max(0,noise_std)

    def step(self):
        if self.shot_index >= self.num_shots:
            return None
        
        dx = self.X[self.shot_index]
        dy = self.Y[self.shot_index]
        dt = self.T[self.shot_index]

        self.pos += np.array([dx,dy])
        ndx,ndy = self.add_noise()
        self.shot_index += 1

        return {"shot": self.shot_index, "pos": self.pos.copy(), "delta": (dx+ndx, dy+ndy), "dt": dt}
    
    def add_noise(self):
        dx = np.random.normal(0,self.noise_std)
        dy = np.random.normal(0,self.noise_std)
        self.pos += np.array([dx,dy])
        return dx,dy

    def reset(self):
        self.shot_index = 0
        self.pos = np.array([0.0,0.0])
        return 
    
    def get_trajectory(self):
        self.reset()
        trajectory = []
        while True:
            state = self.step()
            if state is None:
                break
            trajectory.append(state["pos"].copy())
        return np.array(trajectory)
    
    def plot_trajectory(self):
        trajectory = self.get_trajectory()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 8))
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{self.weapon_name} Recoil Trajectory")
        plt.grid(True)
        plt.show()