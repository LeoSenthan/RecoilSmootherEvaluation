import numpy as np
import random 
from src.simulator import RecoilSimulator


class BaseController:
    """BareBones Controller Interface"""
    def reset(self):
        pass

    def get_action(self, shot_index: int, pos: np.ndarray) -> np.ndarray:
        """Return X/Y Compensation For The Current Shot."""
        raise NotImplementedError
    

class OpenLoopController(BaseController):
    """Predefined Sequence Of Compensation Moves."""
    def __init__(self,compensation_sequence: np.ndarray):
        self.sequence = compensation_sequence
    
    def reset(self):
        pass # No internal state for open-loop controller.

    def get_action(self, shot_index: int, pos: np.ndarray) -> np.ndarray:
        return self.sequence[shot_index]
    

class ClosedLoopController(BaseController):
    """Feedback Controller: Proportional Gain Based On Current Position, Kd Smoothens The Noise"""
    def __init__(self, Kp:float = 0.9, Kd: float = 0.1):
        self.Kp = Kp
        self.Kd = Kd
        self.prev_pos = np.array([0.0,0.0])

    def reset(self):
        self.prev_pos = np.array([0.0,0.0])
        
    def get_action(self, shot_index: int, pos: np.ndarray) -> np.ndarray:
        action = -self.Kp * pos - self.Kd * (pos - self.prev_pos)
        self.prev_pos = pos.copy()
        return action
    

class GAController(BaseController):
    """
    Genetic Algorithm Controller.
    Optimizes a full sequence of X/Y compensations for a weapon.
    Smooth trajectory and per-shot scaling to converge faster and stay near (0,0).
    """
    def __init__(self, weapon_name: str,
                 population: int = 50,
                 generations: int = 10,
                 mutation_rate: float = 0.1,
                 elite_ratio: float = 0.2,
                 mutation_std_factor: float = 0.05,
                 smoothness_weight: float = 0.1,
                 seed: int = None):
        
        self.weapon_name = weapon_name
        self.population_size = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.mutation_std_factor = mutation_std_factor
        self.smoothness_weight = smoothness_weight
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Simulator to get recoil data
        self.simulator = RecoilSimulator(weapon_name)
        self.num_shots = self.simulator.num_shots

        # Genome bounds based on recoil
        self.X = self.simulator.X
        self.Y = self.simulator.Y
        self.max_comp = np.stack([np.abs(self.X), np.abs(self.Y)], axis=1)

        # Placeholder for best sequence
        self.best_sequence = np.zeros((self.num_shots, 2))


    def random_genome(self) -> np.ndarray:
        """
        Initialize genome around rough negative of recoil 
        """
        base = np.stack([-self.X, -self.Y], axis=1)
        perturb = np.random.normal(0, self.max_comp * self.mutation_std_factor)
        genome = base + perturb
        # Clip per-shot so no overshoot
        genome = np.clip(genome, -self.max_comp, self.max_comp)
        return genome

    def fitness(self, genome) -> float:
        """
        Calculates How Accurate The Genonme Is When Simulating Recoil With Noise
        """
        self.simulator.reset()
        error = 0.0
        positions = []
        for i in range(self.num_shots):
            state = self.simulator.step()
            if state is None:
                break
            self.simulator.pos += genome[i]
            positions.append(self.simulator.pos.copy())
            error += np.linalg.norm(self.simulator.pos)**2

        positions = np.array(positions)
        # Add smoothness penalty
        smoothness = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)**2)
        return error + self.smoothness_weight * smoothness

    def select_elite(self, population, fitness):
        """
        Returns only the top elite_ratio percentage of the population based upon the fitness score
        """

        elite_size = max(1, int(self.population_size * self.elite_ratio))
        idx = np.argsort(fitness)
        return [population[i] for i in idx[:elite_size]]

    def crossover(self, p1, p2):
        mask = np.random.rand(self.num_shots, 2) < 0.5
        return np.where(mask, p1, p2)

    def mutate(self, genome):
        """
        Adds Slight Mutation To A Genome To Simulate 'Evolution'
        """
        mask = np.random.rand(self.num_shots, 2) < self.mutation_rate
        genome[mask] += np.random.normal(0, self.max_comp[mask] * self.mutation_std_factor, size=np.sum(mask))
        # Clip again
        genome = np.clip(genome, -self.max_comp, self.max_comp)
        return genome

    def evolve(self):
        population = [self.random_genome() for _ in range(self.population_size)]

        for gen in range(self.generations):
            fitness_vals = np.array([self.fitness(g) for g in population])
            elite = self.select_elite(population, fitness_vals)

            new_population = elite.copy()
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(elite, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population
            if gen % 10 == 0:
                print(f"Gen {gen:05d} | Best fitness: {fitness_vals.min():.2f}")

        # Store best genome
        final_fitness = np.array([self.fitness(g) for g in population])
        self.best_sequence = population[np.argmin(final_fitness)]

    def reset(self):
        self.simulator.reset()

    def get_action(self, shot_index: int, pos: np.ndarray):
        return self.best_sequence[shot_index]


class RHEAController(BaseController):
    """
    Rolling Horizon Evolutionary Algorithm Controller.
    Plans over a short horizon and selects the first action of the best genome.
    """

    def __init__(
        self,
        weapon_name: str,
        horizon: int = 5,           
        population: int = 8,       
        generations: int = 6,      
        mutation_rate: float = 0.1, 
        mutation_std: float = 0.05,  
        noise_std: float = 0.0,      
        rollouts: int = 5,          
        smoothness_weight: float = 0.0,
        seed: int | None = None,
    ):
        self.weapon_name = weapon_name
        self.horizon = horizon
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.noise_std = noise_std
        self.rollouts = rollouts
        self.smoothness_weight = smoothness_weight

        # RNG
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Template simulator (deterministic, for initialization)
        self.sim_template = RecoilSimulator(weapon_name, noise_std=0.0)
        self.num_shots = self.sim_template.num_shots

        # Max per-axis compensation
        self.max_action = np.stack([np.abs(self.sim_template.X),
                                    np.abs(self.sim_template.Y)], axis=1)

        self.current_best = np.zeros((self.horizon, 2))

    def reset(self):
        self.current_best = np.zeros((self.horizon, 2))

    def random_genome(self, horizon, start_shot):
        """Initialize genome around negative recoil (like GA)."""
        end_shot = min(start_shot + horizon, self.num_shots)
        X = self.sim_template.X[start_shot:end_shot]
        Y = self.sim_template.Y[start_shot:end_shot]
        base = np.stack([-X, -Y], axis=1)
        # Add small Gaussian perturbation
        noise = self.rng.normal(0, self.mutation_std, size=base.shape)
        genome = base + noise
        # Clip to max_action per shot
        genome = np.clip(genome, -self.max_action[start_shot:end_shot],
                         self.max_action[start_shot:end_shot])
        return genome

    def mutate(self, genome, start_shot):
        mask = self.rng.random(genome.shape) < self.mutation_rate
        genome[mask] += self.rng.normal(0, self.mutation_std, size=np.sum(mask))
        genome = np.clip(genome, -self.max_action[start_shot:start_shot+genome.shape[0]],
                         self.max_action[start_shot:start_shot+genome.shape[0]])
        return genome

    def crossover(self, p1, p2):
        mask = self.rng.random(p1.shape) < 0.5
        return np.where(mask, p1, p2)

    def fitness(self, genome, current_pos, start_shot):
        """Simulate genome and return expected squared distance from origin."""
        total_cost = 0.0

        for _ in range(self.rollouts):
            sim = RecoilSimulator(self.weapon_name, noise_std=self.noise_std)
            sim.reset()
            sim.pos = current_pos.copy()
            sim.shot_index = start_shot

            positions = []
            for action in genome:
                state = sim.step()
                if state is None:
                    break
                sim.pos += np.clip(action, -self.max_action[sim.shot_index-1],
                                   self.max_action[sim.shot_index-1])
                positions.append(sim.pos.copy())

            if positions:
                positions = np.array(positions)
                error = np.sum(np.linalg.norm(positions, axis=1)**2)
                smoothness = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)**2)
                total_cost += error + self.smoothness_weight * smoothness

        return total_cost / self.rollouts

    def get_action(self, shot_index, current_pos):
        horizon = min(self.horizon, self.num_shots - shot_index)
        population = [
            self.random_genome(horizon, shot_index) 
            for _ in range(self.population)
        ]

        for _ in range(self.generations):
            fitness_vals = [
                self.fitness(genome, current_pos, shot_index) 
                for genome in population
            ]

            # Select top 20% as elite
            elite_size = max(1, int(self.population * 0.2))
            elite_idx = np.argsort(fitness_vals)[:elite_size]
            elite = [population[i] for i in elite_idx]

            # Generate new population
            # Generate new population
            new_pop = elite.copy()
            while len(new_pop) < self.population:
                if len(elite) >= 2:
                    p1, p2 = random.sample(elite, 2)
                else:
                    p1 = p2 = elite[0]
                child = self.crossover(p1, p2)
                child = self.mutate(child, shot_index)
                new_pop.append(child)
            population = new_pop


        # Recompute fitness for final population
        final_fitness = [
            self.fitness(genome, current_pos, shot_index) 
            for genome in population
        ]
        best_genome = population[np.argmin(final_fitness)]
        self.current_best[:best_genome.shape[0]] = best_genome

        return best_genome[0]  # first action
