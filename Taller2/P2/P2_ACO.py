import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class AntColonyOptimization:
    def __init__(self, start, end, obstacles, grid_size=(10, 10), num_ants=10, evaporation_rate=0.1, alpha=0.1, beta=15):
        self.start = start
        self.end = end
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromones = np.ones(grid_size)
        self.best_path = None

    def _get_neighbors(self, position):
        pos_x, pos_y = position
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_x, new_y = pos_x + i, pos_y + j
                if (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1] and
                        (new_x, new_y) != position and (new_x, new_y) not in self.obstacles):
                    neighbors.append((new_x, new_y))
        return neighbors

    def _select_next_position(self, position, visited):
        neighbors = self._get_neighbors(position)
        probabilities = []
        total = 0
        for neighbor in neighbors:
            if neighbor not in visited:
                pheromone = self.pheromones[neighbor[1], neighbor[0]]
                heuristic = 1 / (np.linalg.norm(np.array(neighbor) - np.array(self.end)) + 0.1)
                probabilities.append((neighbor, pheromone ** self.alpha * heuristic ** self.beta))
                total += pheromone ** self.alpha * heuristic ** self.beta
        if not probabilities:
            return None
        probabilities = [(pos, prob / total) for pos, prob in probabilities]
        selected = np.random.choice(len(probabilities), p=[prob for pos, prob in probabilities])
        return probabilities[selected][0]

    def _evaporate_pheromones(self):
        self.pheromones *= (1 - self.evaporation_rate)

    def _deposit_pheromones(self, path):
        for position in path:
            self.pheromones[position[1], position[0]] += 1

    def find_best_path(self, num_iterations):
        for _ in range(num_iterations):
            all_paths = []
            for _ in range(self.num_ants):
                current_position = self.start
                path = [current_position]
                while current_position != self.end:
                    next_position = self._select_next_position(current_position, path)
                    if next_position is None:
                        break
                    path.append(next_position)
                    current_position = next_position
                all_paths.append(path)

            # FIX: el camino "mejor" debe (1) llegar al destino y (2) ser el más corto.
            valid_paths = [p for p in all_paths if p[-1] == self.end]
            if not valid_paths:
                self._evaporate_pheromones()
                continue

            valid_paths.sort(key=lambda x: len(x))
            best_path = valid_paths[0]

            self._evaporate_pheromones()
            self._deposit_pheromones(best_path)

            if self.best_path is None or len(best_path) < len(self.best_path):
                self.best_path = best_path

    def plot(self):
        cmap = LinearSegmentedColormap.from_list('pheromone', ['white', 'green', 'red'])
        plt.figure(figsize=(8, 8))
        plt.imshow(self.pheromones, cmap=cmap, vmin=np.min(self.pheromones), vmax=np.max(self.pheromones))
        plt.colorbar(label='Pheromone intensity')
        plt.scatter(self.start[0], self.start[1], color='orange', label='Start', s=100)
        plt.scatter(self.end[0], self.end[1], color='magenta', label='End', s=100)
        for obstacle in self.obstacles:
            plt.scatter(obstacle[0], obstacle[1], color='gray', s=900, marker='s')
        if self.best_path:
            path_x, path_y = zip(*self.best_path)
            plt.plot(path_x, path_y, color='blue', label='Best Path', linewidth=3)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title('Ant Colony Optimization')
        plt.legend()
        plt.grid(True)
        plt.show()

def study_case_1():
    print("Start of Ant Colony Optimization - First Study Case")
    start = (0, 0)
    end = (4, 7)
    obstacles = [(1, 2), (2, 2), (3, 2)]
    aco = AntColonyOptimization(start, end, obstacles)
    aco.find_best_path(100)
    aco.plot()
    print("End of Ant Colony Optimization")
    print("Best path: ", aco.best_path)

def study_case_2():
    print("Start of Ant Colony Optimization - Second Study Case")
    start = (0, 0)
    end = (4, 7)
    obstacles = [(0, 2), (1, 2), (2, 2), (3, 2)]
    # Más hormigas, evaporación más alta y mayor peso de la heurística para
    # rodear la pared horizontal en y=2.
    aco = AntColonyOptimization(
        start, end, obstacles,
        num_ants=30, evaporation_rate=0.3, alpha=1.0, beta=5.0,
    )
    aco.find_best_path(200)
    aco.plot()
    print("End of Ant Colony Optimization")
    print("Best path: ", aco.best_path)


def random_search(start, end, obstacles, n_trials=20, num_iterations=100, seed=42):
    """Random Search de hiperparámetros para ACO.

    Devuelve la mejor configuración encontrada (la que produce el camino válido
    más corto). Si una configuración no encuentra un camino, se descarta.
    """
    rng = np.random.default_rng(seed)
    search_space = {
        "num_ants":         lambda: int(rng.integers(5, 51)),       # [5, 50]
        "evaporation_rate": lambda: float(rng.uniform(0.05, 0.7)),
        "alpha":            lambda: float(rng.uniform(0.1, 3.0)),
        "beta":             lambda: float(rng.uniform(1.0, 15.0)),
    }

    best = {"length": np.inf, "params": None, "path": None}
    for trial in range(n_trials):
        params = {k: v() for k, v in search_space.items()}
        aco = AntColonyOptimization(start, end, obstacles, **params)
        aco.find_best_path(num_iterations)
        if aco.best_path is None or aco.best_path[-1] != end:
            print(f"[trial {trial:02d}] sin solución | params={params}")
            continue
        length = len(aco.best_path)
        print(f"[trial {trial:02d}] len={length:3d} | params={params}")
        if length < best["length"]:
            best = {"length": length, "params": params, "path": aco.best_path}

    print("\nMejor configuración encontrada:")
    print(f"  longitud = {best['length']}")
    print(f"  params   = {best['params']}")
    return best


def study_case_2_tuned():
    print("Random Search sobre el caso 2")
    start, end = (0, 0), (4, 7)
    obstacles = [(0, 2), (1, 2), (2, 2), (3, 2)]
    best = random_search(start, end, obstacles, n_trials=15, num_iterations=120)
    if best["params"] is None:
        print("Random Search no encontró solución.")
        return
    aco = AntColonyOptimization(start, end, obstacles, **best["params"])
    aco.find_best_path(200)
    aco.plot()
    print("Best path:", aco.best_path)


if __name__ == '__main__':
    study_case_1()
    study_case_2()
    # study_case_2_tuned()  # Descomentar para correr Random Search



