import random
from collections import deque

import numpy as np

from source.structs import cost_function_bit


class MaxMinAntSystem:
    def __init__(self, graph, alpha=1.0, beta=2.0, evaporation_rate=0.5, ant_count=10, iterations=100, tau_min=0.1, tau_max=10):
        self.graph = graph
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta  # Influence of heuristic
        self.evaporation_rate = evaporation_rate
        self.ant_count = ant_count
        self.iterations = iterations
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Map vertex IDs to indices for consistent matrix access
        self.vertex_to_index = {v: i for i, v in enumerate(self.graph.V)}
        self.index_to_vertex = {i: v for i, v in enumerate(self.graph.V)}

    def initialize_pheromone_and_heuristic(self):
        """Initialize pheromone and heuristic matrices."""
        num_vertices = len(self.graph.V)
        pheromone = np.ones((num_vertices, num_vertices), dtype=np.float64) * self.tau_max

        heuristic = np.zeros((num_vertices, num_vertices), dtype=np.float64)
        for v in self.graph.V:
            idx = self.vertex_to_index[v]
            # Use a heuristic metric
            heuristic[idx, :] = 1 / (1 + self.graph.in_degree[v])
            #neighbors = self.graph.node_edges[v]
            #avg_w = np.mean([neigh[1] for neigh in neighbors])
            #heuristic[idx, :] = 1 / (1 + avg_w)

        return pheromone, heuristic


    def construct_solution(self, pheromone, heuristic):
        """Construct a solution probabilistically."""
        num_vertices = len(self.graph.V)
        unvisited = set(self.graph.V)
        solution = []

        for position in range(num_vertices):
            probabilities = []
            for v in unvisited:
                idx = self.vertex_to_index[v]
                probabilities.append(
                    (pheromone[idx][position] ** self.alpha) * (heuristic[idx][position] ** self.beta)
                )
            probabilities = np.array(probabilities) / sum(probabilities)

            # Select vertex probabilistically
            vertex = random.choices(list(unvisited), weights=probabilities, k=1)[0]
            solution.append(vertex)
            unvisited.remove(vertex)

        if not self.is_valid(solution):
            solution = self.repair_solution(solution)

        return solution

    def repair_solution(self, solution):
        """Repair a solution to satisfy constraints."""
        in_degree = self.graph.in_degree.copy()
        adjacency_list = self.graph.constraints
        index_map = {val: idx for idx, val in enumerate(solution)}

        queue = deque(v for v in self.graph.V if in_degree[v] == 0)
        repaired = []
        remaining = set(solution)

        while queue:
            candidates = [v for v in queue if v in remaining]
            if not candidates:
                candidates = list(remaining)

            v = min(candidates, key=index_map.get)
            repaired.append(v)
            remaining.remove(v)
            queue.remove(v)

            for neighbor in adjacency_list[v]:
                if neighbor in remaining:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return repaired

    def is_valid(self, solution):
        """Check if a solution satisfies all constraints."""
        positions = {v: i for i, v in enumerate(solution)}
        for v1 in solution:
            for v2 in self.graph.constraints[v1]:
                if positions[v1] > positions[v2]:
                    return False
        return True

    def update_pheromones(self, pheromone, best_solution, best_cost):
        """Update pheromone levels based on the best solution."""
        pheromone *= (1 - self.evaporation_rate)  # Evaporation

        for position, vertex in enumerate(best_solution):
            idx = self.vertex_to_index[vertex]
            pheromone[idx][position] += 1.0 / (1.0 + best_cost)

        # Apply pheromone bounds
        pheromone = np.clip(pheromone, self.tau_min, self.tau_max)

        return pheromone

    def ant_colony_optimization(self):
        """MMAS algorithm for MWCCP"""
        pheromone, heuristic = self.initialize_pheromone_and_heuristic()

        best_solution = None
        best_cost = float('inf')

        for _ in range(self.iterations):
            solutions = []
            fitnesses = []

            for _ in range(self.ant_count):
                solution = self.construct_solution(pheromone, heuristic)
                cost = cost_function_bit(self.graph, solution)
                solutions.append(solution)
                fitnesses.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution

            # Update pheromone with only the global best solution
            pheromone = self.update_pheromones(pheromone, best_solution, best_cost)

            # Dynamically adjust tau_min and tau_max
            self.tau_max = 1.0 / (1.0 - self.evaporation_rate) * (1.0 / best_cost)
            self.tau_min = self.tau_max / (2 * len(self.graph.V))

        return best_solution, best_cost