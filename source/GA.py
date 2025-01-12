import abc
import os
import random
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import List, Tuple

from source.structs import cost_function_bit, load_instance


@dataclass
class GAParameters:
    population_size: int = 100
    generations: int = 100
    elite_size: int = 10
    tournament_size: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.8
    constraint_penalty: float = 100000

    def __hash__(self):
        return hash((self.population_size, self.generations, self.elite_size, self.tournament_size, self.mutation_rate, self.crossover_rate, self.constraint_penalty))

    def __eq__(self, other):
        return hash(self) == hash(other) if isinstance(other, GAParameters) else False

    def __str__(self):
        return f"GAParameters(population_size={self.population_size}, generations={self.generations}, elite_size={self.elite_size}, tournament_size={self.tournament_size}, mutation_rate={self.mutation_rate}, crossover_rate={self.crossover_rate}, constraint_penalty={self.constraint_penalty})"

    def __repr__(self):
        return str(self)

class GeneticAlgorithm:
    def __init__(self, params: GAParameters = None, verbose: bool = False):
        self.params = params or GAParameters()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.generation_stats = []
        self.verbose = verbose

    # Abstract methods
    @abc.abstractmethod
    def fitness(self, permutation: List[int]) -> float:
        pass

    @abc.abstractmethod
    def _init_population(self) -> List[List[int]]:
        pass

    @abc.abstractmethod
    def _generate_new_generation(self, population: List[any], fitnesses: List[float]) -> List[any]:
        pass

    def _tournament_selection(self, population: List[any], fitnesses: List[float]) -> any:
        """Tournament selection for parent selection"""
        tournament = random.sample(range(len(population)), min(self.params.tournament_size, len(population))) # Select k individuals
        winner = max(tournament, key=lambda i: fitnesses[i]) # Choose winner
        return population[winner]

    def run_generation(self, population: List[any]) -> Tuple[List[any], List[float]]:
        # Evaluate fitness of each individual
        fitnesses = [self.fitness(ind) for ind in population]

        # Store statistics
        gen_best_fitness = max(fitnesses)
        gen_avg_fitness = mean(fitnesses)
        self.generation_stats.append((gen_best_fitness, gen_avg_fitness))

        # Update best solution
        best_idx = fitnesses.index(gen_best_fitness)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_solution = population[best_idx]

        # Create next generation
        population = self._generate_new_generation(population, fitnesses)

        return population, fitnesses

    def run(self) -> Tuple[List[int], float]:
        """Execute the genetic algorithm"""
        # Initialize population
        population = self._init_population()

        for generation in range(self.params.generations):
            if self.verbose:
                print(f"Generation {generation + 1}/{self.params.generations}")
            population, _ = self.run_generation(population)

        return self.best_solution, - self.best_fitness


class GeneticAlgorithmMWCCP(GeneticAlgorithm):
    def __init__(self, graph: 'Graph', params: GAParameters = None, verbose: bool = True):
        self.graph = graph
        super().__init__(params, verbose=verbose)

    def fitness(self, V_order: List[int]) -> float:
        """Calculate fitness with penalty"""
        crossing_cost = cost_function_bit(self.graph, V_order)

        # Penalty for violated constraints
        penalty = self._calculate_constraint_violations(V_order) * self.params.constraint_penalty if self.params.constraint_penalty > 0 else 0
        return - crossing_cost - penalty

    def _calculate_constraint_violations(self, V_order: List[int]) -> int:
        """Count number of violated constraints"""
        violations = 0
        positions = {v: i for i, v in enumerate(V_order)}

        for v1 in V_order:
            for v2 in self.graph.constraints[v1]:
                if positions[v1] > positions[v2]:
                    violations += 1
        return violations

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Implement Order Crossover (OX) operator"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        # Create a mapping for fast lookup
        p1_segment = set(parent1[start:end])

        # Initialize child with the segment from parent1
        child = [-1] * size
        child[start:end] = parent1[start:end]

        # Fill remaining positions with elements from parent2
        j = end
        for i in range(size):
            current = parent2[(end + i) % size]
            if current not in p1_segment:
                child[j % size] = current
                j += 1

        return child

    def _swap_mutation(self, individual: List[int]) -> List[int]:
        """Swap mutation with variable number of swaps"""
        if random.random() < self.params.mutation_rate:
            num_swaps = random.randint(1, max(1, len(individual) // 10))
            mutated = individual.copy()
            for _ in range(num_swaps):
                i, j = random.sample(range(len(mutated)), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
            return mutated
        return individual

    def _repair_individual(self, individual: List[int]) -> List[int]:
        in_degree = self.graph.in_degree.copy()
        adjacency_list = self.graph.constraints
        index_map = {val: idx for idx, val in enumerate(individual)}

        queue = deque(v for v in self.graph.V if in_degree[v] == 0)
        repaired = []
        remaining = set(individual)

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

    def is_valid(self, individual: List[int]) -> bool:
        """Check if an individual satisfies all constraints."""
        positions = {v: i for i, v in enumerate(individual)}
        for v1 in individual:
            for v2 in self.graph.constraints[v1]:
                if positions[v1] > positions[v2]:
                    return False
        return True

    def _init_population(self) -> List[List[int]]:
        """Initialize population"""
        population = [random.sample(self.graph.V, len(self.graph.V)) for _ in range(self.params.population_size)]
        population = [self._repair_individual(ind) if not self.is_valid(ind) else ind for ind in population]

        return population

    def _generate_new_generation(self, population: List[List[int]], fitnesses: List[float]) -> List[List[int]]:
        next_generation = []

        # Elitism
        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        next_generation.extend([population[i] for i in elite_indices[:self.params.elite_size]])

        # Generate offspring
        while len(next_generation) < self.params.population_size:
            if random.random() < self.params.crossover_rate:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child = self._order_crossover(parent1, parent2)
            else:
                child = self._tournament_selection(population, fitnesses).copy()

            child = self._swap_mutation(child)
            child = self._repair_individual(child)
            next_generation.append(child)

        return next_generation

    def get_statistics(self) -> List[Tuple[float, float]]:
        """Return generation statistics"""
        return self.generation_stats

def solve_mwccp(graph: 'Graph', params: GAParameters = None) -> Tuple[List[int], float]:
    """Convenience function to solve MWCCP"""
    ga = GeneticAlgorithmMWCCP(graph, params)
    solution, cost = ga.run()
    return solution, cost

# Test the bottleneck using profiling
if __name__ == "__main__":
    medium_path = "../content/tuning/medium/"
    medium_large_path = "../content/tuning/medium_large/"
    medium_instances = [medium_path + f for f in os.listdir(medium_path) if
                              not f.endswith("DS_Store")]
    medium_large_instances = [medium_large_path + f for f in os.listdir(medium_large_path) if not f.endswith("DS_Store")]
    graphs = [load_instance(f) for f in medium_large_instances]

    GA_params = GAParameters(
        population_size=40,
        generations=150,
        elite_size=13,
        tournament_size=25,
        mutation_rate=0.255,
        crossover_rate=0.79,
        constraint_penalty=213178
    )

    solutions = []

    for graph in graphs:
        solution, cost = solve_mwccp(graph, GA_params)
        solutions.append((solution, cost))