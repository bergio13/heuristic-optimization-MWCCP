import time
import random
from enum import Enum
from typing import List, Generator, Tuple
from dataclasses import dataclass
from structs import Graph, cost_function_bit
import itertools


## Base version
class LocalSearch:
    def __init__(self, initial_solution, neighborhood_function, step_function, objective_function, max_iter=500):
        """
        Local Search framework.

        Args:
        - initial_solution: Starting solution for the search
        - neighborhood_function: Function to generate neighbor solutions
        - step_function: Function to select next step in search
        - objective_function: Function to compute the cost of a solution
        - max_iter: Maximum number of iterations for the search
        """
        self.current_solution = initial_solution
        self.neighborhood_function = neighborhood_function
        self.step_function = step_function
        self.objective_function = objective_function
        self.max_iter = max_iter

    def local_search(self):
          best_solution = self.current_solution
          best_cost = self.objective_function(best_solution)

          for i in range(self.max_iter):

            neighbors = self.neighborhood_function(self.current_solution)
            next_solution = self.step_function(neighbors, self.objective_function)
            next_cost = self.objective_function(next_solution)

            if next_cost < best_cost:
              best_solution = next_solution
              best_cost = next_cost
              continue


            if best_cost == 0:
              break

            self.current_solution = next_solution


          print("Required iterations:", i)
          return best_solution, best_cost


def best_improvement(neighbors, objective_function):
    if not neighbors:
      return None

    best_neighbor = None
    best_cost = float('inf')

    for neighbor in neighbors:
      cost = objective_function(neighbor)
      if cost < best_cost:
        best_neighbor = neighbor
        best_cost = cost

    return best_neighbor


def first_improvement(neighbors, objective_function):
    for neighbor in neighbors:
      if objective_function(neighbor) < objective_function(neighbors[0]):
        return neighbor
    return neighbors[0]


def random_neighbor(neighbors, objective_function):
  return random.choice(neighbors)


def cost_function(graph, permutation): # same function as above
    if permutation is None:
        return float('inf')
    # Create a dictionary for quick lookup of node positions in the current ordering
    position = {node: idx for idx, node in enumerate(permutation)}
    total_cost = 0
    # Iterate over all pairs of edges to count crossings
    for (u1, v1, w1), (u2, v2, w2) in itertools.combinations(graph.edges, 2):
        # Check if edges cross based on their positions
        if (u1 < u2 and position[v1] > position[v2]) or (u1 > u2 and position[v1] < position[v2]):
            # Add the sum of weights for the crossing edges to the total cost
            total_cost += w1 + w2

    return total_cost



## Improved version
class StepFunction(Enum):
    BEST_IMPROVEMENT = "best_improvement"
    FIRST_IMPROVEMENT = "first_improvement"
    RANDOM = "random"

class NeighborhoodType(Enum):
    SWAP = "swap"
    INSERT = "insert"
    REVERSE = "reverse"
    WINDOW = "window"
    BLOCK_SHIFT = "block_shift"

@dataclass
class SearchStatistics:
    iterations: int
    runtime: float
    improvement_history: List[float]
    best_cost: float
    plateau_counts: int
    local_optima_count: int
    constraint_violations_caught: int

class LocalSearch:
    def __init__(self,
                 graph: Graph,
                 initial_solution: List[int],
                 neighborhood_type: NeighborhoodType,
                 step_function: StepFunction,
                 max_iter: int = 500,
                 max_plateau: int = 50,
                 memory_size: int = 10,
                 window_size: int = 3,
                 block_size: int = 2):
        
        # Initialize stats
        self.stats = SearchStatistics(
            iterations=0,
            runtime=0,
            improvement_history=[],
            best_cost=float('inf'),
            plateau_counts=0,
            local_optima_count=0,
            constraint_violations_caught=0
        )

        # Initialize graph
        self.graph = graph
        
        # Now verify initial solution
        if not self.verify_constraints(initial_solution):
            raise ValueError("Initial solution violates precedence constraints")
            
        # Rest of initialization
        self.current_solution = initial_solution.copy()
        self.current_cost = self.calculate_cost(self.current_solution)
        self.neighborhood_type = neighborhood_type
        self.step_function = step_function
        self.max_iter = max_iter
        self.max_plateau = max_plateau
        self.memory_size = memory_size
        self.window_size = window_size
        self.block_size = block_size
        self.best_solutions = []

    def verify_constraints(self, solution: List[int]) -> bool:
        """Check if solution respects all precedence constraints"""
        # First check if solution contains all nodes from V exactly once
        if set(solution) != set(self.graph.V):
            self.stats.constraint_violations_caught += 1
            return False
            
        # Create position map for O(1) lookups
        position = {node: idx for idx, node in enumerate(solution)}
        
        # Check if any v1 appears after its required predecessor v2
        for v1 in self.graph.V:
            for v2 in self.graph.constraints[v1]:
                if position[v1] > position[v2]:
                    self.stats.constraint_violations_caught += 1
                    return False
        return True

    def is_valid_swap(self, solution: List[int], i: int, j: int) -> bool:
        """Check if swapping positions i and j maintains feasibility"""
        if i > j:
            i, j = j, i
            
        node_i, node_j = solution[i], solution[j]
        
        # Direct constraint check between swapped nodes
        if node_i in self.graph.constraints[node_j] or node_j in self.graph.constraints[node_i]:
            self.stats.constraint_violations_caught += 1
            return False
            
        # Create temporary solution with swap
        temp_solution = solution.copy()
        temp_solution[i], temp_solution[j] = temp_solution[j], temp_solution[i]
        return self.verify_constraints(temp_solution)

    def is_valid_sequence(self, solution: List[int], start: int, length: int) -> bool:
        """Validate that a sequence of nodes maintains precedence constraints"""
        sequence = solution[start:start + length]
        temp_solution = solution.copy()
        temp_solution[start:start + length] = sequence
        return self.verify_constraints(temp_solution)


    def swap_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors by swapping pairs"""
        for i in range(len(solution) - 1):
            if self.is_valid_swap(solution, i, i + 1):
                neighbor = solution.copy()
                neighbor[i], neighbor[i+1] = neighbor[i+1], neighbor[i]
                yield neighbor

    def insert_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors by insertion with constraint checking"""
        for i in range(len(solution)):
            element = solution[i]
            for j in range(len(solution)):
                if i != j:
                    neighbor = solution.copy()
                    neighbor.pop(i)
                    neighbor.insert(j, element)
                    if self.verify_constraints(neighbor):
                        yield neighbor

    def reverse_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors by reversing subsequences"""
        for i in range(len(solution)):
            for j in range(i + 2, len(solution)):
                neighbor = solution.copy()
                neighbor[i:j] = reversed(neighbor[i:j])
                if self.verify_constraints(neighbor):
                    yield neighbor

    def window_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors using sliding window operations"""
        n = len(solution)
        for start in range(n - self.window_size + 1):
            window = solution[start:start + self.window_size]
            
            # Adjacent swaps within window
            for i in range(len(window) - 1):
                pos1, pos2 = start + i, start + i + 1
                if self.is_valid_swap(solution, pos1, pos2):
                    neighbor = solution.copy()
                    neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
                    yield neighbor
            
            # Window rotations
            if self.is_valid_sequence(solution, start, self.window_size):
                # Left rotation
                neighbor = solution.copy()
                neighbor[start:start + self.window_size] = window[1:] + window[:1]
                if self.verify_constraints(neighbor):
                    yield neighbor
                
                # Right rotation
                neighbor = solution.copy()
                neighbor[start:start + self.window_size] = window[-1:] + window[:-1]
                if self.verify_constraints(neighbor):
                    yield neighbor

    def block_shift_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors by block shifting"""
        n = len(solution)
        for start in range(n - self.block_size + 1):
            if not self.is_valid_sequence(solution, start, self.block_size):
                continue
                
            block = solution[start:start + self.block_size]
            
            # Shift left
            if start > 0:
                neighbor = solution.copy()
                neighbor[start-1:start+self.block_size] = block + [solution[start-1]]
                if self.verify_constraints(neighbor):
                    yield neighbor
            
            # Shift right
            if start + self.block_size < n:
                neighbor = solution.copy()
                neighbor[start:start+self.block_size+1] = [solution[start+self.block_size]] + block
                if self.verify_constraints(neighbor):
                    yield neighbor
            
            # Handle block reversals for blocks larger than 2
            if self.block_size > 2:
                rev_block = list(reversed(block))
                if self.is_valid_sequence(solution, start, self.block_size):
                    # Shift reversed block left/right
                    if start > 0:
                        neighbor = solution.copy()
                        neighbor[start-1:start+self.block_size] = rev_block + [solution[start-1]]
                        if self.verify_constraints(neighbor):
                            yield neighbor
                    
                    if start + self.block_size < n:
                        neighbor = solution.copy()
                        neighbor[start:start+self.block_size+1] = [solution[start+self.block_size]] + rev_block
                        if self.verify_constraints(neighbor):
                            yield neighbor

    def generate_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors based on selected neighborhood type"""
        generators = {
            NeighborhoodType.SWAP: self.swap_neighborhood,
            NeighborhoodType.INSERT: self.insert_neighborhood,
            NeighborhoodType.REVERSE: self.reverse_neighborhood,
            NeighborhoodType.WINDOW: self.window_neighborhood,
            NeighborhoodType.BLOCK_SHIFT: self.block_shift_neighborhood
        }
        return generators[self.neighborhood_type](solution)

    def calculate_cost(self, solution: List[int]) -> float:
        return cost_function_bit(self.graph, solution)

    def select_next_solution(self, neighbors: List[List[int]]) -> Tuple[List[int], float]:
        """Select next solution based on step function"""
        if not neighbors:
            return self.current_solution, self.calculate_cost(self.current_solution)

        if self.step_function == StepFunction.BEST_IMPROVEMENT:
            best_neighbor = min(neighbors, key=self.calculate_cost)
            return best_neighbor, self.calculate_cost(best_neighbor)

        elif self.step_function == StepFunction.FIRST_IMPROVEMENT:
            current_cost = self.calculate_cost(self.current_solution)
            for neighbor in neighbors:
                neighbor_cost = self.calculate_cost(neighbor)
                if neighbor_cost < current_cost:
                    return neighbor, neighbor_cost
            return self.current_solution, current_cost

        else:  # RANDOM
            selected = random.choice(neighbors)
            return selected, self.calculate_cost(selected)

    def update_statistics(self, iteration: int, current_cost: float, is_improvement: bool):
        """Update search statistics"""
        self.stats.iterations = iteration
        self.stats.improvement_history.append(current_cost)

        if is_improvement:
            self.stats.plateau_counts = 0
        else:
            self.stats.plateau_counts += 1
            if self.stats.plateau_counts >= self.max_plateau:
                self.stats.local_optima_count += 1

    def local_search(self) -> Tuple[List[int], float, SearchStatistics]:
        """Execute local search"""
        start_time = time.time()
        best_solution = self.current_solution.copy()
        best_cost = self.current_cost
        plateau_counter = 0

        for iteration in range(self.max_iter):
            # Generate and collect neighbors
            neighbors = list(self.generate_neighborhood(self.current_solution))
            
            if not neighbors:
                break

            # Select next solution
            next_solution, next_cost = self.select_next_solution(neighbors)

            # Update best solution if improvement found
            if next_cost < best_cost:
                best_solution = next_solution.copy()
                best_cost = next_cost
                plateau_counter = 0
                self.best_solutions.append((best_solution.copy(), best_cost))
                if len(self.best_solutions) > self.memory_size:
                    self.best_solutions.pop(0)
            else:
                plateau_counter += 1

            # Update statistics
            self.update_statistics(iteration, next_cost, next_cost < best_cost)

            # Check stopping conditions
            if plateau_counter >= self.max_plateau:
                break

            self.current_solution = next_solution
            self.current_cost = next_cost

        self.stats.runtime = time.time() - start_time
        return best_solution, best_cost, self.stats