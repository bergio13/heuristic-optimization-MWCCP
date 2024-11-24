import time
from enum import Enum
from typing import List, Tuple
from dataclasses import dataclass
import random
from typing import Generator
from structs import cost_function_bit

class VND:
    def __init__(self, initial_solution, neighborhood_structures, constraints, objective_function, step_function,
                 max_iter=500, verbose=True):
        """
        Variable Neighborhood Descent framework.

        Args:
        - initial_solution: Starting solution for the search
        - neighborhood_structures: List of neighborhood functions to use
        - constraints: List of constraints for the problem
        - objective_function: Function to compute the cost of a solution
        - step_function: Function to select next step in search
        - max_iter: Maximum number of iterations for the search
        """
        self.current_solution = initial_solution
        self.neighborhood_structures = neighborhood_structures
        self.objective_function = objective_function
        self.step_function = step_function
        self.max_iter = max_iter
        self.constraints = constraints
        self.verbose = verbose


    def vnd(self, solution=None):
        if solution is not None:
            self.current_solution = solution
        best_solution = self.current_solution
        best_cost = self.objective_function(best_solution)
        l = 0

        for i in range(self.max_iter):

            # select neighborhood structure l
            neighborhood_function = self.neighborhood_structures[l]


            neighbors = neighborhood_function(self.current_solution, self.constraints)
            next_solution = self.step_function(neighbors, self.objective_function) # find local optimum in neighborhood

            if next_solution is not None:
                next_cost = self.objective_function(next_solution)
                print(next_cost)

                if next_cost < best_cost:
                    self.current_solution = best_solution
                    best_solution = next_solution
                    best_cost = next_cost
                    l = 0
                else:
                    l += 1

            if l >= len(self.neighborhood_structures):
                break

            if best_cost == 0:
                break

        if self.verbose:
            print("Required iterations for VND:", i + 1)
        return best_solution, best_cost
    
    
## Improved version
class NeighborhoodType(Enum):
    SWAP = "swap"
    INSERT = "insert"
    REVERSE = "reverse"

class StepFunction(Enum):
    BEST_IMPROVEMENT = "best_improvement"
    FIRST_IMPROVEMENT = "first_improvement"
    RANDOM = "random"

@dataclass
class VNDStatistics:
    total_iterations: int
    iterations_per_neighborhood: dict
    improvements_per_neighborhood: dict
    runtime: float
    improvement_history: List[float]
    neighborhood_switches: int
    best_cost_history: List[float]
    time_per_neighborhood: dict

class ImprovedVND:
    def __init__(self,
                 graph,
                 initial_solution: List[int],
                 step_function: StepFunction = StepFunction.BEST_IMPROVEMENT,
                 max_iter: int = 100,
                 max_no_improve: int = 20,
                 neighborhood_order: List[NeighborhoodType] = None,
                 objective_function: callable = None,
                 verbose: bool = False):
        """
        Enhanced Variable Neighborhood Descent framework.

        Args:
            graph: The graph object containing edges and constraints
            initial_solution: Starting permutation
            step_function: Method for selecting next solution
            max_iter: Maximum total iterations
            max_no_improve: Maximum iterations without improvement
            neighborhood_order: Custom order of neighborhood structures
        """
        self.graph = graph
        self.current_solution = initial_solution.copy() if isinstance(initial_solution, list) else list(initial_solution)
        self.step_function = step_function
        self.max_iter = max_iter
        self.max_no_improve = max_no_improve
        self.objective_function = objective_function or cost_function_bit
        self.current_cost = self.calculate_cost(self.current_solution)
        self.verbose = verbose

        # Initialize neighborhood structures if not provided
        self.neighborhood_order = neighborhood_order or [
            NeighborhoodType.SWAP,
            NeighborhoodType.INSERT,
            NeighborhoodType.REVERSE,
        ]

        # Initialize statistics
        self.stats = VNDStatistics(
            total_iterations=0,
            iterations_per_neighborhood={n: 0 for n in self.neighborhood_order},
            improvements_per_neighborhood={n: 0 for n in self.neighborhood_order},
            runtime=0,
            improvement_history=[],
            neighborhood_switches=0,
            best_cost_history=[],
            time_per_neighborhood={n: 0 for n in self.neighborhood_order}
        )

    def swap_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors by swapping adjacent pairs that respect constraints"""
        for i in range(len(solution) - 1):
            neighbor = solution.copy() if isinstance(solution, list) else list(solution)
            # Only swap if it doesn't violate constraints
            if (neighbor[i] not in self.graph.constraints[neighbor[i+1]] and
                neighbor[i+1] not in self.graph.constraints[neighbor[i]]):
                neighbor[i], neighbor[i+1] = neighbor[i+1], neighbor[i]
                if self.verify_constraints(neighbor):
                    yield neighbor

    def insert_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors by inserting elements at different positions"""
        for i in range(len(solution)):
            for j in range(len(solution)):
                if i != j:
                    neighbor = solution.copy() if isinstance(solution, list) else list(solution)
                    element = neighbor.pop(i)
                    neighbor.insert(j, element)
                    if self.verify_constraints(neighbor):
                        yield neighbor

    def reverse_neighborhood(self, solution: List[int]) -> Generator[List[int], None, None]:
        """Generate neighbors by reversing subsequences"""
        for i in range(len(solution)):
            neighbor_first = tuple(solution[:i])
            for j in range(i + 2, len(solution)):
                neighbor = neighbor_first + tuple(reversed(solution[i: j])) + tuple(solution[j:])
                if self.verify_constraints(neighbor):
                    yield neighbor

    def generate_neighborhood(self, solution: List[int], neighborhood_type: NeighborhoodType) -> Generator[List[int], None, None]:
        """Generate neighbors based on selected neighborhood type"""
        if neighborhood_type == NeighborhoodType.SWAP:
            return self.swap_neighborhood(solution)
        elif neighborhood_type == NeighborhoodType.INSERT:
            return self.insert_neighborhood(solution)
        elif neighborhood_type == NeighborhoodType.REVERSE:
            return self.reverse_neighborhood(solution)

    def verify_constraints(self, solution: List[int]) -> bool:
        """Check if solution respects all constraints"""
        try:
            position = {node: idx for idx, node in enumerate(solution)}
            for v1 in self.graph.V:
                for v2 in self.graph.constraints[v1]:
                    if position[v1] > position[v2]:
                        return False
        except KeyError as e:
            print(f"Error: {e}")
            return False
        
        return True

    def calculate_cost(self, solution: List[int]) -> float:
        """Calculate cost of a solution"""
        return self.objective_function(self.graph, solution)

    def select_next_solution(self, neighbors: Generator[List[int], None, None], current_cost: float) -> Tuple[List[int], float]:
        """Select next solution based on step function"""
        if not neighbors:
            return self.current_solution, current_cost

        if self.step_function == StepFunction.BEST_IMPROVEMENT:
            try:
                best_neighbor = min(list(neighbors), key=self.calculate_cost)
                return best_neighbor, self.calculate_cost(best_neighbor)
            except ValueError:
                return self.current_solution, current_cost

        elif self.step_function == StepFunction.FIRST_IMPROVEMENT:
            for neighbor in neighbors:
                neighbor_cost = self.calculate_cost(neighbor)
                if neighbor_cost < current_cost:
                    return neighbor, neighbor_cost
            return self.current_solution, current_cost

        elif self.step_function == StepFunction.RANDOM:
            try:
                selected = random.choice(list(neighbors))
                return selected, self.calculate_cost(selected)
            except IndexError | ValueError:
                return self.current_solution, current_cost
                

    def update_statistics(self, neighborhood_type: NeighborhoodType, improved: bool, runtime: float):
        """Update search statistics"""
        self.stats.iterations_per_neighborhood[neighborhood_type] += 1
        if improved:
            self.stats.improvements_per_neighborhood[neighborhood_type] += 1
        self.stats.time_per_neighborhood[neighborhood_type] += runtime

    def vnd_search(self) -> Tuple[List[int], float, VNDStatistics]:
        """Execute VND search"""
        start_time = time.time()

        best_solution = self.current_solution.copy()
        best_cost = self.current_cost
        self.stats.best_cost_history.append(best_cost)

        no_improve_counter = 0
        l = 0  # neighborhood index

        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"Iteration {iteration + 1}")
            self.stats.total_iterations += 1
            neighborhood_start_time = time.time()

            # Get current neighborhood type
            current_neighborhood = self.neighborhood_order[l]

            # Generate and explore neighborhood
            neighbors = self.generate_neighborhood(self.current_solution, current_neighborhood)
            next_solution, next_cost = self.select_next_solution(neighbors, best_cost)

            # Update statistics
            neighborhood_runtime = time.time() - neighborhood_start_time
            improved = next_cost < best_cost
            self.update_statistics(current_neighborhood, improved, neighborhood_runtime)

            if improved:
                self.current_solution = next_solution
                self.current_cost = next_cost
                best_solution = next_solution
                best_cost = next_cost
                self.stats.best_cost_history.append(best_cost)
                self.stats.improvement_history.append(best_cost)
                l = 0  # Reset to first neighborhood
                no_improve_counter = 0
                self.stats.neighborhood_switches += 1
            else:
                l += 1  # Move to next neighborhood
                no_improve_counter += 1

            # Check stopping conditions
            if l >= len(self.neighborhood_order) or no_improve_counter >= self.max_no_improve:
                break

            if best_cost == 0:
                break

        self.stats.runtime = time.time() - start_time
        return best_solution, best_cost, self.stats

    def get_statistics(self) -> dict:
        """Return detailed statistics about the search"""
        return {
            "total_iterations": self.stats.total_iterations,
            "runtime": self.stats.runtime,
            "improvements_per_neighborhood": dict(self.stats.improvements_per_neighborhood),
            "time_per_neighborhood": dict(self.stats.time_per_neighborhood),
            "neighborhood_switches": self.stats.neighborhood_switches,
            "convergence_history": self.stats.best_cost_history
        }
        
        
# Improved version with delta evaluation
class DeltaImprovedVND(ImprovedVND):
    def calculate_cost(self, solution: List[int]) -> float:
        """ Calculate the cost of a solution using the delta evaluation"""

        tuple_solution = tuple(solution)

        cost = self.graph.solution_costs.get(tuple_solution, None)

        if cost is not None:
            return cost

        current_solution = self.current_solution
        current_cost = getattr(self, 'current_cost', cost_function_bit(self.graph, current_solution))


        new_position = {node: idx for idx, node in enumerate(solution)}
        cur_position = {node: idx for idx, node in enumerate(current_solution)}

        diff = [node for node in new_position if new_position[node] != cur_position[node]]

        old_contribution = 0
        new_contribution = 0
        
        already_calculated = set()

        for v1 in diff:
            already_calculated.add(v1)
            # 1. Remove old contributions (from cur_position)
            for u1, w1 in self.graph.node_edges[v1]:
                u1m = u1 - 1
                # Forward edges (nodes after v1)
                for j in range(cur_position[v1] + 1, len(current_solution)):
                    v2 = current_solution[j]
                    if v2 in already_calculated:
                        continue
                    old_contribution += w1 * self.graph.node_edges_prefix_counts[v2][u1m] + self.graph.node_edges_prefix_sum[v2][u1m]
                    
            # Backward edges (nodes before v1)
            for j in range(cur_position[v1]):
                v2 = current_solution[j]
                if v2 in already_calculated:
                    continue
                for u2, w2 in self.graph.node_edges[v2]:
                    old_contribution += w2 * self.graph.node_edges_prefix_counts[v1][u2 - 1] + self.graph.node_edges_prefix_sum[v1][u2 - 1]
            
            # 2. Add new contributions (from new_position)
            for u1, w1 in self.graph.node_edges[v1]:
                # Forward edges (nodes after v1)
                u1m = u1 - 1
                for j in range(new_position[v1] + 1, len(solution)):
                    v2 = solution[j]
                    if v2 in already_calculated:
                        continue
                    new_contribution += w1 * (self.graph.node_edges_prefix_counts[v2][u1m]) + self.graph.node_edges_prefix_sum[v2][u1m]

            # Backward edges (nodes before v1)
            for j in range(new_position[v1]):
                v2 = solution[j]
                if v2 in already_calculated:
                    continue
                for u2, w2 in self.graph.node_edges[v2]:
                    new_contribution += w2 * self.graph.node_edges_prefix_counts[v1][u2 - 1] + self.graph.node_edges_prefix_sum[v1][u2 - 1]
                    

        calculated_cost = current_cost + new_contribution - old_contribution
        # Check if the calculated cost is correct
        # real_cost = cost_function_bit(self.graph, solution)
        # if calculated_cost != real_cost:
        #     print("Error")
        #     print("Calculated cost:", calculated_cost)
        #     print("Real cost:", real_cost)

        self.graph.solution_costs[tuple_solution] = calculated_cost

        return calculated_cost