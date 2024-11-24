from typing import Type, List, Tuple
from dataclasses import dataclass
import time
from VND import ImprovedVND, VNDStatistics
from construction import RandomizedConstruction


@dataclass
class GRASPStatistics:
    total_iterations: int
    runtime: float
    best_cost: float
    improvement_history: List[float]
    vnd_stats: List[VNDStatistics]

class GRASP:
    def __init__(self, graph, alpha, max_iterations, neighborhood_structures, objective_function, step_function, max_iter_vnd=100, max_no_improve_vnd=50, local_search: Type[ImprovedVND] = None, verbose=False):
        """
        GRASP framework with randomized construction and local search.

        Args:
        - graph: The input graph structure.
        - alpha: Parameter for controlling greediness in construction phase (0 = purely greedy, 1 = purely random).
        - max_iterations: Maximum number of GRASP iterations.
        - neighborhood_structures: List of neighborhood functions for local search.
        - objective_function: Function to compute the cost of a solution.
        - step_function: Function for selecting the best solution in a neighborhood.
        - max_iter_vnd: Maximum iterations for VND.
        - max_no_improve_vnd: Maximum iterations without improvement for VND.
        """
        self.graph = graph
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.neighborhood_structures = neighborhood_structures
        self.objective_function = objective_function
        self.step_function = step_function
        self.max_iter_vnd = max_iter_vnd
        self.max_no_improve_vnd = max_no_improve_vnd
        self.local_search = local_search or ImprovedVND
        self.verbose = verbose
        
        self.stats = GRASPStatistics(
            total_iterations=0,
            runtime=0,
            best_cost=float('inf'),
            improvement_history=[],
            vnd_stats=[]
        )

    def run(self) -> Tuple[List[int], float, GRASPStatistics]:
        best_solution = None
        best_cost = float('inf')
        
        start_time = time.time()

        for iteration in range(self.max_iterations):
            # Construction Phase: Generate an initial solution
            constructor = RandomizedConstruction(self.graph, self.alpha)
            initial_solution = constructor.greedy_randomized_construction()
            while len(initial_solution) != len(self.graph.V):
                initial_solution = constructor.greedy_randomized_construction()

            # Local Search Phase: Refine the initial solution
            local_search_constructor = self.local_search(
                graph=self.graph,
                initial_solution=initial_solution,
                step_function=self.step_function,
                max_iter=self.max_iter_vnd,
                max_no_improve=self.max_no_improve_vnd,
                neighborhood_order=self.neighborhood_structures,
                objective_function=self.objective_function
            )
            local_search_solution, local_search_cost, vnd_stats = local_search_constructor.vnd_search()

            self.stats.vnd_stats.append(vnd_stats)
            # local_search_constructor = VND(initial_solution, self.neighborhood_structures, self.constraints, self.objective_function, self.step_function, self.max_iter_vnd)
            # local_search_solution, local_search_cost = local_search_constructor.vnd()
            
            self.stats.total_iterations += vnd_stats.total_iterations

            # Update the best solution if the improved solution is better
            if local_search_cost < best_cost:
                best_solution = local_search_solution
                best_cost = local_search_cost
                self.stats.best_cost = best_cost
                self.stats.improvement_history.append(best_cost)
            if self.verbose:
                print(f"Iteration {iteration + 1}: Cost = {local_search_cost}, Best Cost = {best_cost}")

            if best_cost == 0:
                break
                
        self.stats.runtime = time.time() - start_time

        return best_solution, best_cost, self.stats
    
    def verify_constraints(self, solution: List[int]) -> bool:
        """Check if solution respects all constraints"""
        try:
            position = {node: idx for idx, node in enumerate(solution)}
            for v1 in self.graph.V:
                for v2 in self.graph.constraints[v1]:
                    if position[v1] > position[v2]:
                        return False
        except KeyError:
            return False
        
        return True
    
    def get_statistics(self):
        return {
            "total_iterations": self.stats.total_iterations,
            "runtime": self.stats.runtime,
            "best_cost": self.stats.best_cost,
            "improvement_history": self.stats.improvement_history,
            "vnd_stats": [s.__dict__ for s in self.stats.vnd_stats]
        }