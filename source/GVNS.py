from dataclasses import dataclass
import time
import random
from typing import List
from VND import ImprovedVND, VNDStatistics, StepFunction



def verify_constraints(solution: List[int], constraints) -> bool:
    """Check if solution respects all constraints"""
    
    if len(set(solution)) != len(solution):
        print("Duplicate elements in solution")
        return False
    try:
        position = {node: idx for idx, node in enumerate(solution)}
        for v1 in constraints:
            for v2 in constraints[v1]:
                if position[v1] > position[v2]:
                    return False
    except KeyError as e:
        print(f"KeyError: {e}")
        return False
            
    return True

def swap_neighborhood_shake(solution, constraints):
    if solution is None:
        return []

    # Randomly select two nodes to swap
    while True:
        i, j = random.sample(range(len(solution)), 2)
        neighbor = solution.copy() if isinstance(solution, list) else list(solution)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        if verify_constraints(neighbor, constraints):
            return neighbor

def n_swap_neighborhood_shake_generator(n=2):
    def shake(solution, constraints):
        if solution is None:
            return []

        neighbor = solution.copy() if isinstance(solution, list) else list(solution)
        for _ in range(n):
            neighbor = swap_neighborhood_shake(neighbor, constraints)
        return neighbor

    return shake

def insert_neighborhood_shake(solution, constraints):
    if solution is None:
        return []

    while True:
        i = random.randint(0, len(solution) - 1)
        neighbor = solution.copy() if isinstance(solution, list) else list(solution)
        element = neighbor.pop(i)
        j = random.randint(0, len(neighbor) - 1)
        neighbor.insert(j, element)
        if verify_constraints(neighbor, constraints):
            return neighbor


def reverse_neighborhood_shake(solution, constraints):
    if solution is None:
        return []

    while True:
        i, j = random.sample(range(len(solution)), 2)
        if abs(i - j) < 2:
            continue
        if i > j:
            i, j = j, i
        neighbor_first = tuple(solution[:i])
        neighbor = neighbor_first + tuple(reversed(solution[i: j + 1])) + tuple(solution[j + 1:])
        if verify_constraints(neighbor, constraints):
            return neighbor
        
        
@dataclass
class GVNSStatistics:
    total_iterations: int
    runtime: float
    best_cost: float
    improvement_history: List[float]
    vnd_stats: List[VNDStatistics]

class GVNS:
    def __init__(self, graph, initial_solution, local_search_neighborhoods,
                 objective_function, shaking_neighborhoods=None, max_iter=500, verbose=True, vnd_class=None, vnd_step_function=StepFunction.BEST_IMPROVEMENT, vnd_max_iter=100, vnd_max_no_improve=50, max_time = 60 * 5):
        """
        General Variable Neighborhood Search framework.

        Args:
        - initial_solution: Starting solution for the search
        - shaking_neighbourhood: Neighborhood function for shaking
        - local_search_neighbourhoods: List of neighborhood functions for local search
        - objective_function: Function to compute the cost of a solution
        - max_iter: Maximum number of iterations for the search
        """
        self.graph = graph
        self.current_solution = initial_solution
        self.shaking_neighborhoods = shaking_neighborhoods
        self.local_search_neighborhoods = local_search_neighborhoods
        self.objective_function = objective_function
        self.max_iter = max_iter
        self.VND = vnd_class or ImprovedVND
        self.verbose = verbose
        self.vnd_step_function = vnd_step_function
        self.vnd_max_iter = vnd_max_iter
        self.vnd_max_no_improve = vnd_max_no_improve
        self.max_time = max_time
        
        self.shaking_neighborhoods = shaking_neighborhoods or [
            swap_neighborhood_shake,
            n_swap_neighborhood_shake_generator(3),
            n_swap_neighborhood_shake_generator(4),
            insert_neighborhood_shake,
            reverse_neighborhood_shake
        ]
        
        self.stats = GVNSStatistics(
            total_iterations=0,
            runtime=0,
            best_cost=float('inf'),
            improvement_history=[],
            vnd_stats=[]
        )
        
    def construct_vnd(self, solution):
        return self.VND(
                    graph=self.graph, 
                    initial_solution=solution, 
                    step_function=self.vnd_step_function,
                    max_iter=self.vnd_max_iter,
                    max_no_improve=self.vnd_max_no_improve,
                    neighborhood_order=self.local_search_neighborhoods,
                )
    
    def verify_constraints(self, solution: List[int]) -> bool:
        """Check if solution respects all constraints"""
        position = {node: idx for idx, node in enumerate(solution)}
        for v1 in self.graph.V:
            for v2 in self.graph.constraints[v1]:
                if position[v1] > position[v2]:
                    return False
        return True

    def find_solution(self):
        start_time = time.time()
        vnd = self.construct_vnd(self.current_solution)
            
        self.current_solution, best_cost, stats = vnd.vnd_search()
        best_solution = self.current_solution
        
        self.stats.vnd_stats.append(stats)
        self.stats.best_cost = best_cost
        self.stats.improvement_history.append(best_cost)

        for i in range(self.max_iter):
            k = 0
            self.stats.total_iterations += 1
            
            if self.verbose:
                print(f"Iteration {i + 1}: Cost = {best_cost}")
            
            while k < len(self.shaking_neighborhoods):
                # pick random solution from the shaking neighbourhood
                shaken_x = self.shaking_neighborhoods[k](best_solution, self.graph.constraints)
                vnd = self.construct_vnd(shaken_x)
                new_solution, new_cost, stats = vnd.vnd_search()
                
                self.stats.vnd_stats.append(stats)

                if new_cost < best_cost:
                    best_solution, best_cost = new_solution, new_cost
                    self.stats.improvement_history.append(best_cost)
                    k = 0
                else:
                    k += 1
                    
                if time.time() - start_time > self.max_time:
                    break

            if best_cost == 0 or time.time() - start_time > self.max_time:
                break
        if self.verbose:
            print("Required iterations for GVNS:", i + 1)
            
        self.stats.runtime = time.time() - start_time
        return best_solution, best_cost, self.stats