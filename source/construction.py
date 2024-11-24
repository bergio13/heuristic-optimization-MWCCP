from collections import deque, defaultdict
import numpy as np
from structs import cost_function_bit

class DeterministicConstruction:
    def __init__(self, graph):
        self.graph = graph
        self.pi = []  # store the final order of nodes in V

        # Precompute edge weights per node for efficiency
        self.node_weights = defaultdict(int)
        for u, v, weight in graph.edges:
            self.node_weights[v] += weight

    def greedy_construction(self):
        # Initialize candidates with in-degree 0 nodes
        candidates = deque([v for v in self.graph.V if self.graph.in_degree[v] == 0])

        while candidates:
            # Select node with lowest total edge weight
            best_node = min(candidates, key=lambda v: self.node_weights[v])

            # Add the selected best_node to the pi
            self.pi.append(best_node)
            candidates.remove(best_node)

            # Update in-degrees and add new candidates
            for v_next in self.graph.constraints[best_node]:
                self.graph.in_degree[v_next] -= 1
                if self.graph.in_degree[v_next] == 0:
                    candidates.append(v_next)
        # Verify the solution before returning
        if not self.verify_solution():
            raise ValueError("Construction resulted in invalid solution!")

        return self.pi
 

    def verify_solution(self):
        """
        Verify that the solution respects all constraints
        """
        if not self.pi:
            return False

        # Check if all nodes from V are present
        if set(self.pi) != set(self.graph.V):
            return False

        # Check if constraints are respected
        position = {node: idx for idx, node in enumerate(self.pi)}
        for v1 in self.graph.V:
            for v2 in self.graph.constraints[v1]:
                if position[v1] > position[v2]:
                    return False

        return True
    
    
    
class RandomizedConstruction:
    def __init__(self, graph, alpha=0.5, num_repeats=1):
        self.graph = graph
        self.pi = []  # Final order of nodes in V

        # Precompute edge weights per node for efficiency
        self.node_weights = defaultdict(int)
        self.node_connections = defaultdict(list)
        for u, v, weight in graph.edges:
            self.node_weights[v] += weight
            self.node_connections[v].append((u, weight))

        # Parameters for randomization
        self.alpha = alpha
        self.num_repeats = num_repeats

    def calculate_node_score(self, node, current_ordering):
        """
        Calculate a score for a node based on both its weights and potential crossings
        """
        if not current_ordering:
            return self.node_weights[node]

        score = 0
        position = len(current_ordering)

        # Consider existing edges
        for u, weight in self.node_connections[node]:
            score += weight  # Base weight

        return score

    def calculate_probabilities(self, candidates, current_ordering):
        """
        Calculate selection probabilities using Boltzmann distribution
        """
        scores = []
        for node in candidates:
            score = self.calculate_node_score(node, current_ordering)
            scores.append(score)

        scores = np.array(scores)

        # Convert scores to probabilities (lower scores = higher probability)
        max_score = max(scores) if scores.size > 0 else 1
        normalized_scores = scores / max_score  # Normalize to prevent overflow

        # Apply Boltzmann distribution with temperature
        probs = np.exp(-normalized_scores / self.alpha)
        probs = probs / np.sum(probs)

        return probs

    def greedy_randomized_construction(self):
        """
        Perform multiple iterations and return the best solution
        """
        best_solution = None
        best_cost = float('inf')

        for _ in range(self.num_repeats):
            solution = self.construct_single_solution()
            cost = cost_function_bit(self.graph, solution)

            if cost < best_cost:
                best_cost = cost
                best_solution = solution.copy()


        self.pi = best_solution
        return best_solution

    def construct_single_solution(self):
        """
        Construct a single solution using randomized greedy approach
        """
        # Reset in-degrees for this iteration
        in_degree = self.graph.in_degree.copy()
        current_ordering = []

        # Initialize candidates with in-degree 0 nodes
        candidates = deque([v for v in self.graph.V if in_degree[v] == 0])

        while candidates:
            candidates_list = list(candidates)

            if len(candidates_list) == 1:
                # If only one candidate, select it directly
                selected_node = candidates_list[0]
            else:
                # Calculate probabilities and select node
                probs = self.calculate_probabilities(candidates_list, current_ordering)
                selected_node = np.random.choice(candidates_list, p=probs)

            current_ordering.append(selected_node)
            candidates.remove(selected_node)

            # Update in-degrees and add new candidates
            for v_next in self.graph.constraints[selected_node]:
                in_degree[v_next] -= 1
                if in_degree[v_next] == 0:
                    candidates.append(v_next)

        return current_ordering

    def verify_solution(self):
        """
        Verify that the solution respects all constraints
        """
        if not self.pi:
            return False

        # Check if all nodes from V are present
        if set(self.pi) != set(self.graph.V):
            return False

        # Check if constraints are respected
        position = {node: idx for idx, node in enumerate(self.pi)}
        for v1 in self.graph.V:
            for v2 in self.graph.constraints[v1]:
                if position[v1] > position[v2]:
                    return False

        return True