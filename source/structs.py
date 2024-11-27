from collections import defaultdict

from helpers import *

class Graph:
    def __init__(self, U_size, V_size):
        self.U = list(range(1, U_size + 1))  # Nodes in fixed layer U
        self.V = list(range(U_size + 1, U_size + V_size + 1))  # Nodes in V
        self.edges = []  # List of edges (u, v, weight)
        self.constraints = defaultdict(list)  # Constraints as adjacency list for V
        self.in_degree = {v: 0 for v in self.V}   # Dictionary to store in-degree of nodes in V
        self.node_edges = defaultdict(list)  # Edges connected to each node
        self.node_edges_prefix_sum = defaultdict(lambda : [0] * U_size)  # Prefix sum of edge weights for each node
        self.node_edges_prefix_counts = defaultdict(lambda : [0] * U_size)  # Prefix count of edge weights for each node
        self.solution_costs = {}  # Store costs for each solution

    def add_node_U(self, node):
        self.U.append(node)

    def add_node_V(self, node):
        self.V.append(node)
        self.in_degree[node] = 0  # Initialize in-degree for nodes in V

    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))
        um = u - 1
        for i in range(um + 1, len(self.node_edges_prefix_sum[v])):
            self.node_edges_prefix_sum[v][i] += weight
            self.node_edges_prefix_counts[v][i] += 1

    def add_constraint(self, v1, v2):
        self.constraints[v1].append(v2)
        self.in_degree[v2] += 1  # Update in-degree due to precedence constraint
        
def load_instance(filename):
    with open(filename, 'r') as file:
        U_size, V_size, C_size, E_size = map(int, file.readline().split())
        graph = Graph(U_size, V_size)

        # Read constraints section
        line = file.readline().strip()
        while line != "#constraints":
            line = file.readline().strip()

        for _ in range(C_size):
            v, v_prime = map(int, file.readline().split())
            graph.add_constraint(v, v_prime)

        # Read edges section
        line = file.readline().strip()
        while line != "#edges":
            line = file.readline().strip()

        for _ in range(E_size):
            u, v, weight = file.readline().split()
            graph.add_edge(int(u), int(v), float(weight))

    return graph
    
    
class BinaryIndexedTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx, val):
        idx += 1  # Convert to 1-based indexing
        while idx <= self.size:
            self.tree[idx] += val
            idx += idx & (-idx)
    
    def query(self, idx):
        idx += 1  # Convert to 1-based indexing
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & (-idx)
        return total

def cost_function_bit(graph, permutation):
    """
    Implementation using Binary Indexed Tree for very large graphs.
    Time complexity: O(E log V) where E is number of edges and V is size of layer V.
    
    Args:
        graph: Graph object containing edges and weights
        permutation: List representing the ordering of nodes in layer V
    
    Returns:
        float: Total cost of crossings
    """
    position = {node: idx for idx, node in enumerate(permutation)}
    max_pos = len(permutation)
    
    # Sort edges by u value
    edges = [(u, position[v], w) for u, v, w in graph.edges]
    edges.sort()
    
    total_cost = 0
    bit = BinaryIndexedTree(max_pos) # BIT to count number of crossings
    weight_sum = BinaryIndexedTree(max_pos) # BIT to store sum of weights of edges
    
    # Process edges in order of increasing u
    for i, (u, pos_v, w) in enumerate(edges):
        # Count crossings with previous edges
        crossings = bit.query(max_pos - 1) - bit.query(pos_v)
        total_cost += crossings * w
        
        # Add contribution from weights of crossed edges
        weight_contribution = (weight_sum.query(max_pos - 1) - weight_sum.query(pos_v))
        total_cost += weight_contribution
        
        # Update BITs
        bit.update(pos_v, 1)
        weight_sum.update(pos_v, w)
    
    return total_cost