import time
import os
from construction import DeterministicConstruction
from structs import load_instance, cost_function_bit
from construction import RandomizedConstruction

# Utility function to list files in a folder
def list_files_in_folder(folder_path):
    items = []
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # Check if it's a file and not a directory (excluding .ipynb_checkpoints)
            if os.path.isfile(item_path) and '.ipynb_checkpoints' not in item:
                items.append(item_path)
        return items
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return []
    
    
# Utility function for timing
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.delta = self.end - self.start
        
        
# Utility function to load and order an instance          
def load_and_order_instance(filename):
    graph = load_instance(filename)
    solution = DeterministicConstruction(graph)
    ordering = solution.greedy_construction()
    return graph, ordering

# Utility function to run multiple trials of the randomized construction
def run_multiple_trials(graph, num_trials=30):
    """
    Run multiple trials of the randomized construction and return average results.
    
    Args:
        graph: Graph instance
        num_trials: Number of trials to run (default: 30)
        
    Returns:
        tuple: (average_time, average_cost, min_cost, max_cost, std_dev_cost)
    """
    times = []
    costs = []
    
    for _ in range(num_trials):
        with Timer() as t:
            solution = RandomizedConstruction(graph)
            ordering = solution.greedy_randomized_construction()
        cost = cost_function_bit(graph, ordering)
        
        times.append(t.delta)
        costs.append(cost)
    
    # Calculate statistics
    avg_time = sum(times) / num_trials
    avg_cost = sum(costs) / num_trials
    min_cost = min(costs)
    max_cost = max(costs)
    
    # Calculate standard deviation for costs
    variance = sum((x - avg_cost) ** 2 for x in costs) / num_trials
    std_dev = variance ** 0.5
    
    return avg_time, avg_cost, min_cost, max_cost, std_dev



folder_path = "/content/competition_instances"
test_folder_path_small = "/content/test_instances/small"
test_folder_path_med = "/content/test_instances/medium"
test_folder_path_med_large = "/content/test_instances/medium_large"
test_folder_path_large = "/content/test_instances/large"


items = list_files_in_folder(folder_path)
items_test_small = list_files_in_folder(test_folder_path_small)
items_test_med = list_files_in_folder(test_folder_path_med)
items_test_med_large = list_files_in_folder(test_folder_path_med_large)
items_test_large = list_files_in_folder(test_folder_path_large)