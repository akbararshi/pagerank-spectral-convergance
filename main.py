import json
from src.engine import run_matrix_power_iteration
from data.simple_graphs import spider_trap, star_graph

def execute_experiments():
    alpha_variants = [0.50, 0.70, 0.85, 0.95, 0.99]
    metrics_output = {"spider_trap": {}, "star_graph": {}}
    
    print("--- RUNNING SPIDER TRAP SIMULATIONS ---")
    for alpha in alpha_variants:
        _, dynamic_loops = run_matrix_power_iteration(3, spider_trap, damping_coeff=alpha)
        metrics_output["spider_trap"][str(alpha)] = dynamic_loops
        
    print("--- RUNNING STAR GRAPH SIMULATIONS ---")
    for alpha in alpha_variants:
        _, dynamic_loops = run_matrix_power_iteration(6, star_graph, damping_coeff=alpha)
        metrics_output["star_graph"][str(alpha)] = dynamic_loops

    with open("data/experiment_results.json", "w") as storage_file:
        json.dump(metrics_output, storage_file, indent=4)
    print("Success! Data logged to data/experiment_results.json")

if __name__ == "__main__":
    execute_experiments()
