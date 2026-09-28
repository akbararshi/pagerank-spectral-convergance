import json
from src.engine import run_matrix_power_iteration

from data.simple_graphs import spider_trap, star_graph  # Fixed: matches your exact folder structure

def execute_experiments():

    # 1. Variables to test scale vs damping factor
    alpha_variants = [0.50, 0.70, 0.85, 0.95, 0.99]
    network_sizes = [10, 50, 100, 500, 1000, 5000] # Compares small vs large networks

    metrics_output = {
        "spider_trap": {},
        "star_graph": {}
    }


    # 2. Run Spider Trap Simulations
    print("--- RUNNING SPIDER TRAP SIMULATIONS ---")
    for size in network_sizes:
        metrics_output["spider_trap"][str(size)] = {}
        print(f"\nTesting Network Size: {size} nodes")
        

        current_spider_matrix = spider_trap(size)
        
        for alpha in alpha_variants:
            spider_scores, dynamic_loops = run_matrix_power_iteration(
                nodes_count=size,
                connection_graph=current_spider_matrix,
                damping_coeff=alpha
            )
            metrics_output["spider_trap"][str(size)][str(alpha)] = dynamic_loops
            print(f"  Alpha {alpha:.2f} -> Converged in {dynamic_loops} loops")

    # 3. Run Star Graph Simulations
    print("\n\n--- RUNNING STAR GRAPH SIMULATIONS ---")
    for size in network_sizes:
        metrics_output["star_graph"][str(size)] = {}
        print(f"\nTesting Network Size: {size} nodes")
        

        current_star_matrix = star_graph(size)
        
        for alpha in alpha_variants:
            star_scores, dynamic_loops = run_matrix_power_iteration(
                nodes_count=size,
                connection_graph=current_star_matrix,
                damping_coeff=alpha
            )
            metrics_output["star_graph"][str(size)][str(alpha)] = dynamic_loops
            print(f"  Alpha {alpha:.2f} -> Converged in {dynamic_loops} loops")



    # 4. Save the multi-dimensional results to JSON
    with open("data/experiment_results.json", "w") as storage_file:
        json.dump(metrics_output, storage_file, indent=4)
        

    print("\n[Success] Scaled simulation data logged to data/experiment_results.json")




if __name__ == "__main__":
    execute_experiments()



