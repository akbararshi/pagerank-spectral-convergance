import json
from src.engine import run_matrix_power_iteration
from data.simple_graphs import spider_trap, star_graph 

def execute_experiments():
    # Testing different alpha values to see how the damping coefficient affects convergence speed and stability of the PageRank algorithm
    alpha_variants = [0.50, 0.70, 0.85, 0.95, 0.99]
    metrics_output = {"spider_trap": {}, "star_graph": {}}
    
    # Establishing the number of nodes for each graph type to ensure consistent testing conditions across different damping coefficients
    SPIDER_TRAP_NODES = 4
    STAR_GRAPH_NODES = 6
    
    print("--- RUNNING SPIDER TRAP SIMULATIONS ---")
    for alpha in alpha_variants:
        current_spider_matrix = spider_trap(SPIDER_TRAP_NODES)
        
        # Using "spider_scores" here to capture the final probability distribution scores for each node in the spider trap graph 
        spider_scores, dynamic_loops = run_matrix_power_iteration(
            nodes_count=SPIDER_TRAP_NODES, 
            connection_graph=current_spider_matrix, 
            damping_coeff=alpha
        )
        metrics_output["spider_trap"][str(alpha)] = dynamic_loops

        # Sort spider trap scores in descending order to identify the nodes that are absorbing the most rank in the network 
        spider_leaderboard = sorted(enumerate(spider_scores), key=lambda x:x[1], reverse=True)
        print(f"---Spider Trap Leaderboard for Alpha {alpha}---")
        for rank_pos, (page_num, score_val) in enumerate(spider_leaderboard):
            print(f"Rank {rank_pos + 1}: Page {page_num} - Score: {score_val:.4f}")
        print("-" * 40)

    print("\n--- RUNNING STAR GRAPH SIMULATIONS ---")
    for alpha in alpha_variants:
        current_star_matrix = star_graph(STAR_GRAPH_NODES)
        
        # Using "star_scores" here to capture the final probability distribution scores for each node in the star graph
        star_scores, dynamic_loops = run_matrix_power_iteration(
            nodes_count=STAR_GRAPH_NODES, 
            connection_graph=current_star_matrix, 
            damping_coeff=alpha
        )
        metrics_output["star_graph"][str(alpha)] = dynamic_loops

        # Sort star graph scores in descending order to identify the nodes that are absorbing the most rank in the network
        star_leaderboard = sorted(enumerate(star_scores), key=lambda x:x[1], reverse=True)
        print(f"---Star Graph Leaderboard for Alpha {alpha}---")
        for rank_pos, (page_num, score_val) in enumerate(star_leaderboard):
            print(f"Rank {rank_pos + 1}: Page {page_num} - Score: {score_val:.4f}")
        print("-" * 40)
        
    # Save the results to a JSON file so we can use it later
    with open("data/experiment_results.json", "w") as storage_file:
        json.dump(metrics_output, storage_file, indent=4)
        
    print("\nSuccess! Data logged to data/experiment_results.json")

# Main entry point to run the script
if __name__ == "__main__":
    execute_experiments()
