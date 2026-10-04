import json
import matplotlib.pyplot as plt

# Load the newly unlocked, accurate experimental data
with open("data/experiment_results.json", "r") as f:
    data = json.load(f)

graph_types = ["spider_trap", "star_graph"]

for graph in graph_types:
    plt.figure(figsize=(10, 6))
    
    # Sort network sizes numerically for proper chronological axis scaling
    sizes = sorted([int(s) for s in data[graph].keys()])
    alphas = ["0.5", "0.7", "0.85", "0.95", "0.99"]
    
    # Extract values dynamically per sizing key
    for alpha in alphas:
        loops = [data[graph][str(size)][alpha] for size in sizes]
        plt.plot(sizes, loops, marker='o', linewidth=2.5, label=f"alpha = {alpha}")

    # Apply professional, academic styling
    clean_title = graph.replace('_', ' ').title()
    plt.title(f"PageRank Convergence Complexity Analysis ({clean_title})", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Network Scale (Total Nodes)", fontsize=12)
    plt.ylabel("Iterations to Reach Equilibrium (L1 Delta < 1e-6)", fontsize=12)
    
    # Logarithmic scale cleanly formats exponential growth zones across massive size margins
    plt.xscale('log')  
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Position legend cleanly away from intersecting curves
    plt.legend(fontsize=11, loc="upper left", frameon=True, facecolor='white', edgecolor='gainsboro')
    
    # Export to data folder
    plt.tight_layout()
    plt.savefig(f"data/{graph}_convergence_chart.png", dpi=300)
    print(f"📊 Chart successfully exported to: data/{graph}_convergence_chart.png")
    plt.close()
