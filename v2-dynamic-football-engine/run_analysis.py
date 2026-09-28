import numpy as np
from data.fetcher import get_match_passing_matrix
from engine.adaptive_pagerank import sports_adaptive_pagerank

def run_comparative_study(match_id: int, team_name: str):
    """
    Main pipeline: Pulls a real match, calculates standard 0.85 PageRank,
    calculates your custom Adaptive Sigmoid PageRank, and displays 
    a side-by-side comparison chart for your research paper.
    """
    # 1. Fetch real match matrix via your data layer
    try:
        matrix, players = get_match_passing_matrix(match_id, team_name)
    except Exception as e:
        print(f"❌ Failed to fetch match data: {e}")
        return

    # 2. RUN YOUR UPGRADED ADAPTIVE ENGINE (Sigmoid-Scaled)
    adaptive_scores, applied_coeff, density = sports_adaptive_pagerank(
        passing_matrix=matrix,
        player_names=players,
        max_loops=150,
        vector_delta=1e-6
    )

    # 3. RUN STANDARD BASELINE PAGERANK (Fixed d=0.85) FOR COMPARISON
    # To isolate your algorithm's impact, we run the same loop but freeze the slider at 0.85
    n = len(players)
    player_pass_total = np.sum(matrix, axis=0)
    
    # Standard 0.85 fixed vector setup
    d_standard = np.full(n, 0.85)
    for j in range(n):
        if player_pass_total[j] == 0: d_standard[j] = 0.0
        
    transition_matrix = np.zeros_like(matrix)
    for j in range(n):
        if player_pass_total[j] > 0:
            transition_matrix[:, j] = matrix[:, j] / player_pass_total[j]
        else:
            transition_matrix[:, j] = np.ones(n) / n

    r = np.ones(n) / n
    v = np.ones(n) / n
    for _ in range(150):
        r_next = np.zeros(n)
        for j in range(n):
            r_next += r[j] * d_standard[j] * transition_matrix[:, j]
        r_next += (1.0 - np.sum(r_next)) * v
        if np.linalg.norm(r_next - r, 1) < 1e-6:
            r = r_next
            break
        r = r_next
        
    standard_scores = {players[i]: float(r[i]) for i in range(n)}

    # 4. PRINT FORMATTED DATA CHART FOR YOUR RESEARCH PAPER
    print("\n" + "="*70)
    print(f" ACADEMIC COMPARISON DATA REPORT: {team_name.upper()} (ID: {match_id})")
    print(f" Calculated Match Density (Rho): {density:.3f}")
    print(f" Custom Adaptive Sigmoid Damping Center: {applied_coeff:.4f}")
    print("="*70)
    print(f"{'PLAYER SQUAD NAME':<25} │ {'RAW PASSES':<10} │ {'STD 0.85 SCORE':<15} │ {'YOUR CUSTOM SCORE':<15}")
    print("-" * 70)
    
    # Sort and display by your custom engine values
    for player in adaptive_scores.keys():
        idx = players.index(player)
        raw_vol = int(player_pass_total[idx])
        std_val = standard_scores[player]
        cust_val = adaptive_scores[player]
        print(f"{player:<25} │ {raw_vol:<10} │ {std_val:.4f} │ {cust_val:.4f}")
    print("="*70)

if __name__ == "__main__":
    # Test on Barcelona vs Alavés (Messi masterclass)
    run_comparative_study(15946, "Barcelona")
