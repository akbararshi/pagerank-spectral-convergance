import numpy as np
from operator import itemgetter
def sports_adaptive_pagerank(passing_matrix, player_names, max_loops = 100, vector_delta=1e-6):
    """
    Custom PageRank algorith for sports passing networks. Dynamically scales the damping factor upward to prevent the standard 0.85 damping factor from diluting
    the true structual playmaking authority.
    """
    #Initializes core matrix structure
    P = np.array(passing_matrix, dtype=float)
    n= P.shape[0]
    
    #Calculate total successful passes made by each player
    player_pass_total = np.sum(P, axis=0)

    #THE ADAPTIVE DAMPING ENGINE
    #Graph Density = Active Passing Lanes/Max Possible Passing Lanes
    graph_density = np.count_nonzero(P) / (n * (n - 1))
    damping_coeff = 0.85 + 0.13/(1 + np.exp(-10*(graph_density-0.45)))

    
    player_prob_vector = np.zeros(n)
    avg_passes = np.mean(player_pass_total[player_pass_total >0])
    for i in range(n):
        if player_pass_total[i] >0:
            activity_factor = player_pass_total[i]/avg_passes
            player_prob_vector[i]=damping_coeff * (1-(0.05/(1.0+activity_factor)))
        else:
            player_prob_vector[i] = 0.0

    # Normalize Transition Matrix, General Rule: (Column = From | Row = To)
    transition_matrix = np.zeros_like(P)
    for i in range(n):
        if player_pass_total[i]>0:
            transition_matrix[:,i] = P[:,i]/player_pass_total[i]
        else:
            transition_matrix[:,i]= np.ones(n)/n    

    v=np.ones(n)/n #uniform teleportation baseline
    r=np.copy(v) #initialize rank vector across the squad
    for cycle in range(max_loops):
        r_next = np.zeros(n) #clear the scoreboard for the next cycle
        for j in range(n):
            r_next += r[j] * player_prob_vector[j] * transition_matrix[:,j]

        remaining_teleportation = 1.0 -np.sum(r_next)
        r_next += remaining_teleportation * v #redistribute the remaining teleportation probability across all players

    # Calculate the margin of error between the current and previous round of scores to see if the matrix has stabilized
        margin_of_error = np.linalg.norm(r_next - r, 1)
        if margin_of_error < vector_delta:
            break
        r = r_next #Update the rank vector for the next cycle

    # Return the final player rankings
    rankings = {player_names[i]: r[i] for i in range(n)}
    sorted_rankings = dict(sorted(rankings.items(), key=itemgetter(1), reverse=True))
    return sorted_rankings, damping_coeff, graph_density


#Sample Test Case for the Adaptive PageRank Engine
if __name__ == "__main__":
    # Test Data: A standard 4-player football passing diamond [CB, DM, CM, ST]
    # Rows = Passed TO | Columns = Passed FROM
    mock_football_matrix = [
        [0.0,  5.0,  2.0,  1.0],  # Passes received TO Center-Back (CB)
        [12.0, 0.0,  8.0,  3.0],  # Passes received TO Defensive-Midfield (DM)
        [4.0,  15.0, 0.0,  5.0],  # Passes received TO Attacking-Midfield (CM)
        [1.0,  2.0,  9.0,  0.0]   # Passes received TO Striker (ST)
    ]
    player_squad_names = ["Center-Back", "Def-Midfield", "Att-Midfield", "Striker"]
    
    try:
        # Run your newly verified calculation loop
        final_scores, applied_coeff, match_density = sports_adaptive_pagerank(
            passing_matrix=mock_football_matrix,
            player_names=player_squad_names,
            max_loops=150,
            vector_delta=1e-6
        )
        
        print("\n🏆 TACTICAL PULSE ADAPTIVE ENGINE INITIALIZED SUCCESSFULLY!")
        print("-" * 55)
        print(f"Calculated Match Graph Density: {match_density:.3f}")
        print(f"Automated Sigmoid Damping Vector Center: {applied_coeff:.4f}")
        print("-" * 55)
        print("Final Structural Squad Rankings:")
        for rank, (player, score) in enumerate(final_scores.items(), 1):
            print(f"  {rank}. {player:<15} │ Structural Weight Score: {score:.4f}")
            
    except Exception as e:
        print(f"\n❌ Execution Terminal Crash Error: {str(e)}")
