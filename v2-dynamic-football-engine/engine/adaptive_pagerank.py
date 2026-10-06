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
    scaled_transition_matrix = transition_matrix * player_prob_vector
    for cycle in range(max_loops):
        r_next = np.dot(scaled_transition_matrix, r) # Calculate the next round of scores based on current rank vector
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


def system_dependency(passing_matrix, player_names):
    """
    Simulates node-deletion by removing each player one-by-one and calculating
    the percentage drop in Global Network Efficiency using the Floyd-Warshall algorithm.
    """
    n = len(player_names)
    
    def calculate_global_efficiency(matrix):
        """Calculates network efficiency based on the inverse of shortest path distances."""
        size = matrix.shape[0]
        # Construct an adjacency cost matrix: high pass volume = low cost (short path)
        cost_matrix = np.full((size, size), np.inf)
        np.fill_diagonal(cost_matrix, 0)
        
        for i in range(size):
            for j in range(size):
                if matrix[i, j] > 0:
                    cost_matrix[i, j] = 1.0 / matrix[i, j]
        
        # Floyd-Warshall algorithm loop to find all-pairs shortest paths
        dist = cost_matrix.copy()
        for k in range(size):
            dist = np.minimum(dist, dist[:, np.newaxis, k] + dist[k, np.newaxis, :])
            
        # Compute the sum of inverse shortest distances across all pairs
        efficiency_sum = 0.0
        pairs_count = 0
        for i in range(size):
            for j in range(size):
                if i != j and dist[i, j] < np.inf and dist[i, j] > 0:
                    efficiency_sum += 1.0 / dist[i, j]
                    pairs_count += 1
                    
        return efficiency_sum / (size * (size - 1)) if size > 1 and pairs_count > 0 else 0.0

    # Calculate baseline global efficiency of the fully active 11-node network
    baseline_eff = calculate_global_efficiency(passing_matrix)
    if baseline_eff == 0:
        baseline_eff = 1.0
        
    dependency_scores = {}
    
    # Execute the node-deletion simulation loop
    for idx, target_player in enumerate(player_names):
        # Generate a sub-matrix completely removing the target player's row and column
        sub_matrix = np.delete(passing_matrix, idx, axis=0)
        sub_matrix = np.delete(sub_matrix, idx, axis=1)
        
        # Recalculate remaining network efficiency capabilities post-deletion
        subgraph_eff = calculate_global_efficiency(sub_matrix)
        
        # Calculate absolute network vulnerability drop percentage share
        eff_loss = (baseline_eff - subgraph_eff) / baseline_eff
        dependency_scores[target_player] = max(0.0, eff_loss)
        
    # Extract indices tracking the absolute structural system nexus hub
    most_dependent_player = max(dependency_scores, key=dependency_scores.get)
    max_network_loss = dependency_scores[most_dependent_player]
    
    return most_dependent_player, max_network_loss, dependency_scores


def filter_and_scale_telemetry(events_df, start_min, end_min, base_locations):
    """
    Slices raw match telemetry down to an interval, drops empty records, 
    and applies a strict, absolute hard cap of exactly the top 11 players.
    """
    # 1. Apply Dynamic Time-Binning constraints
    interval_events = events_df[
        (events_df['minute'] >= start_min) & (events_df['minute'] <= end_min)
    ].copy()
    
    # Absolute cleaning of empty string records to eliminate nan artifacts permanently
    interval_events = interval_events.dropna(subset=['player_name'])
    interval_events = interval_events[interval_events['player_name'].astype(str).str.lower() != 'nan']
    
    interval_duration = max(1.0, float(end_min - start_min))
    per_90_multiplier = 90.0 / interval_duration
    
    # 2. THE STRICT 11-PLAYER LOCK: Isolate ONLY the top 11 players by touch volume in this block
    player_event_volumes = interval_events['player_name'].value_counts()
    top_11_players = sorted(list(player_event_volumes.head(11).index))
    
    # 🚨 BRUTE FORCE GUARDRAIL: If dataset errors pull less or more, hard-slice it right here
    top_11_players = top_11_players[:11]
    
    # Filter the active events dataframe down strictly to ONLY our 11 isolated players
    interval_events = interval_events[
        interval_events['player_name'].isin(top_11_players) & 
        (interval_events['pass_recipient_name'].isin(top_11_players) | interval_events['pass_recipient_name'].isna())
    ]
    
    # Calculate average layout positions for ONLY the top 11 nodes
    player_positions = {}
    for player in top_11_players:
        player_df = interval_events[interval_events['player_name'] == player]
        if not player_df.empty and 'location' in player_df.columns:
            locs = np.array([loc for loc in player_df['location'].dropna() if len(loc) >= 2])
            if len(locs) > 0:
                player_positions[player] = (np.mean(locs[:, 0]), np.mean(locs[:, 1]))
            else:
                player_positions[player] = base_locations.get(player, (50.0, 50.0))
        else:
            player_positions[player] = base_locations.get(player, (50.0, 50.0))
            
    # 3. Handle Substitutions Elegantly (Prevent overlapping node layouts across the 11 nodes)
    resolved_positions = {}
    for idx, player in enumerate(top_11_players):
        coord = np.array(player_positions[player])
        for already_mapped in resolved_positions:
            prev_coord = np.array(resolved_positions[already_mapped])
            distance = np.linalg.norm(coord - prev_coord)
            if distance < 4.5:
                coord -= 5.0  # Shift left to clear the canvas overlap
                coord += 3.0  # Shift vertically to indicate tactical variation
        resolved_positions[player] = coord  # Store directly as a clean coordinate tuple
        
    # 4. Compute Per-90 Normalized Passing Weight Matrix
    player_to_idx = {player: i for i, player in enumerate(top_11_players)}
    n = len(top_11_players)
    raw_matrix = np.zeros((n, n))
    
    passes_df = interval_events[interval_events['type_name'] == 'Pass']
    for _, row in passes_df.iterrows():
        p_from = row['player_name']
        p_to = row['pass_recipient_name']
        if p_from in player_to_idx and p_to in player_to_idx:
            raw_matrix[player_to_idx[p_from], player_to_idx[p_to]] += 1.0
            
    normalized_matrix = raw_matrix * per_90_multiplier
    
    # Return ONLY the top 11 player parameters to app.py
    return normalized_matrix, top_11_players, resolved_positions


