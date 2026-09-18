def run_matrix_power_iteration(nodes_count, connection_graph, damping_coeff=0.85, max_loops=100, precision=1e-6):
    # Initialize the score vector with equal probability for each node
    vector_p = [1.0/nodes_count]*nodes_count
    # Pre-calculate the out-degree (link count) for each node that has outgoing links to optimize score distribution
    out_degrees = {src: len(targets) for src, targets in connection_graph.items() if len(targets) > 0}
    # Calculate the teleportation probability for each page based on the damping coefficient
    teleport_probability = (1.0 - damping_coeff) / nodes_count

    print(f"Executing simulation for damping coefficient = {damping_coeff}")

    for cycle in range(max_loops):
        # Create a fresh array to hold the next iteration of scores at the start of each round
        next_vector_p = [0.0] * nodes_count
        # Keep track of the total score that will be distributed to all pages from dangling nodes (nodes with no outbound links)
        dangling_sum = 0.0

        # Loop through each node and distribute its score to outbound links or add it to dangling sum if the node has no outbound links
        for node in range(nodes_count):
            if node in out_degrees:
                allocated_rank = vector_p[node] / out_degrees[node]
                for target in connection_graph[node]:
                    next_vector_p[target] += allocated_rank
            else:
                dangling_sum += vector_p[node] / nodes_count
        # Combine the teleporation probability, the damping factor, and the dangling sum to calculate the next iteration of scores for each node
        for node in range(nodes_count):
            next_vector_p[node] = teleport_probability + damping_coeff * (next_vector_p[node] + dangling_sum)

        # Calculate the margin of error between the current and previous round of scores to see if the matrix has stabilized
        vector_delta = sum(abs(next_vector_p[i] - vector_p[i]) for i in range(nodes_count))
        print(f"Loop {cycle + 1} - Absolute Matrix Delta: {vector_delta:.8f}")
        # If the margin of error is below the precision threshold, we can conclude that the matrix has stabilized and return the current scores and number of cycles it took to achieve equilibrium
        if vector_delta < precision:
            print(f"--> Matrix stabilized at state equilibrium in {cycle + 1} cycles.\n")
            return next_vector_p, cycle + 1
        #  Overwrite the current scores with the next iteration of scores to prepare for the next round      
        vector_p = next_vector_p

    print("--> Upper loop iteration cap reached.\n")
    return vector_p, max_loops
