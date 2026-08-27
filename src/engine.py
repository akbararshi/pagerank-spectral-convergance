def run_matrix_power_iteration(nodes_count, connection_graph, damping_coeff=0.85, max_loops=100, precision=1e-6):
    vector_p = [1.0 / nodes_count] * nodes_count
    out_degrees = {src: len(targets) for src, targets in connection_graph.items() if len(targets) > 0}
    teleport_probability = (1.0 - damping_coeff) / nodes_count

    print(f"Executing simulation for damping coefficient = {damping_coeff}")

    for loop_idx in range(max_loops):
        next_vector_p = [0.0] * nodes_count
        dangling_sum = 0.0
        
        for node in range(nodes_count):
            if node in out_degrees:
                allocated_rank = vector_p[node] / out_degrees[node]
                for target in connection_graph[node]:
                    next_vector_p[target] += allocated_rank
            else:
                dangling_sum += vector_p[node] / nodes_count

        for node in range(nodes_count):
            next_vector_p[node] = teleport_probability + damping_coeff * (next_vector_p[node] + dangling_sum)

        vector_delta = sum(abs(next_vector_p[i] - vector_p[i]) for i in range(nodes_count))
        print(f"Loop {loop_idx + 1} - Absolute Matrix Delta: {vector_delta:.8f}")
        
        if vector_delta < precision:
            print(f"--> Matrix stabilized at state equilibrium in {loop_idx + 1} cycles.\n")
            return next_vector_p, loop_idx + 1
            
        vector_p = next_vector_p

    print("--> Upper loop iteration cap reached.\n")
    return vector_p, max_loops
