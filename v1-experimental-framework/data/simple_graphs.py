# data/simple_graphs.py

def spider_trap(nodes_count):
    # Dynamically generates a closed cyclic network to test how the engine handles absorbant matrix states and infinite loop topology
    graph = {}
    
    # 1. Create a chain of nodes that flow downward to the next node in the sequence 
    for i in range(nodes_count - 1):
        graph[i] = [i + 1]
        
    # 2. Create a closed loop at the bottom of the network to trap rank
    # This forms a circular loop between the last two nodes in the network, creating a spider trap effect
    if nodes_count > 1:
        graph[nodes_count - 1] = [nodes_count - 2]
    else:
        graph[0] = [0] # Fallback self loop for a single node network to avoid dangling nodes
        
    return graph


def star_graph(nodes_count):
    # Dynamically generates a centralized hub-and-spoke star network topology. Node 0 acts as the central hub, while all other nodes point exclusively to it, and node 0 distributes rank outwards to all other nodes.
    graph = {}

    # Safety Fallback for a single node network to avoid dangling nodes
    if nodes_count <=1:
        return {0: [0]}  # Self loop for a single node network

    graph[0] = list(range(1, nodes_count))  # Central hub node points to all other nodes

    for i in range(1, nodes_count):
        graph[i] = [0]  # All other nodes point back to the central hub node

    return graph



