# data/simple_graphs.py

# A linear chain network topology
linear_chain = {
    0: list([1]),
    1: list([2]),
    2: list([3]),
    3: list([])
}

# A cyclic network topology acting as a systemic spider trap
spider_trap = {
    0: list([1]),
    1: list([0]),
    2: list([0, 1])
}

# A hub-and-spoke star network topology
star_graph = {
    0: list([1, 2, 3, 4, 5]),
    1: list([]),
    2: list([]),
    3: list([]),
    4: list([]),
    5: list([])
}

