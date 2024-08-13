import pynauty

# Define the graph
# For example, a triangle graph (3-cycle)
# The graph is represented as an adjacency list where each vertex is a key
# and its connected vertices are the values in the list.
# This represents a triangle: vertex 0 connected to 1 and 2, 1 to 0 and 2, and 2 to 0 and 1.

graph = {0: [4], 1: [4, 5], 2: [4, 5], 3: [5], 4: [0, 1, 2], 5: [1, 2, 3]}
#graph = {0: [5], 1: [5, 6], 2: [5, 6, 7], 3: [6, 7], 4: [7], 5: [0, 1, 2], 6: [1, 2, 3], 7: [2, 3, 4]}
#graph = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}


def compose_permutations(perm1, perm2):
    """Compose two permutations to produce a new permutation."""
    return [perm1[i] for i in perm2]

def apply_permutation_to_graph(graph, permutation):
    """Applies a permutation to an adjacency list representation of a graph."""
    # Apply permutation to vertices
    permuted_graph = {permutation[vertex]: [] for vertex in graph}
    # Apply permutation to edges
    for vertex, edges in graph.items():
        permuted_vertex = permutation[vertex]
        permuted_graph[permuted_vertex] = [permutation[edge] for edge in edges]
    return permuted_graph

def is_permutation_unique(permutation, permutations_list):
    """Check if a permutation is unique among a list of permutations."""
    return permutation not in permutations_list

def generate_all_automorphisms(generators):
    """Generates all automorphisms from a set of generator permutations."""
    all_automorphisms = set(tuple(g) for g in generators)  # Use tuples for hashability
    new_automorphisms = set(tuple(g) for g in generators)

    while new_automorphisms:
        current_generation = set()
        for gen1 in all_automorphisms:
            for gen2 in generators:
                new_perm = compose_permutations(gen1, gen2)
                if tuple(new_perm) not in all_automorphisms:
                    current_generation.add(tuple(new_perm))
        new_automorphisms = current_generation
        all_automorphisms.update(new_automorphisms)

    return [list(perm) for perm in all_automorphisms]

# -----------------------------------------------------------------------------------

# Convert the adjacency list to pynauty's Graph format
n = len(graph) # Number of vertices
g = pynauty.Graph(number_of_vertices=n, directed=False, adjacency_dict=graph)

# Compute the automorphism group of the graph
auts = pynauty.autgrp(g)

# Print the result
print("Automorphisms:", auts)
print("Generators:", auts[0])

generators = auts[0]

# -----------------------------------------------------------------------------------

# Example usage
# Generators for a simple graph's automorphism group
#generators = [[0, 2, 1, 3], [1, 0, 2, 3]]  # Example generators

all_automorphisms = generate_all_automorphisms(generators)
print("All automorphisms:", all_automorphisms)
