import pynauty

# Define the graph
# For example, a triangle graph (3-cycle)
# The graph is represented as an adjacency list where each vertex is a key
# and its connected vertices are the values in the list.
# This represents a triangle: vertex 0 connected to 1 and 2, 1 to 0 and 2, and 2 to 0 and 1.

#graph = {0: [4], 1: [4, 5], 2: [4, 5], 3: [5], 4: [0, 1, 2], 5: [1, 2, 3]}
#graph = {0: [5], 1: [5, 6], 2: [5, 6, 7], 3: [6, 7], 4: [7], 5: [0, 1, 2], 6: [1, 2, 3], 7: [2, 3, 4]}
graph = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}

# Convert the adjacency list to pynauty's Graph format
n = len(graph) # Number of vertices
g = pynauty.Graph(number_of_vertices=n, directed=False, adjacency_dict=graph)

# Compute the automorphism group of the graph
auts = pynauty.autgrp(g)

# Print the result
print("Automorphisms:", auts)

print(auts[0])
