import pynauty

# test a cost function isomorphism, to check what happens in 2-spin variables which come out of 3-spin
# 0.125*Z[0]*Z[1]*Z[2] - 0.125*Z[0]*Z[1] - 0.125*Z[0]*Z[2] + 0.25*Z[0]*Z[3] - 0.125*Z[0] - 0.125*Z[1]*Z[2] + 0.125*Z[1] + 0.125*Z[2]*Z[4]*Z[5] - 0.125*Z[2]*Z[4] - 0.125*Z[2]*Z[5] + 0.25*Z[2] - 0.25*Z[3] - 0.125*Z[4]*Z[5] + 0.125*Z[4] + 0.125*Z[5]

# Define a graph
# The graph is represented as a dictionary where keys are vertex indices (starting from 0)
# and values are sets of vertices to which there is an edge.
# For example, the following graph is a triangle (3-cycle):
#graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}

#graph = {0: {6, }, 1: {6, }, 2: {6, 7}, 3: {}, 4: {7,}, 5: {7, }, 6: {0, 1, 2}, 7: {2, 4, 5}}
#graph = {0: {6, 1, 2, 3}, 1: {0, 2, 6}, 2: {0, 1, 4, 5, 6, 7}, 3: {0, }, 4: {2, 5, 7}, 5: {2, 4, 7}, 6: {0, 1, 2}, 7: {2, 4, 5}}
#graph = {0: {5,}, 1: {5,6}, 2: {5, 6, 7}, 3: {6, 7}, 4: {7,}, 5: {0,1,2}, 6: {1,2,3}, 7: {2,3,4}}
#graph = {0: {4,}, 1: {4, 5}, 2: {4, 5}, 3: {5}, 4: {0, 1, 2}, 5: {1,2,3}}
#graph = {0: {0,1,2,6}, 1: {0,3,6,7}, 2: {0, 2, 4, 6, 7, 8}, 3: {1, 3, 5, 7, 8, 9}, 4: {2,4,8,9}, 5: {3,4,5,9,}, 6: {0,1,2}, 7: {1,2,3}, 8: {2,3,4}, 9: {3,4,5}}

#graph = {0: {6,}, 1: {6,7}, 2: {6, 7, 8}, 3: {7, 8, 9}, 4: {8,9}, 5: {9,}, 6: {0,1,2}, 7: {1,2,3}, 8: {2,3,4}, 9: {3,4,5}}

graph = {0: {4,}, 1: {4, 5}, 2: {4, 5}, 3: {5}, 4: {0, 1, 2}, 5: {1,2,3}}

#nx_graph = nx.Graph()
#nx_graph.add_edges_from([(0, 6), (1, 6), (2, 6), (2, 7), (4, 7), (5, 7)])
#nx_graph.add_node(3)

# Convert graph to pynauty format
nauty_graph = pynauty.Graph(number_of_vertices=len(graph),
                            directed=False,
                            adjacency_dict=graph)

# Find automorphisms using Nauty
autgrp = pynauty.autgrp(nauty_graph)
print("Automorphism group size:", autgrp)

(gen, grpsize1, grpsize2, orbits, numorb) = pynauty.autgrp(nauty_graph)
print("gen", gen)
print("grpsize1", grpsize1)
print("grpsize2", grpsize2)
print("orbits", orbits)
print("numorb", numorb)

'''
(v3100_venv) ecamilleri@gluttony:~/projects/nauty$ python test-pynauty.py
graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
Automorphism group size: ([[0, 2, 1], [1, 0, 2]], 6.0, 0, [0, 0, 0], 1)

(v3100_venv) ecamilleri@gluttony:~/projects/nauty$ python test-pynauty.py
graph = {0: {6, }, 1: {6, }, 2: {6, 7}, 3: {}, 4: {7,}, 5: {7, }, 6: {0, 1, 2}, 7: {2, 4, 5}}
Automorphism group size: ([[0, 1, 2, 3, 5, 4, 6, 7], [1, 0, 2, 3, 4, 5, 6, 7], [4, 5, 2, 3, 0, 1, 7, 6]], 8.0, 0, [0, 0, 2, 3, 0, 0, 6, 6], 4)

Automorphism group size: ([[0, 1, 2, 3, 5, 4, 6, 7], [1, 0, 2, 3, 4, 5, 6, 7], [4, 5, 2, 3, 0, 1, 7, 6]], 8.0, 0, [0, 0, 2, 3, 0, 0, 6, 6], 4)
gen [[0, 1, 2, 3, 5, 4, 6, 7], [1, 0, 2, 3, 4, 5, 6, 7], [4, 5, 2, 3, 0, 1, 7, 6]]
grpsize1 8.0
grpsize2 0
orbits [0, 0, 2, 3, 0, 0, 6, 6]
numorb 4
'''