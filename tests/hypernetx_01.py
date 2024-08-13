from typing import List
import hypernetx as hnx
import itertools as it
import matplotlib.pyplot as plt

H = hnx.Hypergraph([[1,2,3],[3,4,5],[1,5]])
print(type(H))
#H = hnx.Hypergraph([[0,1,2],[1,2,3],[2,3,4],[3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [11,12,13], [12,13,14]])
plt.subplots(figsize=(5,5))
hnx.draw(H)
plt.show()

M = H.incidence_matrix()
R = [list(x) for x in it.permutations(range(5))]

listOfIsomorphicHypergraphs: List[hnx.Hypergraph] = [hnx.Hypergraph.from_incidence_matrix(M[r].todense()) for r in R]

#for item in listOfIsomorphicHypergraphs:
#    print(item.edges)

# Function to print hypergraphs in the desired format
def print_hypergraph_edges(hypergraph: hnx.Hypergraph):
    # Using the .edges attribute to access the hyperedges
    hyperedges = [list(edge) for edge in hypergraph.edges]
    print(f"hnx.Hypergraph({hyperedges})")

# Iterating through each isomorphic hypergraph and printing its edges
for isomorphic_hypergraph in listOfIsomorphicHypergraphs:
    print_hypergraph_edges(isomorphic_hypergraph)
