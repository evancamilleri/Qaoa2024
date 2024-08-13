import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

# Define the adjacency list
adj_list = {0: {5,}, 1: {5,6}, 2: {5, 6, 7}, 3: {6, 7}, 4: {7,}, 5: {0,1,2}, 6: {1,2,3}, 7: {2,3,4}}
#adj_list = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}
#adj_list = {0: [4], 1: [4, 5], 2: [4, 5], 3: [5], 4: [0, 1, 2], 5: [1, 2, 3]}

# Create the graph
G = nx.Graph(adj_list)

# Create a GraphMatcher object
matcher = GraphMatcher(G, G)

# Find and print the automorphisms
for iso in matcher.isomorphisms_iter():
    # Check if the automorphism is not the identity mapping
    if all(key == value for key, value in iso.items()):
        continue  # Skip the identity automorphism
    print(iso)
    '''
    # Filter out the identity mappings
    non_identity_mappings = [(y, x) for x, y in iso.items() if x != y]
    if non_identity_mappings:  # Only print if there are non-identity mappings
        print(non_identity_mappings)
        # Create a set of sorted tuples to automatically remove duplicates
        unique_pairs_set = {tuple(sorted(pair)) for pair in non_identity_mappings}
        # Convert the set back into a list of tuples
        unique_pairs = list(unique_pairs_set)
        print(unique_pairs)
    '''

