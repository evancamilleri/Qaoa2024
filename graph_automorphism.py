import networkx as nx
from igraph import Graph as IGraph
import matplotlib.pyplot as plt
from sympy.combinatorics import Permutation, PermutationGroup
from itertools import permutations

# Function to convert a NetworkX graph to an igraph graph
def convert_networkx_to_igraph(nx_graph):
    mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    g = IGraph(len(nx_graph))
    g.add_edges([(mapping[u], mapping[v]) for u, v in nx_graph.edges()])
    return g, mapping


def get_automorphism_groups(nx_graph: nx.Graph):
    # Convert the NetworkX graph to an igraph graph
    igraph_graph, mapping = convert_networkx_to_igraph(nx_graph)

    # Find automorphisms
    automorphisms = igraph_graph.get_isomorphisms_vf2(igraph_graph)

    # Convert automorphisms back to the original node labels
    automorphisms_labels = [[list(mapping.keys())[list(mapping.values()).index(i)] for i in automorphism] for
                            automorphism in automorphisms]

    # Print the automorphisms
    #print("Automorphisms (in terms of original node labels):")
    #for automorphism in automorphisms_labels:
    #    print(automorphism)

    # Create permutations from the zero-indexed automorphisms
    permutations = [Permutation(automorphism) for automorphism in automorphisms_labels]

    # Create the permutation group
    group = PermutationGroup(*permutations)

    # Now `group` contains the group of automorphisms
    return group


def select_automorphism(nx_graph: nx.Graph):
    group = get_automorphism_groups(nx_graph)

    # Initialize variables to store the desired permutation
    max_degree = -1
    min_cycles = float('inf')
    selected_permutation = None

    for p in group:
        #print(type(p))
        # Count the number of points the permutation moves (degree)
        degree = sum(1 for i in range(p.size) if p(i) != i)
        # Count the number of non-singleton cycles
        non_singleton_cycles = sum(1 for cycle in p.cyclic_form if len(cycle) > 1)

        # Check if this permutation has a higher degree and fewer non-singleton cycles
        if degree > max_degree or (degree == max_degree and non_singleton_cycles < min_cycles):
            max_degree = degree
            min_cycles = non_singleton_cycles
            selected_permutation = p

    return selected_permutation


def find_vertex_orbits(nx_graph):
    print('***IGRAPH find_vertex_orbits***')
    permutation: Permutation = select_automorphism(nx_graph)
    n = nx_graph.number_of_nodes()

    # Get the cyclic form of the permutation
    cycles = permutation.cyclic_form

    # Create a set of all elements included in the cycles
    included_elements = set(element for cycle in cycles for element in cycle)

    # Convert each cycle to a tuple
    cycle_tuples = [tuple(cycle) for cycle in cycles]

    # Add single-element cycles for elements not included in any cycle
    single_element_cycles = [(i,) for i in range(n) if i not in included_elements]

    # Combine and return all cycles
    return cycle_tuples + single_element_cycles


def find_edge_orbits(nx_graph, vertex_automorphisms):
    print('-'*25)
    print(nx_graph)
    print(nx_graph.nodes)
    print(nx_graph.edges)
    quit()
    print('***IGRAPH find_edge_orbits***')
    # Create a dictionary for vertex automorphisms for easy lookup
    automorphism_dict = {v: v for v in nx_graph.nodes()}  # Start with identity mapping
    for orbit in vertex_automorphisms:
        for v in orbit:
            automorphism_dict[v] = orbit[0]  # Map each vertex to the first vertex in its orbit

    # List all edges of the graph
    edges = list(nx_graph.edges())

    # Group edges into orbits
    edge_orbits = {}
    for edge in edges:
        # Apply automorphism to the edge
        mapped_edge = tuple(sorted((automorphism_dict[edge[0]], automorphism_dict[edge[1]])))

        # Add to the corresponding orbit
        edge_orbits.setdefault(mapped_edge, set()).add(edge)

    # Extract unique orbits
    unique_orbits = list(edge_orbits.values())

    return unique_orbits


def find_edge_cycle_index(edge_cycle: list, edge: tuple):
    for index, edge_set in enumerate(edge_cycle):
        if edge in edge_set or tuple(reversed(edge)) in edge_set:
            return index

    print(f'edge {edge} not found in edge cycle {edge_cycle}')
    return -1  # Return -1 if the edge is not found


def find_vertex_index(vertex_cycle, vertex):
    for index, vertex_tuple in enumerate(vertex_cycle):
        if vertex in vertex_tuple:
            return index

    print(f'vertex {vertex} not found in vertex cycle {vertex_cycle}')
    return -1  # Return -1 if the vertex is not found


'''
# Create a NetworkX graph (Example: a triangle graph)
#nx_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
nx_graph = nx.Graph([(0, 1), (1, 2), (2, 3), (1, 3), (3, 4)])
#nx_graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])

#group = get_automorphism_groups(nx_graph)
#print(group)

cycles: Permutation = select_automorphism(nx_graph)
print(cycles)
print(type(cycles))

# Example usage
vertex_orbits = find_vertex_orbits(nx_graph)
print(vertex_orbits)
print(len(vertex_orbits))

edge_orbits = find_edge_orbits(nx_graph, vertex_orbits)
print(edge_orbits)
print(len(edge_orbits))
print(type(edge_orbits))


# Display the graph using matplotlib
plt.figure(figsize=(8, 6))
nx.draw(nx_graph, with_labels=True, font_weight='bold', node_color='skyblue', node_size=700, font_size=18)
plt.show()
'''


import networkx as nx
import itertools

def is_automorphism(G, mapping):
    for u, v in G.edges():
        if (mapping[u], mapping[v]) not in G.edges() and (mapping[v], mapping[u]) not in G.edges():
            return False
    return True

def find_automorphisms(G):
    automorphisms = []
    nodes = list(G.nodes())
    for permutation in itertools.permutations(nodes):
        mapping = {nodes[i]: permutation[i] for i in range(len(nodes))}
        if is_automorphism(G, mapping):
            automorphisms.append(mapping)
    return automorphisms

'''
# Example usage
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0), (2,3)])
#automorphisms = find_automorphisms(G)
#print(automorphisms)

vertex_cycle = find_vertex_orbits(G)
print(vertex_cycle)

edge_cycle = find_edge_orbits(G, vertex_cycle)
print(edge_cycle)

cycle_index = find_edge_cycle_index(edge_cycle, (0, 1))
print(cycle_index)
cycle_index = find_edge_cycle_index(edge_cycle, (1, 2))
print(cycle_index)
cycle_index = find_edge_cycle_index(edge_cycle, (2, 0))
print(cycle_index)
cycle_index = find_edge_cycle_index(edge_cycle, (2, 3))
print(cycle_index)

vertex_index = find_vertex_index(vertex_cycle, 2)
print(vertex_index)
'''
