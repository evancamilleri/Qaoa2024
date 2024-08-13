import hypernetx as hnx
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import matplotlib.pyplot as plt
from evan_library import AngleStudy
from itertools import combinations
import re


def find_vertex_orbits(cost_function: str, graph_name: str = ''):
    print('***GENERIC*** find_vertex_orbits')
    #edge_list = create_edge_list_from_z(cost_function)
    #H = hnx.Hypergraph(edge_list)
    #hypernodes = H.number_of_nodes()

    # Separate terms in an array, accounting for '-' as a separate operation
    terms = cost_function.replace(" - ", " + -").split(" + ")
    # Filter and extract Z numbers from terms with only one Z term
    single_z_terms = [term for term in terms if term.count("Z") == 1]
    z_numbers = [term.split('[')[1].split(']')[0] for term in single_z_terms]
    #print(z_numbers)

    #full_list = [(i,) for i in range(hypernodes)]
    tuple_list = [(int(num),) for num in z_numbers]

    return tuple_list


def find_edge_orbits_coef(cost_function: str, edge_orbits: list):
    # Clean the cost_function string
    cost_function = cost_function.replace(" ", "")

    # Correctly use regex to split the cost_function by + and -, keeping the sign with the term
    # The adjusted pattern looks for any + or - that is not immediately after an opening bracket [
    terms = re.split(r'(?<!\[)(?=[+-])', cost_function)

    # The resulting terms should be split correctly, with each sign retained
    #print(terms)

    ###########################

    # Prepare to store the new structure similar to edge_orbits but with coefficients
    coef_orbits = []

    # Iterate over each set in edge_orbits
    for orbit_set in edge_orbits:
        coef_set = set()

        for orbit_tuple in orbit_set:
            # For each term in terms, find matching z indices and extract coefficient
            for term in terms:
                # Extract all Z indices from the term
                z_indices = [int(match.group(1)) for match in re.finditer(r'Z\[(\d+)\]', term)]

                # Attempt to extract a coefficient, defaulting to 1.0 if not found or empty
                coefficient_match = re.match(r'^([+-]?\d*\.?\d*)(?=\*)', term)
                coefficient = float(coefficient_match.group(1)) if coefficient_match and coefficient_match.group(1) else 1.0

                # Check if current orbit tuple matches z indices from term
                if set(orbit_tuple) == set(z_indices):
                    coef_set.add(coefficient)
                    break  # Break if match is found to avoid duplicate additions

        coef_orbits.append(coef_set)

    #print(coef_orbits)

    ###########################

    # Initialize a list to store the results (averages or the single value)
    averaged_coef_orbits = []

    # Loop through each set in coef_orbits
    for coef_set in coef_orbits:
        # Check if there is more than one value in the set
        if len(coef_set) > 1:
            # Calculate the average if there are multiple coefficients
            #average = sum(coef_set) / len(coef_set)
            average = sum(abs(coef) for coef in coef_set) / len(coef_set)
            averaged_coef_orbits.append({average})  # Add the average as a single-item set
        else:
            # If there's only one coefficient, just add it as is
            averaged_coef_orbits.append(coef_set)

    # Display the resulting list
    print(averaged_coef_orbits)
    #quit()
    values_list = [list(s)[0] for s in averaged_coef_orbits]
    return values_list


def find_edge_orbits_0(hnx_graph: hnx.Hypergraph, hnx_x: hnx.Hypergraph, angle_study: AngleStudy, show_plot = False):

    # [1] create a bipartite incidence graph of the hypergraph
    nx_bipartite = nx.Graph()
    hypernodes = hnx_graph.number_of_nodes()
    hyperedges = hnx_graph.number_of_edges()
    for edge in hnx_graph.edges:
        nodes = hnx_graph.edges[edge]
        #print(f"Edge {edge} contains nodes: {list(nodes)}")
        for node in hnx_graph.edges[edge]:
            nx_bipartite.add_edge(edge + hypernodes, node)
    #print(nx_bipartite)
    #print(nx_bipartite.nodes)
    #print(nx_bipartite.edges)

    # [2] print the hypergraph
    if show_plot:
        if hnx_x is not None:
            hnx.drawing.draw(hnx_x)
            plt.title('HyperNetX Binary Function Hypergraph')
            plt.show()

        # Set up a matplotlib figure and axes
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # Plot the HyperNetX hypergraph on the second subplot
        hnx.drawing.draw(hnx_graph, ax=axs[0])
        axs[0].set_title('HyperNetX ' + ('Binary Function' if angle_study == AngleStudy.automorphism_local else 'Spin Function') + ' Hypergraph')
        # Plot the NetworkX graph on the first subplot
        nx.draw(nx_bipartite, ax=axs[1], with_labels=True, node_color='skyblue')
        axs[1].set_title('NetworkX Bipartite Incidence Graph')
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    # [3] find the automorphisms in graph

    # Create a GraphMatcher object
    matcher = GraphMatcher(nx_bipartite, nx_bipartite)

    '''
    #TODO: trying to find best AM
    for index, iso in enumerate(matcher.isomorphisms_iter()):
        print(iso)
        mismatch_count = 0
        for key, value in iso.items():
            mismatch_count += 1 if key != value else 0
        print(index, mismatch_count)
    quit()
    '''

    # Find and print the automorphisms
    for iso in matcher.isomorphisms_iter():
        # Check if the automorphism is not the identity mapping
        if all(key == value for key, value in iso.items()):
            continue  # Skip the identity automorphism
        #print(iso)
        edge_orbit0 = []
        seen = set()
        for key, value in iso.items():
            sorted_pair = tuple(sorted((key, value)))  # Create a sorted tuple from the key-value pair
            if sorted_pair not in seen:
                seen.add(sorted_pair)
                if key - hypernodes >= 0:
                    #print(key, value)
                    if key == value:
                        # Use a tuple instead of a set
                        edge_orbit0.append((hnx_graph.edges[key - hypernodes],))
                    else:
                        # Use a tuple instead of a set
                        edge_orbit0.append((hnx_graph.edges[key - hypernodes], hnx_graph.edges[value - hypernodes]))

        edge_orbit = [{tuple(sublist) for sublist in item} for item in edge_orbit0]
        #print(edge_orbit)

        if angle_study == AngleStudy.automorphism_local:
            # now add pairs in each edge
            for edge in hnx_graph.edges:
                nodes = hnx_graph.edges[edge]
                #print(nodes)
                # Generate all 2-combinations (pairs) of nodes
                pairs = set(combinations(nodes, 2))
                edge_orbit.append(pairs)

        print(iso)
        print(edge_orbit)
        #quit()
        #6
        #edge_orbit = [{(0, 1, 2), (1, 2, 3), (0, 1, 5), (0, 4, 5), (2, 3, 4), (3, 4, 5)}, {(0, 1), (4, 5), (0, 2), (2, 4), (0, 4), (1, 3), (1, 5), (3, 5)}, {(0,), (1,), (2,), (3,), (4,), (5,)}]
        #12
        #edge_orbit = [{(0, 1, 11), (0, 1, 2), (0, 10, 11), (10, 11, 9), (1, 2, 3), (2, 3, 4), (10, 8, 9), (7, 8, 9), (3, 4, 5), (4, 5, 6), (6, 7, 8), (5, 6, 7)}, {(0, 1), (10, 11), (0, 2), (0, 10), (1, 11), (1, 3), (11, 9), (2, 4), (10, 8), (7, 9), (3, 5), (6, 8), (4, 6), (5, 7)}, {(0,), (11,), (1,), (2,), (10,), (3,), (9,), (8,), (4,), (7,), (5,), (6,)}]

        return edge_orbit


def find_edge_orbits(angle_study: AngleStudy, cost_function: str, binary_expression: str, show_plot = False):
    edge_list = create_edge_list(angle_study, cost_function, binary_expression)

    if show_plot and angle_study == AngleStudy.automorphism_global:
        Hx = hnx.Hypergraph(create_edge_list_from_x(binary_expression))
    else:
        Hx = None

    H = hnx.Hypergraph(edge_list)
    edge_orbits = find_edge_orbits_0(H, Hx, angle_study, show_plot)
    if edge_orbits is None:
        return None, None
    else:
        return edge_orbits, find_edge_orbits_coef(cost_function, edge_orbits)


def create_edge_list_from_z(cost_function: str):
    # Split the input string into terms based on '+' and '-' operators, accounting for spacing
    # The replace method is used to handle subtraction by considering it as adding a negative term
    terms = cost_function.replace(" - ", " + -").split(" + ")

    # Initialize a list to store the extracted Z indices for each term
    z_indices_list = []

    # Iterate over each term to extract Z indices
    for term in terms:
        # Find all occurrences of 'Z[n]' and extract 'n'
        # The condition z.startswith("Z[") checks if a substring starts with 'Z[' indicating a Z variable
        # int(z[2:-1]) converts the extracted index 'n' from string to integer
        z_indices = [int(z[2:-1]) for z in term.split("*") if z.startswith("Z[")]
        # Append the list of indices for this term to the overall list
        z_indices_list.append(z_indices)

    # Print the final list of Z indices
    #print('>>>>>>>>>>', z_indices_list)
    return z_indices_list


def create_edge_list_from_x(binary_expression: str):
    ## Split the input string into terms based on '+' and '-' operators, accounting for spacing
    terms = binary_expression.replace(" - ", " + -").split(" + ")

    # Initialize a list to store the extracted x indices for each term
    x_indices_list = []

    # Use a regular expression to find patterns of x(xn)
    pattern = re.compile(r'x\(x(\d+)\)')

    # Iterate over each term to extract x indices
    for term in terms:
        # Use regular expression to find all occurrences and extract 'n'
        x_indices = [int(match.group(1)) for match in pattern.finditer(term)]
        # Append the list of indices for this term to the overall list
        x_indices_list.append(x_indices)

    # Print the final list of x indices
    return x_indices_list


def create_edge_list(angle_study: AngleStudy, cost_function: str, binary_expression: str):
    if angle_study == AngleStudy.automorphism_global:
        return create_edge_list_from_z(cost_function)
    elif angle_study == AngleStudy.automorphism_local:
        return create_edge_list_from_x(binary_expression)
    else:
        return ''


'''
#edge_list: list = [[0, 1, 2], [2, 3, 4], [0, 4]]
#7 binary cost function
#edge_list: list = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
#6 binary cost function
#edge_list: list = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
#6 full cost func
#edge_list: list = [[0, 1, 2], [0, 1], [0, 2], [0], [1, 2, 3], [1, 3], [2, 3, 4], [2, 4], [2], [3, 4, 5], [3, 5], [3], [4, 5], [5]]
#6 less single-expressions
#edge_list: list = [[0, 1, 2], [0, 1], [0, 2], [1, 2, 3], [1, 3], [2, 3, 4], [2, 4], [3, 4, 5], [3, 5], [4, 5]]
#q12 full cf
#edge_list =[[0,1,2], [0,1], [0,2],  [0],  [1,2,3],  [1,3],  [10,11,9],  [10,11],  [10,8,9],  [10,8],  [11,9],  [11],  [2,3,4], [2,4],  [2],  [3,4,5],  [3,5],  [3],  [4,5,6],  [4,6],  [4],  [5,6,7],  [5,7],  [5], [6,7,8],  [6,8],  [6],  [7,8,9],  [7,9],  [7],  [8],  [9]]
#12 binary cost function
#edge_list: list = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11]]
#e9 evan binary cost function
#edge_list: list = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [2, 5, 8]]
#e9 spin
#edge_list = [[0, 1, 2], [0, 1], [0, 2], [0, 3, 6], [0, 3], [0, 6], [1, 2], [1], [2, 5, 8], [2, 5], [2, 8], [2], [3, 4, 5], [3, 4], [3, 5], [3, 6], [3], [4, 5], [4], [5, 8], [6, 7, 8], [6, 7], [6, 8], [7, 8], [7], [8]]


#H = hnx.Hypergraph(edge_list)

cost_function = "0.125*Z[0]*Z[1]*Z[2] - 0.125*Z[0]*Z[1] - 0.125*Z[0]*Z[2] + 0.125*Z[0] - 0.125*Z[1]*Z[2]*Z[3] + 0.125*Z[1]*Z[3] + 0.125*Z[2]*Z[3]*Z[4] - 0.125*Z[2]*Z[4] + 0.125*Z[2] - 0.125*Z[3]*Z[4]*Z[5] + 0.125*Z[3]*Z[5] - 0.125*Z[3] + 0.125*Z[4]*Z[5] - 0.125*Z[5]"
#x = find_vertex_orbits(H, cost_function, 'bob')
#print(x)

y = create_edge_list_from_z(cost_function)
print(y)

binary_expression = "-x(x0) x(x1) x(x2) + x(x1) x(x2) x(x3) - x(x2) x(x3) x(x4) + x(x3) x(x4) x(x5)"
y = create_edge_list_from_x(binary_expression)
print(y)

y = create_edge_list(AngleStudy.automorphism_local, cost_function, binary_expression)
print(y)

#find_edge_orbits(H, None, True)
'''

