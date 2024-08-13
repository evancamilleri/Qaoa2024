import itertools
from random import sample, seed
from math import comb
from problem_data import Problem, Pubo, Qubo, Action
import graph_automorphism_generic as gen
from qaoa_circuit import AngleStudy

# Seed for reproducibility
seed(0)

# Generate all possible hyperedges for 12 nodes
nodes = list(range(12))
all_hyperedges = list(itertools.combinations(nodes, 3))


def format_hyper_string(hyper: set, my_alternate: bool):
    formatted_string = ""
    sign = "+"

    for i, tup in enumerate(hyper):
        if my_alternate and i != 0:
            sign = "-" if i % 2 != 0 else "+"
        formatted_string += f" {sign} x{tup[0]} x{tup[1]} x{tup[2]}"

    # Remove the leading sign
    if formatted_string.startswith(('+', '-')):
        formatted_string = formatted_string[2:].strip()

    return formatted_string


def generate_hypergraph(all_hyperedges, nodes):
    hypergraph = set()

    # Ensure each node is in at least one hyperedge
    for node in nodes:
        while not any(node in edge for edge in hypergraph):
            hypergraph.add(sample(all_hyperedges, 1)[0])

    # Optionally, add more hyperedges to introduce variability
    additional_edges = sample(all_hyperedges, k=sample(range(1, 5), 1)[0])  # Add 1-4 more hyperedges randomly
    for edge in additional_edges:
        hypergraph.add(edge)

    return hypergraph


def update_or_insert_field(field_name, new_result):
    pass


# Generate 10,000 hypergraphs
num_hypergraphs = 10000
hypergraphs = [generate_hypergraph(all_hyperedges, nodes) for _ in range(num_hypergraphs)]

# Example output
print(f"Generated {len(hypergraphs)} hypergraphs.")
print(type(hypergraphs))
print(type(hypergraphs[0]))
#print(f"Example hypergraph: {list(hypergraphs[0])[:5]}")  # Print first 5 hyperedges of the first hypergraph
for hypergraph in hypergraphs:
    print(hypergraph)

with open('../graphs_hyper/12a.hyper', 'w') as f:
    for n, s in enumerate(hypergraphs):
        # get binary expression
        # get cost function
        problem = format_hyper_string(s, True)
        problem_instance = Problem(problem, None, None, None, Action.MAXIMIZE, update_or_insert_field)

        print(problem_instance.cost_function.pretty_str())
        print(problem_instance.qubo.Z)

        eo, eoc = gen.find_edge_orbits(AngleStudy.automorphism_global, problem_instance.qubo.Z, problem_instance.cost_function.pretty_str(), False)

        #if it has an AM then save
        if eo is not None:
            print(f'{n} -> OK')
            f.write(str(s) + '\n')
        else:
            print(n)
