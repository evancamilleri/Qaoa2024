from enum import Enum
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import hypernetx as hnx
from queue import Queue
#
from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
from qulacs.gate import H, CNOT, RX, RZ, RY
from qulacsvis.qulacs.circuit import to_model
from qulacsvis.visualization import MPLCircuitlDrawer # Use instead of circuit_drawer
from scipy.optimize import minimize
from qulacs.circuit import QuantumCircuitOptimizer
#
import qiskit
from qiskit.providers.fake_provider import FakeJakarta
from qiskit.visualization import circuit_drawer
#
import orqviz
from orqviz.scans import plot_2D_scan_result_as_3D
from orqviz.loss_function import LossFunctionWrapper
from gradient_descent_optimizer import gradient_descent_optimizer
from orqviz.gradients import calculate_full_gradient
from orqviz.pca import (get_pca, perform_2D_pca_scan, plot_pca_landscape,
                        plot_optimization_trajectory_on_pca)
#
from graph_automorphism import find_vertex_orbits, find_edge_orbits, find_edge_cycle_index, find_vertex_index
import graph_automorphism as gax
import graph_automorphism_nauty as gan
import graph_automorphism_generic as gen
from evan_library import info, insert_with_padding, AngleStudy, GraphType
#

coef_hell:bool = False

graph_fields = ('graph_type', 'graph_nodes', 'graph_edges', 'graph_is_weighted', 'graph_weight'
              , 'graph_is_hyper', 'graph_edge_list', 'graph_x', 'graph_z'
              , 'graph_degree', 'graph_average_degree'
              , 'graph_average_degree_connectivity', 'graph_density'
              , 'graph_clustering', 'graph_average_clustering', 'graph_average_geodesic_distance'
              , 'graph_betweenness_centrality'
              , 'graph_average_betweenness_centrality')


class InitialAngles(Enum):
    one_p = 0
    tenth_p = 1
    one_p2 = 2
    random0_to_1_p = 3
    random0_to_2pi_p = 4
    constant = 5
    constant_tuple = 6
    tqa100 = 7
    tqa075 = 8


class CircuitIndex(Enum):
    problem = 1
    mixer = 2


class CircuitType(Enum):
    Qaoa_Problem_Specific = 0
    Hardware_Optimised_Ansatz = 1
    Pubo_Qubo_A = 2

class OrbitLibrary(Enum):
    igraph = 0
    nauty = 1
    generic = 2

edge_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'teal', 'brown', 'navy', 'gray', 'olive', 'maroon']


class QaoaCircuit:
    def __init__(self, layers: int, qubits: int, cost_observable: Observable, circuit_list: list
                 , initial_angles_type: InitialAngles, angle_study: AngleStudy, show_circuit: bool
                 , cost_function: str, binary_function: str, graph_type: GraphType
                 , save_to_db: bool, db,  testfk: int, graphpk: int, problem: dict
                 , ansatz: CircuitType = CircuitType.Qaoa_Problem_Specific
                 , actual_initial_angles = None, update_or_insert_field = None
                 , orbit_library: OrbitLibrary = OrbitLibrary.igraph, internal_graph_name: str = ''):

        self._layers = layers
        self._qubits = qubits
        self._circuit_list = circuit_list
        self._cost_observable = cost_observable
        self._initial_angles_type = initial_angles_type
        self._initial_angles = np.empty(0)
        self._angle_study = angle_study
        self._angle_iteration = 0
        self._show_circuit = show_circuit

        self._cost_function = cost_function
        self._binary_function = binary_function
        self._graph_type = graph_type

        self._orbit_library = orbit_library
        self._internal_graph_name = internal_graph_name

        self._update_or_insert_field = update_or_insert_field

        # create Graph for our problem: used for AngleStudy.automorphism_? and visual display
        G = self.qubit_graph()
        self._save_to_db = False
        if save_to_db:
            self._save_to_db = True
            self._db = db
            self._table_testfk = testfk
            self._table_fields = ('test_fk', 'minimize_iteration', 'expectation', 'angle_string')

            if graphpk == 0:
                self._graph_pk = self.save_graph_data(problem)
            else:
                self._graph_pk = graphpk

        self.automorphism_error = False
        match angle_study:
            case AngleStudy.default:
                self._angle_count_per_layer_mixer = 1
                self._angle_count_per_layer_problem = 1

            case AngleStudy.multi_angle:
                self._angle_count_per_layer_mixer = self._qubits
                self._angle_count_per_layer_problem = self.get_problem_rotation_gates_count()

            case AngleStudy.just_one:
                self._angle_count_per_layer_mixer = 1  # this is managed during processing
                self._angle_count_per_layer_problem = 1

            case AngleStudy.polynomial:
                self._angle_count_per_layer_mixer = 1
                self._angle_count_per_layer_problem = self.get_problem_max_poly_count()

            case AngleStudy.ka:
                self._angle_count_per_layer_mixer = self._qubits
                self._angle_count_per_layer_problem = self.get_problem_max_poly_count()

            case AngleStudy.automorphism_global | AngleStudy.automorphism_local:
                print(self._orbit_library)
                self._vertex_cycle = gen.find_vertex_orbits(self._cost_function) if self._orbit_library == OrbitLibrary.generic else  gan.find_vertex_orbits(G, self._internal_graph_name) if self._orbit_library == OrbitLibrary.nauty else gax.find_vertex_orbits(G)
                info(f'vertex cycle ({len(self._vertex_cycle)})', self._vertex_cycle)
                self._edge_cycle, self._coef_cycle = gen.find_edge_orbits(angle_study, self._cost_function, self._binary_function, self._show_circuit) if self._orbit_library == OrbitLibrary.generic else gan.find_edge_orbits(G, self._vertex_cycle, self._internal_graph_name) if self._orbit_library == OrbitLibrary.nauty else gax.find_edge_orbits(G, self._vertex_cycle)
                if self._edge_cycle is None:
                    self.automorphism_error = True
                    return
                info(f'edge cycle ({len(self._edge_cycle)})', self._edge_cycle)
                #self._angle_count_per_layer_mixer = len(self._vertex_cycle) ..... will not cycle the MIXER
                self._angle_count_per_layer_mixer = self._qubits
                self._angle_count_per_layer_problem = len(self._edge_cycle)
                if self._angle_study == AngleStudy.automorphism_local:
                    self._angle_count_per_layer_problem += len(self._vertex_cycle)  # 2-qubit gates + 1 qubit gates

            case AngleStudy.qubits:
                self._angle_count_per_layer_mixer = self._angle_count_per_layer_problem = self._qubits

        self._ansatz = ansatz
        #
        match ansatz:
            case CircuitType.Qaoa_Problem_Specific:
                if actual_initial_angles is None:
                    self.generate_initial_angles()
                else:
                    self._initial_angles = actual_initial_angles
            case CircuitType.Hardware_Optimised_Ansatz:
                self.generate_custom_ansatz_initial_angles()
        #
        self._trajectory = [np.copy(self._initial_angles)]

        self._state = QuantumState(qubits)

        # do we need to sort the circuit_list?
        #TODO: why did I color the graph? do I need to do it again and do i need to check if this is OK for hyper?
        #if self._show_circuit:
            #edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            #nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold', edge_color=edge_colors, node_size=700, font_size=14)
            #plt.show()
            # reduce the depth according to Vizing's Theorem (where possible)

        self._processX = False
        self._mixerGates = []
        self._processC = False
        self._costGates = []

        self._final_angles = []
        self._ncalls = 0
        self._nfev = 0

    def save_graph_data(self, problem: dict):
        print('#' * 50)
        print('#' * 50)
        print('#' * 50)

        pk = 0

        if self._update_or_insert_field is not None:
            if problem["nx"] is not None:
                G = problem["nx"]
                try:
                    graph_average_geodesic_distance = nx.average_shortest_path_length(G)
                except:
                    graph_average_geodesic_distance = 0

                pk = self._db.insert_db('tb_Graph', 'graph_pk', graph_fields,
                                   (self._graph_type.value
                                    , G.number_of_nodes()
                                    , G.number_of_edges()
                                    , nx.is_weighted(G)
                                    , G.size(weight="weight")
                                    , False  # not Hyper
                                    , f'{G.edges.data("weight", default=1) if nx.is_weighted(G) else G.edges}'
                                    , problem["stX"]
                                    , problem["stZ"]
                                    , f'{nx.degree(G)}'
                                    , np.sum(G.degree(), axis=0)[1] / G.number_of_nodes()
                                    , f'{nx.average_degree_connectivity(G, weight="weight")}'
                                    , nx.density(G)
                                    , f'{nx.clustering(G)}'
                                    , nx.average_clustering(G)
                                    , graph_average_geodesic_distance
                                    , f'{nx.betweenness_centrality(G)}'
                                    , sum(nx.betweenness_centrality(G).values()) / G.number_of_nodes()
                                    ))

            elif problem["hnx"] is not None:
                H: hnx.Hypergraph = problem["hnx"]
                try:
                    # HyperNetX does not directly support average shortest path length calculation as in NetworkX,
                    # TODO: implement a custom function or approximate based on your needs.
                    hypergraph_average_geodesic_distance = 0
                except:
                    hypergraph_average_geodesic_distance = 0

                # Insert hypergraph properties into the database. Note that HyperNetX has different methods and might lack direct equivalents for some metrics.
                pk = self._db.insert_db('tb_Graph', 'graph_pk', graph_fields,
                                        (self._graph_type.value
                                         , H.number_of_nodes()
                                         , H.number_of_edges()
                                         , False  # HyperNetX does not have an is_weighted equivalent
                                         , 0  # G.size(weight="weight")
                                         , True
                                         , 'N/A'  # HyperNetX edges
                                         , problem["stX"]
                                         , problem["stZ"]
                                         , 0  # Degree information in hypergraphs can be more complex
                                         , 0  # Average node degree (or similar metric)
                                         , 0  # Average degree connectivity might not have a direct equivalent in HyperNetX
                                         , 0  # Density calculation provided by HyperNetX
                                         , 0  # Clustering might not have a direct equivalent in HyperNetX
                                         , 0  # Average clustering coefficient might need custom calculation
                                         , 0  # hypergraph_average_geodesic_distance
                                         , 0  # Betweenness centrality calculation for hypergraphs might require custom implementation
                                         , 0  # Sum of betweenness centrality values / number of nodes might require custom calculation
                                         ))

        return pk


    def qubit_graph(self):
        # save in a Graph
        G = nx.Graph()

        # add vertices
        for _i in range(self._qubits):
            G.add_node(_i)

        #print(type(self._circuit_list))
        for _c in self._circuit_list:
            _part = _c.get_pauli_matrices()
            _arr = QaoaCircuit.decompose_circuit_observable_token(_part)
        #    print(_part, _arr)

        for _c in self._circuit_list:
            _part = _c.get_pauli_matrices()
            _arr = QaoaCircuit.decompose_circuit_observable_token(_part)
            if len(_arr) == 2:
                # Add an edge between the two nodes
                G.add_edge(_arr[0], _arr[1])
            elif len(_arr) == 3:
                # Add edges between the nodes as per the specified rule
                G.add_edge(_arr[0], _arr[1])
                G.add_edge(_arr[1], _arr[2])
                G.add_edge(_arr[2], _arr[0])
            elif len(_arr) > 3:
                raise ValueError("Array must have either 2 or 3 elements")

        # declaring vector of vector of pairs, to define Graph
        gra = []
        edgeColors = []
        isVisited = [False] * 100000

        # Enter the Number of Vertices and the number of edges
        ver = nx.number_of_nodes(G)
        edge = nx.number_of_edges(G)

        gra = [[] for _ in range(ver)]
        edgeColors = [-1] * edge

        for index, edge_object in enumerate(G.edges):
            # graph[x].append((y, i))
            # graph[y].append((x, i))
            u, v = edge_object
            gra[u].append((v, index))
            gra[v].append((u, index))

        QaoaCircuit.color_edges(0, gra, edgeColors, isVisited)

        #print(G.edges)
        #print(edgeColors)
        #print(edge_colors)
        for index, edge_object in enumerate(G.edges):
            u, v = edge_object
            #print(index)
            #print(u)
            #print(v)
            #print('.')
            G[u][v]['color'] = edge_colors[edgeColors[index]]

        # printing all the edge colors
        max_degree = 0
        for node in G.nodes:
            max_degree = G.degree(node) if G.degree(node) > max_degree else max_degree
        info('Graph Degree', G.degree)
        info('Max Degree', max_degree)
        print('--- printing all the edge colors -----------------')
        for i in range(edge):
            print(f"Edge {i} is of color ({edgeColors[i]}) {edge_colors[edgeColors[i]]}")
        print('-' * 50)

        return G

    # function to determine the edge colors
    @staticmethod
    def color_edges(ptr, gra, edgeColors, isVisited):
        # Initialize gra as a list of empty lists
        q = Queue()
        c = 0

        colored = set()

        # return if isVisited[ptr] is true
        if (isVisited[ptr]):
            return

        # Mark the current node visited
        isVisited[ptr] = True

        # Traverse all edges of current vertex
        for i in range(len(gra[ptr])):
            # if already colored, insert it into the set
            if (edgeColors[gra[ptr][i][1]] != -1):
                colored.add(edgeColors[gra[ptr][i][1]])

        for i in range(len(gra[ptr])):
            # if not visited, inset into the queue
            if not isVisited[gra[ptr][i][0]]:
                q.put(gra[ptr][i][0])

            if (edgeColors[gra[ptr][i][1]] == -1):
                # if col vector -> negative
                while c in colored:
                    # increment the color
                    c += 1

                # copy it in the vector
                edgeColors[gra[ptr][i][1]] = c

                # then add it to the set
                colored.add(c)
                c += 1

        # while queue's not empty
        while not q.empty():
            temp = q.get()
            QaoaCircuit.color_edges(temp, gra, edgeColors, isVisited)

        return

    def no_automorphism(self):
        return self.automorphism_error

    def graph_pk(self):
        return self._graph_pk

    def get_initial_angles(self):
        return self._initial_angles

    def get_final_angles(self):
        return self._final_angles

    def get_problem_rotation_gates_count(self):
        return sum(1 for c in self._circuit_list if c.get_pauli_matrices() != '')

    def get_mixer_gates(self):
        return self._mixerGates

    def get_cost_gates(self):
        return self._costGates

    def get_problem_max_poly_count(self):
        max_poly = 0
        for c in self._circuit_list:
            _part = c.get_pauli_matrices()
            _arr = QaoaCircuit.decompose_circuit_observable_token(_part)
            max_poly = len(_arr) if len(_arr) > max_poly else max_poly
        return max_poly

    def get_iteration_count(self):
            return self._ncalls

    def get_function_evaluation_call_count(self):
            return self._nfev

    def generate_initial_angles(self):
        angle = 1
        #
        match self._initial_angles_type:
            case InitialAngles.tqa100:  # https://arxiv.org/pdf/2101.05742.pdf
                tqa_delta_t = 1
            case InitialAngles.tqa075:
                tqa_delta_t = 0.75
            case _:
                tqa_delta_t = 0
        #
        match self._initial_angles_type:
            case InitialAngles.tqa100 | InitialAngles.tqa075:
                # mixer angles
                for _layerIndex in range(self._layers * self._angle_count_per_layer_mixer):
                    angle = (1 - ((_layerIndex + 1) / self._layers)) * tqa_delta_t
                    self._initial_angles = np.append(self._initial_angles, angle)

                # problem angles
                for _layerIndex in range(self._layers * self._angle_count_per_layer_problem):
                    angle = ((_layerIndex + 1) / self._layers) * tqa_delta_t
                    self._initial_angles = np.append(self._initial_angles, angle)
            #
            case InitialAngles.random0_to_1_p | InitialAngles.random0_to_2pi_p:
                # mixer angles
                for _ in range(self._layers * self._angle_count_per_layer_mixer):
                    match self._initial_angles_type:
                        case InitialAngles.random0_to_1_p:
                            angle = random.uniform(0, 1) / self._layers
                        case InitialAngles.random0_to_2pi_p:
                            angle = random.uniform(0, 2 * math.pi) / self._layers
                    self._initial_angles = np.append(self._initial_angles, angle)

                # problem angles
                for _ in range(self._layers * self._angle_count_per_layer_problem):
                    match self._initial_angles_type:
                        case InitialAngles.random0_to_1_p:
                            angle = random.uniform(0, 1) / self._layers
                        case InitialAngles.random0_to_2pi_p:
                            angle = random.uniform(0, 2 * math.pi) / self._layers
                    self._initial_angles = np.append(self._initial_angles, angle)
            #
            case _:
                match self._initial_angles_type:
                    case InitialAngles.one_p:
                        angle = 1 / self._layers
                    case InitialAngles.tenth_p:
                        angle = 0.1 / self._layers
                    case InitialAngles.one_p2:
                        angle = 1 / self._layers**2

                # mixer angles
                for _ in range(self._layers * self._angle_count_per_layer_mixer):
                    self._initial_angles = np.append(self._initial_angles, angle)

                # problem angles
                for _ in range(self._layers * self._angle_count_per_layer_problem):
                    self._initial_angles = np.append(self._initial_angles, angle)

    def generate_initial_angles_BK(self, ):
        circuit_loop = CircuitIndex.mixer, CircuitIndex.problem
        angle_count_per_layer = len(circuit_loop)
        angle = 1
        #
        match self._initial_angles_type:
            case InitialAngles.tqa100:
                tqa_delta_t = 1
            case InitialAngles.tqa075:
                tqa_delta_t = 0.75
            case _:
                tqa_delta_t = 0
        #
        match self._initial_angles_type:
            case InitialAngles.tqa100 | InitialAngles.tqa075:
                for _circuitIndex in circuit_loop:
                    for _layerIndex in range(self._layers):
                        match _circuitIndex:
                            case CircuitIndex.mixer:
                                angle = (1 - ((_layerIndex + 1) / self._layers)) * tqa_delta_t
                            case CircuitIndex.problem:
                                angle = ((_layerIndex + 1) / self._layers) * tqa_delta_t
                        #
                        self._initial_angles = np.append(self._initial_angles, angle)
            #
            case InitialAngles.random0_to_1_p | InitialAngles.random0_to_2pi_p:
                for _ in range(self._layers * angle_count_per_layer):
                    match self._initial_angles_type:
                        case InitialAngles.random0_to_1_p:
                            angle = random.uniform(0, 1) / self._layers
                        case InitialAngles.random0_to_2pi_p:
                            angle = random.uniform(0, 2 * math.pi) / self._layers
                    #
                    self._initial_angles = np.append(self._initial_angles, angle)
            #
            case _:
                match self._initial_angles_type:
                    case InitialAngles.one_p:
                        angle = 1 / self._layers
                    case InitialAngles.tenth_p:
                        angle = 0.1 / self._layers
                    case InitialAngles.one_p2:
                        angle = 1 / self._layers**2
                #
                for _ in range(self._layers * angle_count_per_layer):
                    self._initial_angles = np.append(self._initial_angles, angle)

    def generate_custom_ansatz_initial_angles(self):
        angle = random.uniform(0, 2 * math.pi) / self._layers
        match self._ansatz:
            case CircuitType.Hardware_Optimised_Ansatz:
                for i in range(6):
                    self._initial_angles = np.append(self._initial_angles, angle)

    def custom_ansatz(self, angles: np.ndarray):
        circuit = QuantumCircuit(self._qubits)

        for _ in range(self._layers):
            match self._ansatz:
                case CircuitType.Hardware_Optimised_Ansatz:
                    for i in range(self._qubits):
                        circuit.add_RY_gate(i, angles[i])

                    for i in range(self._qubits):
                        circuit.add_RZ_gate(i, angles[i+self._qubits])

                    circuit.add_CNOT_gate(0, 1)
                    circuit.add_CNOT_gate(1, 2)
                    circuit.add_CNOT_gate(2, 0)  # << optional? >>

                case CircuitType.Pubo_Qubo_A:
                    # make Pubo to Qubo x₀x₁x₂ = x₀x₃, where x₃ = x₁x₂
                    # [1] make Pubo circuit of x₀x₃
                    circuit.add_H_gate(0)
                    circuit.add_H_gate(1)  # ???
                    circuit.add_H_gate(2)  # ???
                    circuit.add_H_gate(3)
                    circuit.add_CNOT_gate(0, 3)
                    circuit.add_RZ_gate(3, 0)
                    circuit.add_CNOT_gate(0, 3)
                    # [2] then add the term = 3x₃ + x₁x₂ − 2x₁x₃ − 2x₂x₃
                    # ??????????????????????????????????????????????????
                    # ??????????????????????????????????????????????????

        print('********************************************')
        print(circuit.calculate_depth())
        print('********************************************')
        if self._show_circuit:
            QaoaCircuit.show_circuit(circuit)
            print(circuit)

        self._state.set_zero_state()
        circuit.update_quantum_state(self._state)

        # print(self._cost_observable.get_expectation_value(self._state))

        return self._cost_observable.get_expectation_value(self._state)

    def count_non_parallel_cx(self, qc: qiskit.QuantumCircuit, qbits):
        #todo: test well
        # Define the dimensions of the array
        rows = qbits
        columns = qc.depth()  # depth of circuit

        # Initialize the 2D boolean array with all values set to False : 1 = CX GATE, 2 = OTHER GATE
        gate_grid = [[0 for _ in range(columns)] for _ in range(rows)]
        # print(gate_grid); print(columns)

        for instruction, qargs, cargs in qc.data:
            bit0 = qc.qubits.index(qargs[0])
            bit1 = qc.qubits.index(qargs[1]) if instruction.num_qubits == 2 else -1
            # print(f'{instruction.name} {bit0} {bit1}')

            # Loop through columns and select where this gate will be placed
            my_idx = -1
            for col_idx in range(columns):
                column = [row[col_idx] for row in gate_grid]
                if instruction.num_qubits == 1 and column[bit0] != 0:
                    my_idx = col_idx
                else:
                    if instruction.num_qubits == 2 and not (column[bit0] == 0 and column[bit1] == 0):
                        my_idx = col_idx
            my_idx += 1

            if instruction.name == 'cx':
                gate_grid[bit0][my_idx] = 1
                gate_grid[bit1][my_idx] = 1
            else:
                gate_grid[bit0][my_idx] = 2
                if instruction.num_qubits == 2:
                    gate_grid[bit1][my_idx] = 2
        #[print(row) for row in gate_grid]

        # Check how many columns have at least one True value
        cx_count = 0
        for col_idx in range(columns):
            if any(row[col_idx] == 1 for row in gate_grid):
                cx_count += 1
        return cx_count

    def build_qaoa_circuit(self, angles: np.ndarray, build_in_qiskit: bool = False, update_or_insert_field = None, shuffle: int = 0, show_circuit: bool = False):
        np.set_printoptions(linewidth=2000)

        # angles = [beta/mixer | gamma/problem]
        match self._angle_study:
            case AngleStudy.just_one:
                beta = gamma = angles
            case _:
                beta = angles[:self._layers * self._angle_count_per_layer_mixer]    # X-Mixer (first angles)
                gamma = angles[self._layers * self._angle_count_per_layer_mixer:]   # Cost Function (last angles)

        #print(angles)  #EVAN
        #print('beta', beta)
        #print(self._layers, self._angle_count_per_layer_problem)
        #print('gamma', gamma)

        circuit = QuantumCircuit(self._qubits)
        qiskit_qc = None
        if build_in_qiskit:
            qiskit_qc = qiskit.QuantumCircuit(self._qubits)

        # initialize using Hadamard gates
        for qubit in range(self._qubits):
            circuit.add_H_gate(qubit)
            qiskit_qc.h(qubit) if build_in_qiskit else None

        for layer in range(self._layers):
            _from = layer * self._angle_count_per_layer_problem
            _to = _from + --self._angle_count_per_layer_problem
            #print('g#', _from, _to, gamma) #EVAN
            self.add_U_C(circuit, gamma[_from:_to], build_in_qiskit, qiskit_qc, shuffle)
            #
            _from = layer * self._angle_count_per_layer_mixer
            _to = _from + --self._angle_count_per_layer_mixer
            #print('b#', _from, _to, beta)
            self.add_U_X(circuit, -2, beta[_from:_to], build_in_qiskit, qiskit_qc)

        if show_circuit:
            #print('BEFORE OPT********************************************')
            print(circuit.calculate_depth())
            QaoaCircuit.show_circuit(circuit)
            print(circuit)

            #TODO:
            '''
            # Optimization
            opt = QuantumCircuitOptimizer()
            # The maximum quantum gate size allowed to be created
            max_block_size = 2
            opt.optimize(circuit, max_block_size)

            print('AFTER OPT********************************************')
            print(circuit.calculate_depth())
            QaoaCircuit.show_circuit(circuit)
            print(circuit)
            '''

        self._state.set_zero_state()
        circuit.update_quantum_state(self._state)

        if build_in_qiskit:
            transpiled_circuit = qiskit.transpile(qiskit_qc, basis_gates=['sx', 'rz', 'cx'], optimization_level=3)
            #transpiled_circuit = qiskit.transpile(qiskit_qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
            #transpiled_circuit = qiskit.transpile(qiskit_qc, FakeJakarta(), optimization_level=3)
            if self._show_circuit:
                qiskit_qc.draw(output='mpl')
                plt.show()
                circuit_drawer(transpiled_circuit, fold=30, output='mpl')
                plt.show()

            print(f'CIRCUIT depth    := {qiskit_qc.depth()}')
            print(f"CIRCUIT CX       := {qiskit_qc.count_ops()['cx']}")
            print(f"CIRCUIT CX //    := {self.count_non_parallel_cx(qiskit_qc, self._qubits)}")

            print(f'TRANSPILED depth := {transpiled_circuit.depth()}')
            cx_count = transpiled_circuit.count_ops().get('cx', 0)
            print(f"TRANSPILED CX    := {cx_count}")
            print(f"TRANSPILED CX // := {self.count_non_parallel_cx(transpiled_circuit, self._qubits)}")

            if update_or_insert_field is not None:
                update_or_insert_field('circuit_depth', qiskit_qc.depth())
                update_or_insert_field('circuit_depth_cx', qiskit_qc.count_ops()['cx'])
                update_or_insert_field('circuit_depth_cx_parallel', self.count_non_parallel_cx(qiskit_qc, self._qubits))

                update_or_insert_field('transpiled_circuit_depth', transpiled_circuit.depth())
                update_or_insert_field('transpiled_circuit_depth_cx', cx_count)
                update_or_insert_field('transpiled_circuit_depth_cx_parallel', self.count_non_parallel_cx(transpiled_circuit, self._qubits))

            if self._show_circuit:
                plt.show()

        expectation = self._cost_observable.get_expectation_value(self._state)
        #print(f'expectation: {expectation} | angles {angles}')
        if self._save_to_db:
            angle_string = '|'.join(f'{angle}' for angle in angles)
            self._angle_iteration += 1
            #EC 2024.10.04 do NOT save angles
            #self._db.insert_db('tb_Test_Angle', 'test_pk', self._table_fields, (self._table_testfk, self._angle_iteration, expectation, angle_string))

        return expectation

    def build_circuit(self, angles: np.ndarray, build_in_qiskit: bool = False, update_or_insert_field = None, shuffle: int = 0, show_circuit: bool = False):
        if self._ansatz == CircuitType.Qaoa_Problem_Specific:
            return self.build_qaoa_circuit(angles, build_in_qiskit, update_or_insert_field, shuffle, show_circuit)
        else:
            return self.custom_ansatz(angles)

    def visualize(self):
        # params = initial params
        dir1 = orqviz.geometric.get_random_normal_vector(len(self._initial_angles))
        dir2 = orqviz.geometric.get_random_orthonormal_vector(dir1)

        scan2D_result = orqviz.scans.perform_2D_scan(self._initial_angles, self.build_circuit,
                                                     direction_x=dir1, direction_y=dir2,
                                                     n_steps_x=250)
        orqviz.scans.plot_2D_scan_result(scan2D_result)
        plt.show()
        plot_2D_scan_result_as_3D(scan2D_result)
        plt.show()

    def visualize2(self):
        def gradient_function2(parameters):
            return calculate_full_gradient(parameters, self.build_circuit, stochastic=False, eps=1e-3)

        n_iters = 50
        parameter_trajectory2, losses2 = gradient_descent_optimizer(self._initial_angles,
                                                                    self.build_circuit,
                                                                    n_iters,
                                                                    learning_rate=0.2,
                                                                    full_gradient_function=gradient_function2)

        pca2 = get_pca(parameter_trajectory2)
        scan_pca_result3 = perform_2D_pca_scan(pca2, self.build_circuit, n_steps_x=100, offset=2)

        fig, ax = plt.subplots()
        plot_pca_landscape(scan_pca_result3, pca2, fig=fig, ax=ax)
        plot_optimization_trajectory_on_pca(parameter_trajectory2, pca2, ax=ax)
        plt.show()
        plot_2D_scan_result_as_3D(scan_pca_result3)
        plt.show()

    @staticmethod
    def show_circuit(circuit: QuantumCircuit):
        drawer = MPLCircuitlDrawer(to_model(circuit))
        fig = drawer.draw()
        plt.show()

    # circuit U_C(gamma)
    def add_U_C(self, circuit: QuantumCircuit, gamma, build_in_qiskit: bool = False, qiskit_qc: qiskit.QuantumCircuit = None, shuffle: int = 0):
        # shuffle == 0  # no shuffle
        # shuffle == -1 # try to squash
        # shuffle >= 0  # shuffle with given seed

        shuffled_list = self._circuit_list.copy()

        #result_string = ', '.join(shuffled_list)
        #print('*'*50)
        #print(result_string)
        #print('e*E*'*50)

        result_string = ""

        if shuffle > 0:
            random.seed(shuffle)
            random.shuffle(shuffled_list)

        index = -1
        #print('gamma', gamma) # EVAN
        for _c in shuffled_list:
            _part = _c.get_pauli_matrices()
            _coefficient = _c.get_coefficient()
            _count = QaoaCircuit.decompose_circuit_observable_token(_part)

            #########################
            #print(_part, _coefficient, _count)

            result_string += f'<<{_part}|{_coefficient}|{_count}>> '

            #print(_coefficient, '-> ', _part)
            #get index in gamma, only if there is a Pauli Matrix #todo: need to make sure Identity is ignored
            index += 1 if _part != '' else 0
            n = len(_count)
            if n > 0:  # build RZ or RZ...Z gate
                for i in range(n - 1):
                    circuit.add_CNOT_gate(_count[i], _count[i + 1])
                    qiskit_qc.cx(_count[i], _count[i + 1]) if build_in_qiskit else None
                    #print(f'    circuit.add_CNOT_gate({_count[i]}, {_count[i + 1]})') if self._show_circuit else None
                #print(index, _c, gamma)

                # _gamma = gamma[index] if self._multi_angle else gamma[0]
                _gamma = 0
                _index = 0
                match self._angle_study:
                    case AngleStudy.multi_angle:
                        _gamma = gamma[index]
                        _index = index
                    case AngleStudy.default | AngleStudy.just_one:
                        _gamma = gamma[0]
                        _index = 0
                    case AngleStudy.polynomial | AngleStudy.ka:
                        _gamma = gamma[n-1]  # Z (i.e. 0), ZZ (i.e. 1), ZZZ (i.e. 2)
                        _index = n-1
                    case AngleStudy.automorphism_global | AngleStudy.automorphism_local:
                        if n == 3:
                            cycle_index = find_edge_cycle_index(self._edge_cycle, (_count[0], _count[1], _count[2]))
                        elif n == 2:
                            cycle_index = find_edge_cycle_index(self._edge_cycle, (_count[0], _count[1]))
                        else:
                            if self._angle_study == AngleStudy.automorphism_global:
                                cycle_index = find_edge_cycle_index(self._edge_cycle, (_count[0], ))
                            else:  # AngleStudy.automorphism_local
                                cycle_index = len(self._edge_cycle) + find_vertex_index(self._vertex_cycle, _count[0])

                        _index = cycle_index
                        _gamma = gamma[cycle_index]

                if not self._processC:
                    if not (0 <= _index < len(self._costGates)):
                        insert_with_padding(self._costGates, _index, _part, '')
                        #self._costGates.insert(_index, _part)
                    else:
                        self._costGates[_index] += ', ' if self._costGates[_index] != '' else ''
                        self._costGates[_index] += _part

                if coef_hell:
                    circuit.add_gate(RZ(_count[n - 1], self._coef_cycle[_index] * _gamma))
                else:
                    circuit.add_gate(RZ(_count[n - 1], _coefficient * _gamma))
                qiskit_qc.rz(_coefficient * _gamma, _count[n - 1]) if build_in_qiskit else None
                #print(f'    circuit.add_gate(RZ({_count[n - 1]}, ç:{_coefficient} * γ:{_gamma}))') #if self._show_circuit else None

                for i in range(n - 2, -1, -1):
                    circuit.add_CNOT_gate(_count[i], _count[i + 1])
                    qiskit_qc.cx(_count[i], _count[i + 1]) if build_in_qiskit else None
                    #print(f'    circuit.add_CNOT_gate({_count[i]}, {_count[i + 1]})') if self._show_circuit else None

        self._processC = True

        #print('c*C*' * 50)
        #if self._show_circuit:
        #    print('*'*50)
        #    print(result_string)
        #    print('*'*50)


    # circuit U_X(beta)
    def add_U_X(self, circuit: QuantumCircuit, coefficient: float, beta, build_in_qiskit: bool = False, qiskit_qc: qiskit.QuantumCircuit = None):
        #print('beta',beta)
        for i in range(self._qubits):
            #_beta = beta[i] if self._multi_angle else beta[0]
            _beta = 0
            _index = 0
            match self._angle_study:
                case AngleStudy.multi_angle | AngleStudy.automorphism_global | AngleStudy.ka | AngleStudy.automorphism_local:
                    _beta = beta[i]
                    _index = i
                case AngleStudy.default | AngleStudy.just_one | AngleStudy.polynomial:
                    _beta = beta[0]

            if not self._processX:
                #print(_index)
                if not (0 <= _index < len(self._mixerGates)):
                    self._mixerGates.insert(_index, f'X_{i}')
                else:
                    self._mixerGates[_index] += f', X_{i}'

            circuit.add_gate(RX(i, coefficient * _beta))
            qiskit_qc.rx(coefficient * _beta, i) if build_in_qiskit else None

        self._processX = True

    @staticmethod
    def decompose_circuit_observable_token(pauli_matrices_string: str):
        numbers = [int(s) for s in pauli_matrices_string.split() if s.isdigit()]
        return numbers

    # To store the trajectory for visualization
    def callback_function(self, xk):
        self._trajectory.append(np.copy(xk))
        self._ncalls += 1


    def minimize(self, method: str = 'BFGS'):
        if method.lower() == 'none':
            expectation = self.build_circuit(self._initial_angles)
        else:
            self._ncalls = 0
            self._nfev = 0
            result = minimize(self.build_circuit, self._initial_angles, options={'maxiter': 500}, method=method, callback=self.callback_function)
            expectation = result.fun
            self._nfev = result.nfev
            info(f'Function Evaluation', result.nfev)
            info(f'final angles ({len(result.x)})', result.x)     # final angles
            self._final_angles = result.x

        info('expectation', expectation)  # expectation

        # The square of the absolute value of each component of the state vector = Observation probability
        probs = np.abs(self._state.get_vector()) ** 2

        # A bit string that can be obtained when projectively measured in the z direction
        z_basis = [format(i, "b").zfill(self._qubits) for i in range(probs.size)]

        # merge probs with z_basis and print the sorted results
        t = []
        for i in range(len(z_basis)):
            t.append((z_basis[i], round(probs[i] * 100, 2)))

        final_answer = sorted(t, key=lambda x: x[1], reverse=True)[:20]
        info('final answer', final_answer)

        #--------------
        if self._show_circuit:
            # Now, use ORQVIZ for visualization
            # Convert trajectory to the format expected by ORQVIZ
            parameter_trajectory_orqviz = np.array(self._trajectory)

            # Use ORQVIZ functions for PCA and visualization
            pca_orqviz = get_pca(parameter_trajectory_orqviz)
            scan_pca_result_orqviz = perform_2D_pca_scan(pca_orqviz, self.build_circuit, n_steps_x=100, offset=2)

            # Visualization - 2D PCA Landscape
            fig, ax = plt.subplots()
            plot_pca_landscape(scan_pca_result_orqviz, pca_orqviz, fig=fig, ax=ax)
            plot_optimization_trajectory_on_pca(parameter_trajectory_orqviz, pca_orqviz, ax=ax)
            plt.show()

            # Visualization - 3D Plot with Trajectory

            # Assuming pca_orqviz has attributes for principal components and the mean
            plot_2D_scan_result_as_3D(scan_pca_result_orqviz)
            #
            #
            plt.show()

        #--------------

        return final_answer, expectation

    @staticmethod
    def show_gantt(variable_limits, bitstring: str):
        def reverse_qubit_index(index: int):
            reverse_dict = {}
            multiplier = 1

            for variable, limit in reversed(variable_limits.items()):
                value = index // multiplier % limit
                reverse_dict[variable.lower()] = value
                multiplier *= limit

            return reverse_dict

        x_list = []

        for index, char in enumerate(reversed(bitstring)):
            if char == '1':
                x = reverse_qubit_index(index)
                if 't' not in x:
                    x['t'] = 0
                x_list.append(x)
                #print(index, char, x)

        if 'T' not in variable_limits:
            variable_limits['T'] = 1

        #print(variable_limits)
        print(x_list)

        # tabulate jssp data

        from tabulate import tabulate
        from itertools import product

        # Prepare the table data
        table_data = []

        # Get the unique 't' values from variable_limits
        t_values = list(range(variable_limits['T']))

        # Get the unique 'm' values from variable_limits
        m_values = list(range(variable_limits['M']))

        # Generate all combinations of 'm' and 't' values
        combinations = list(product(m_values, t_values))

        # Populate the table data with empty lists
        for m in m_values:
            row = [['m_{}'.format(m)]] + [[] for _ in range(len(t_values))]
            table_data.append(row)

        # Fill the table data with the 's' values from x_list
        for x in x_list:
            m_index = m_values.index(x['m'])
            t_index = t_values.index(x['t'])
            table_data[m_index][t_index + 1].append('s_{}'.format(x['s']))

        # Convert the lists in table_data to comma-separated strings
        table_data = [[', '.join(cell) for cell in row] for row in table_data]

        # Add the 't' headers
        t_headers = [''] + ['t_{}'.format(t) for t in t_values]

        # Print the table using tabulate
        table = tabulate(table_data, t_headers, tablefmt='pipe')

        print(table)
