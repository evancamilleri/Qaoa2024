import sys
import yaml
from datetime import datetime
from time import monotonic, sleep
from humanfriendly import format_timespan
from enum import Enum
from collections import OrderedDict
from itertools import zip_longest
import networkx as nx
import hypernetx as hnx
from localdb import LocalDB
import qubovert
from qubovert import boolean_var
# internal
from observable_builder import ObservableBuilder
from qaoa_circuit import QaoaCircuit, InitialAngles, CircuitType, OrbitLibrary
from problem_data import Problem, Pubo, Qubo, Action
from evan_library import info, AngleStudy, ClassicalOptimiser, GraphType
from scipy.linalg import eigh


class OptimizationType(Enum):
    PUBO = 0
    QUBO = 1

# ************************************************************************************

# region * Load the settings from the YAML file
settings_filename = sys.argv[1] if len(sys.argv) > 1 else 'settings'

with open(f'{settings_filename}.yaml', 'r') as file:
    settings = yaml.safe_load(file)

# Accessing and using the settings
execution_ref = settings['execution_ref']
load_graphs = settings['load_graphs']
internal_graph_indexes = settings['internal_graph_indexes']
save_to_db = settings['save_to_db']
console_to_file = settings['console_to_file']
action = getattr(Action, settings['action'])  # Action is an enum
layer_range = settings['layer_range']

# Mapping string values to corresponding class attributes or enums
classical_optimizer_loop = tuple(settings['classical_optimizer_loop'])
angle_study_loop = tuple(getattr(AngleStudy, angle) for angle in settings['angle_study_loop'])
initial_angles_loop = tuple(getattr(InitialAngles, angle) for angle in settings['initial_angles_loop'])
optimization_loop = tuple(getattr(OptimizationType, opt_type) for opt_type in settings['optimization_loop'])
orbit_library = getattr(OrbitLibrary, settings['orbit_library'])

try_all_initial_angles = settings['try_all_initial_angles']
build_circuit = settings['build_circuit']
show_circuit = settings['show_circuit']
build_in_qiskit = settings['build_in_qiskit']
run_qaoa = settings['run_qaoa']

# Handling the dynamic folder name with datetime
folder_template = settings['folder']
folder = folder_template.replace("%Y%m%d_%H%M%S", datetime.now().strftime("%Y%m%d_%H%M%S")).replace("%REF%", execution_ref)

# ************************************************************************************
if console_to_file:
    # Open the file for writing
    output_file = open(f'{folder}.txt', 'w')
    sys.stdout = output_file  # Redirect stdout to the file

# PRINT PARAMETERS OF THIS RUN
print('*' * 50)
print(f'SETTINGS FILE: {settings_filename}')
print('-' * 25)
info('execution_ref', execution_ref)
info('load_graphs', load_graphs)
info('internal_graph_indexes', internal_graph_indexes)
info('action', action)
info('optimization_loop', optimization_loop)
info('orbit_library', orbit_library)
info('classical_optimizer_loop', classical_optimizer_loop)
info('angle_study_loop', angle_study_loop)
info('layer_range', layer_range)
info('initial_angles_loop', initial_angles_loop)
info('show_circuit', show_circuit)
info('try_all_initial_angles', try_all_initial_angles)
info('build_circuit', build_circuit)
info('build_in_qiskit', build_in_qiskit)
info('run_qaoa', run_qaoa)
info('save_to_db', save_to_db)
info('console_to_file', console_to_file)
info('folder', folder)
#print(f'n_loop = {n_loop}')
#print(f'shuffle_loop = {shuffle_loop}')
#print(f'coefficients_loop = {coefficients_loop}')
#print(f'generate_initial_angles = {generate_initial_angles}')
print('*' * 50)
print('')

# ************************************************************************************
#coefficients_loop = Coefficient_Generation.RANDOM, #Coefficient_Generation.ONE, Coefficient_Generation.RANDOM, #Coefficient_Generation.ALTERNATING
#class OptimizationType(Enum):    PUBO = 0    QUBO = 1
#class Coefficient_Generation(Enum):    ALTERNATING = 0    ONE = 1    RANDOM = 2

# endregion

# region > Define Global Variables
table_fields = ('execution_ref', 'n', 'maximize', 'classical_optimiser', 'poly_problem', 'angle_study'
                , 'angle_count', 'layers', 'initial_angles_type', 'coefficients_type', 'pubo_variables'
                , 'qubo_variables', 'qubits', 'ancillary_count', 'cost_function', 'execution_time', 'shuffle'
                , 'circuit_depth', 'circuit_depth_cx', 'circuit_depth_cx_parallel'
                , 'transpiled_circuit_depth', 'transpiled_circuit_depth_cx', 'transpiled_circuit_depth_cx_parallel'
                , 'result_bitstring', 'nfev', 'result', 'result_optimal', 'probability', 'expectation'
                , 'approximation_ratio', 'graph_fk', 'final_mixer_angles', 'final_cost_angles', 'classical_call_count')

record: tuple = tuple()
problems = []
db = None
shuffle = 0
# endregion

# region > Some Functions
def format_edge_string(G, my_alternate: bool):
    formatted_string = ""
    sign = "+"

    for i, edge in enumerate(G.edges()):
        if my_alternate and i != 0:
            sign = "-" if i % 2 != 0 else "+"
        formatted_string += f" {sign} x{edge[0]} x{edge[1]}"

    # Remove the leading sign
    if formatted_string.startswith(('+', '-')):
        formatted_string = formatted_string[2:].strip()

    return formatted_string


def format_hyper_string(hyper: set, my_alternate: bool, is_weighted: bool):
    formatted_string = ""
    sign = "+"

    for i, tup in enumerate(hyper):
        if is_weighted:
            if tup[0] < 0:
                coefficient = f'- {abs(tup[0])}'
            elif tup[0] > 0:
                coefficient = f'+ {tup[0]}'
            else:
                coefficient = '0'

            if coefficient != '0':
                formatted_string += f" {coefficient} x{tup[1]} x{tup[2]} x{tup[3]}"
        else:
            if my_alternate and i != 0:
                sign = "-" if i % 2 != 0 else "+"
            formatted_string += f" {sign} x{tup[0]} x{tup[1]} x{tup[2]}"

    # Remove the leading sign
    if formatted_string.startswith(('+', '-')):
        formatted_string = formatted_string[2:].strip()

    return formatted_string


def calculate_pubo_answer(C: qubovert.PUBO, qaoa_bitstring: str) -> (dict, int):
    variable_assignments: dict = {}

    # Iterate through the last n bits of the bitstring
    for i in range(C.num_binary_variables):
        variable_name = f'x{i}'
        variable_assignments[variable_name] = int(qaoa_bitstring[-(i + 1)])

    result = 0
    for term, coefficient in C.items():
        term_value = 1
        for variable in term:
            term_value *= variable_assignments.get(variable, 0)
        result += coefficient * term_value

    return variable_assignments, result


def get_max_answer(C_problem: qubovert.PUBO, answers, max_answer):
    _myResult = -sys.maxsize - 1 if action == Action.MAXIMIZE else sys.maxsize
    _myProbability = _myProbCount = 0
    _myBitStr = ''

    # loop all answers to make sure to find the top value
    for answer in answers:
        _v, _r = calculate_pubo_answer(C_problem, answer[0])
        if (action == Action.MAXIMIZE and _r > _myResult) or (action == Action.MINIMIZE and _r < _myResult):
            _myResult = _r

    # loop all answers to extract the top value only
    for answer in answers:
        _v, _r = calculate_pubo_answer(C_problem, answer[0])
        if (_r == _myResult):
            _myBitStr += answer[0] if _myBitStr == '' else ', ' + answer[0]
            _myProbability += answer[1]
            _myProbCount += 1

    if action == Action.MAXIMIZE:
        if _myResult >= max_answer:
            max_answer = _myResult
        else:
            _myBitStr = ''
            _myProbability = 0
            _myProbCount = 0

    # variable_assignments, result = calculate_pubo_answer(C_problem, answer[0][0])
    # print(f'Answer Prob.   : {answer[0][1]}')
    # print(f'Variable Values: {variable_assignments}')
    # print(f'Result ({action.name[:3]})   : {result}')
    info('Answer Prob.', f'{_myProbability} ({_myProbCount})')
    info('Correct Values', _myBitStr)
    info(f'Result ({action.name[:3]})', _myResult)
    AR = 0 if max_answer == 0  else (-expectation if action == Action.MAXIMIZE else expectation) / max_answer
    info('AR', AR)

    update_or_insert_field('probability', _myProbability)
    update_or_insert_field('result', _myResult)
    update_or_insert_field('result_bitstring', _myBitStr)
    update_or_insert_field('result_optimal', max_answer)
    update_or_insert_field('approximation_ratio', AR)

    return max_answer, _myProbability, _myProbCount, _myBitStr


def load_observable(problem_instance):
    variable_limits = OrderedDict()
    variable_limits['S'] = problem_instance.pubo.vertices if problem_instance.num_variables > 2 else problem_instance.qubo.vertices
    _constant = 1
    _obs_build = ObservableBuilder(variable_limits,
                                   problem_instance.pubo.Z if problem_instance.num_variables > 2 else problem_instance.qubo.Z,
                                   False, 1)
    variables = {}
    _obs_build.add_to_observable(variables)
    o_o = _obs_build.get_final_observable()
    obs, circuit_list = _obs_build.get_observable2(o_o)
    qubits = _obs_build.get_qubits()

    #MIRCO
    u, v = eigh(obs.get_matrix().todense())
    print(u)
    print(v)
    #quit()

    return obs,  circuit_list, qubits


def update_or_insert_field(field_name, new_result):
    global table_fields, record

    try:
        # Find the index of 'field_name' in table_fields
        field_index = table_fields.index(field_name)

        # Ensure record has enough empty values
        while len(record) <= field_index:
            record += (None,)

        # Update the corresponding index in record
        record = record[:field_index] + (new_result,) + record[field_index + 1:]
        x = 1
    except ValueError:
        print(f"'{field_name}' not found in 'table_fields'")
        # If 'field_name' is not found, insert it into table_fields and record
        table_fields = table_fields + (field_name,)
        record = record + (new_result,)


def divide_array(arr, x):
    part_length = len(arr) // x
    remainder = len(arr) % x
    parts = []

    for i in range(x):
        start_index = i * part_length + min(i, remainder)
        end_index = start_index + part_length + (1 if i < remainder else 0)
        parts.append(arr[start_index:end_index])

    return parts


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


def do_nothing(p1, p2):
    pass

#endregion

# region > Initialise Database
if save_to_db:
    db = LocalDB(folder)
    db.create_db()
# endregion

graph_type: GraphType = GraphType.na

# region > Load my Boolean Problems
if load_graphs == '':
    x_problem: str = ''
    for graph_index in internal_graph_indexes:
        match graph_index:
            case 'q4':
                x_problem = "+ x0 x1 x2 - x1 x2 x3"
                graph_type = GraphType.path
            case 'q4c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x0 - x3 x0 x1"
                graph_type = GraphType.cyclic
            case 'q5':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 "
                graph_type = GraphType.path
            case 'q5c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x0 + x4 x0 x1"
                graph_type = GraphType.cyclic
            case 'q6':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5"
                #x_problem = "+ x0 x1 x2 + x1 x2 x3 - x2 x3 x4 - x3 x4 x5"
                graph_type = GraphType.path
            case 'q6c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x0 - x5 x0 x1"
                graph_type = GraphType.cyclic
            case 'q6w':
                x_problem = "+ 29 x0 x1 x2 - 42 x1 x2 x3 + 12 x2 x3 x4 - 71 x3 x4 x5"
            case 'q7':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6"
                graph_type = GraphType.path
            case 'q7c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x0 + x6 x0 x1"
                graph_type = GraphType.cyclic
            case 'q8':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7"
                graph_type = GraphType.path
            case 'q8c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x0 - x7 x0 x1"
                graph_type = GraphType.cyclic
            case 'e9':
                x_problem = "+ x0 x1 x2 - x3 x4 x5 + x6 x7 x8 - x0 x3 x6 + x2 x5 x8"
            case 'q9':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8"
                graph_type = GraphType.path
            case 'q9c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x0 + x8 x0 x1"
                graph_type = GraphType.cyclic
            case 'q10':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9"
                graph_type = GraphType.path
            case 'q10c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x0 - x9 x0 x1"
                graph_type = GraphType.cyclic
            case 'q11':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10"
                graph_type = GraphType.path
            case 'q11c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x0 + x10 x0 x1"
                graph_type = GraphType.cyclic
            case 'q12':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11"
                graph_type = GraphType.path
            case 'q12c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x0 - x11 x0 x1"
                graph_type = GraphType.cyclic
            case 'g12':
                x_problem = "+ x2 x6 x7 - x1 x3 x6 + x3 x7 x10 - x0 x1 x10 + x0 x7 x10 - x0 x3 x7 + x4 x5 x10 - x7 x8 x9 + x0 x7 x11"
            case 'q13':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12"
                graph_type = GraphType.path
            case 'q13c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x0 + x12 x0 x1"
                graph_type = GraphType.cyclic
            case 'q14':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13"
                graph_type = GraphType.path
            case 'q14c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x0 - x13 x0 x1"
                graph_type = GraphType.cyclic
            case 'q15':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14"
                graph_type = GraphType.path
            case 'q15c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x0 + x14 x0 x1"
                graph_type = GraphType.cyclic
            case 'q16':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15"
                graph_type = GraphType.path
            case 'q16c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x0 - x15 x0 x1"
                graph_type = GraphType.cyclic
            case 'q17':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16"
            case 'q17c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16 - x15 x16 x0 + x16 x0 x1"
            case 'q18':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16 - x15 x16 x17"
            case 'q18c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16 - x15 x16 x17 + x16 x17 x0 - x17 x0 x1"
            case 'q19':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16 - x15 x16 x17 + x16 x17 x18"
            case 'q19c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16 - x15 x16 x17 + x16 x17 x18 - x17 x18 x0 + x18 x0 x1"
            case 'q20':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16 - x15 x16 x17 + x16 x17 x18 - x17 x18 x19"
            case 'q20c':
                x_problem = "+ x0 x1 x2 - x1 x2 x3 + x2 x3 x4 - x3 x4 x5 + x4 x5 x6 - x5 x6 x7 + x6 x7 x8 - x7 x8 x9 + x8 x9 x10 - x9 x10 x11 + x10 x11 x12 - x11 x12 x13 + x12 x13 x14 - x13 x14 x15 + x14 x15 x16 - x15 x16 x17 + x16 x17 x18 - x17 x18 x19 + x18 x19 x0 - x19 x0 x1"

            #1
            #x_problem = "-12.55 x0 x1 x2 + 45.07 x4 x1 x3 + 23.2 x0 x5 x3 - x6 - 42 x4 + 99"
            #x_problem = "1 x0 x1 x2"
            #x_problem = "1 x0 x1 - x1 x2 + x1 x3 - x2 x3 + x3 x4"
            #x_problem = "1 x0 x1 - x1 x2 + x1 x4 - x2 x3 + x2 x4 - x2 x6 + x4 x5 - x4 x6 + x6 x7"
            #x_problem = "1 x0 x1 - x0 x4 + x0 x5 - x1 x6 + x1 x2 - x2 x7 + x2 x3 - x3 x8 + x3 x4 - x4 x9 + x5 x7 - x5 x8 + x6 x8 - x6 x9 + x7 x9"
            #x_problem = "1 x0 x1 - x2 x7 + x2 x3 - x3 x8 + x3 x4 - x5 x8 + x6 x8 - x6 x9 + x7 x9"
            #x_problem = "+ x0 x1 - x0 x4 + x1 x3 - 7 x1 x2 + x2 x3 - x2 x4 + x3 x4"
            #x_problem = "+ x0 x1 x2 - x2 x3 + x2 x4 x5"
            #x_problem = "+ x0 x1 x2 - x2 x3 + x2 x4 x5 - x3 x 0"

        if x_problem != '':
            problem_instance = Problem(x_problem, None, None, None, Action.MAXIMIZE, do_nothing)
            edge_list = create_edge_list_from_z(problem_instance.z_problem)
            H = hnx.Hypergraph(edge_list)
            problems.append({"stX": x_problem, "stZ": problem_instance.z_problem, "nx": None, "hnx": H})
            #print(x_problem)
            #print(problem_instance.z_problem)

    print(problems)
else:
    if load_graphs.endswith(".g6"):
        # load problems from path to the graph6 file
        file_path = f'./graphs/{load_graphs}'

        # Read the graph(s) from the file
        graphs = nx.read_graph6(file_path)

        # Process each graph
        for G in graphs:
            print("Nodes:", G.number_of_nodes(), G.nodes())
            print("Edges:", G.edges())
            x_problem = format_edge_string(G, True)
            problem_instance = Problem(x_problem, None, None, None, Action.MAXIMIZE, do_nothing)
            problems.append({"stX": x_problem, "stZ": problem_instance.z_problem, "nx": G, "hnx": None})

    elif load_graphs.endswith(".hyper") or load_graphs.endswith(".hyperw"):
        hypergraph_sets = []
        is_weighted: bool = load_graphs.endswith(".hyperw")

        with open(f'./graphs_hyper/{load_graphs}', 'r') as f:
            for line_number, line in enumerate(f, start=1):
                print(f'### File Line Number: {line_number} ### ({line})')
                # Convert string back to a set of tuples using eval
                set_of_tuples = eval(line.strip())
                x_problem = format_hyper_string(set_of_tuples, True, is_weighted)
                problem_instance = Problem(x_problem, None, None, None, Action.MAXIMIZE, do_nothing)
                edge_list = create_edge_list_from_z(problem_instance.z_problem)
                H = hnx.Hypergraph(edge_list)
                problems.append({"stX": x_problem, "stZ": problem_instance.z_problem, "nx": None, "hnx": H})


# endregion

start_program_time = monotonic()

#load entry from db or generate-list
for problem_index, problem in enumerate(problems, start=0):

    problem_x = problem["stX"]
    problem_z = problem["stZ"]
    problem_nx = problem["nx"]
    problem_hnx = problem["hnx"]

    graphpk = 0
    record = tuple()
    problem_instance = Problem(problem_x, None, None, None, Action.MAXIMIZE, update_or_insert_field)
    record_backup = tuple(record)
    max_answer = -sys.maxsize - 1 if action == Action.MAXIMIZE else sys.maxsize
    # do we need to try all initial_angles (or is this more internal...?)

    for angle_study in angle_study_loop:

        for initial_angles in initial_angles_loop:
            my_initial_angles = None

            #my_initial_angles = [1.37412] * 6 + [3.19718] * 10
            #my_initial_angles = [7.85400547e-01, 3.92700590e+00, 7.85398605e-01,-2.74233551e-05, 4.03686142e+00, 3.08284530e-01, -3.01648381e-05, -6.65243337e-05, 3.05372498e-05, 7.91923845e+00, 1.25669170e+01, -9.82735483e-05, 1.25660837e+01, 1.25663232e+01, 1.84595323e+00, 7.21183884e+00]

            for optimization in optimization_loop:
                obs, circuit_list, qubits = load_observable(problem_instance)

                record = tuple(record_backup)
                update_or_insert_field('execution_ref', execution_ref)
                update_or_insert_field('n', problem_instance.num_variables)
                update_or_insert_field('maximize', action.value)
                update_or_insert_field('angle_study', angle_study.value)
                #update_or_insert_field('coefficients_type', coefficients_gen.value)
                update_or_insert_field('initial_angles_type', initial_angles.value)
                update_or_insert_field('poly_problem', 3 if optimization == OptimizationType.PUBO else 2)
                print('')
                print('-' * 100)
                print('')
                #print(f"boolean variables -> {n}")
                info('angle study', angle_study.name)
                info('initial angles', initial_angles.name)
                info('optimization', optimization.name)
                #print(f"coefficients gen. -> {coefficients_gen.name}")
                print('')

                for layers in layer_range:

                    info('Qubits', qubits)
                    info('Layers', layers)
                    update_or_insert_field('layers', layers)
                    update_or_insert_field('qubits', qubits)

                    for classical_optimzer in classical_optimizer_loop:
                        if build_circuit:
                            #get the internal graph name
                            internal_graph_name = internal_graph_indexes[problem_index] if load_graphs == '' else ''
                            info('Graph Name', internal_graph_name) if load_graphs == '' else ''
                            q = QaoaCircuit(layers, qubits, obs, circuit_list, initial_angles, angle_study, show_circuit
                                            , problem_instance.z_problem, problem_instance.cost_function.pretty_str(), graph_type
                                            , save_to_db, db if save_to_db else None, 0, graphpk, problem
                                            , CircuitType.Qaoa_Problem_Specific, my_initial_angles
                                            , update_or_insert_field, orbit_library, internal_graph_name)
                            if q.no_automorphism():
                                print('Graph has no automorphism')
                            else:
                                graphpk = q.graph_pk()
                                update_or_insert_field('graph_fk', graphpk)
                                # print(q.get_initial_angles())


                                # build_circuit
                                q.build_circuit(q.get_initial_angles(), build_in_qiskit, update_or_insert_field, shuffle, show_circuit)

                                # visualize2
                                #if show_circuit:
                                #    q.visualize2()

                                # run_qaoa
                                if run_qaoa:
                                    start_time = datetime.now()

                                    answer, expectation = q.minimize(classical_optimzer)

                                    update_or_insert_field('classical_optimiser', classical_optimzer)
                                    update_or_insert_field('expectation', -expectation if action == Action.MAXIMIZE else expectation)
                                    update_or_insert_field('execution_time', (datetime.now() - start_time).total_seconds())

                                    # loop all answers to make sure to find the top value
                                    max_answer, prob, nfound, bitstrings_found = get_max_answer(problem_instance.cost_problem, answer, max_answer)

                                    print('')
                                    print('### THIS IS WHAT WE WILL SAVE TO DB ###')
                                    print(record)
                                    print('')
                                    mixer_gates = q.get_mixer_gates()
                                    cost_gates = q.get_cost_gates()
                                    max_len = max(max(len(gate) for gate in mixer_gates),
                                                  max(len(gate) for gate in cost_gates))

                                    print('FINAL MIXER ANGLES')
                                    final_mixer_angles = ''
                                    parts = divide_array(q.get_final_angles()[:len(mixer_gates) * layers], layers)
                                    for layer, part in enumerate(parts, start=1):
                                        final_mixer_angles += f'---> Layer {layer}\n'
                                        for item1, item2 in zip_longest(mixer_gates, part, fillvalue='?'):
                                            final_mixer_angles += f"{item1.ljust(max_len)} ... {item2:.5f} ... {item2}\n"
                                    print(final_mixer_angles)

                                    print('FINAL COST ANGLES')
                                    final_cost_angles = ''
                                    parts = divide_array(q.get_final_angles()[len(mixer_gates) * layers:], layers)
                                    for layer, part in enumerate(parts, start=1):
                                        final_cost_angles += f'---> Layer {layer}\n'
                                        for item1, item2 in zip_longest(cost_gates, part, fillvalue='?'):
                                            final_cost_angles += f"{item1.ljust(max_len)} ... {item2:.5f} ... {item2}\n"
                                    print(final_cost_angles)
                                    print('')
                                    print(f"P                   : {layers}")
                                    print(f"GRAPH NAME          : {internal_graph_name if load_graphs == '' else ''}")
                                    print(f'ANGLE STUDY         : {angle_study.name}')
                                    print(f'ANGLES              : {len(q.get_initial_angles())}')
                                    print(f'ITERATIONS          : {q.get_iteration_count()}')
                                    print(f'FUNCTION EVALUATIONS: {q.get_function_evaluation_call_count()}')
                                    print(f'PROBABILITY (%)     : {prob} ({bitstrings_found})')
                                    print(f'RESULTS FOUND       : {nfound}')
                                    print(f'AR                  : {(-expectation if action == Action.MAXIMIZE else expectation) / max_answer}')

                                    update_or_insert_field('angle_count', len(q.get_initial_angles()))
                                    update_or_insert_field('final_mixer_angles', final_mixer_angles)
                                    update_or_insert_field('final_cost_angles', final_cost_angles)
                                    update_or_insert_field('classical_call_count', q.get_iteration_count())
                                    update_or_insert_field('nfev', q.get_function_evaluation_call_count())

                                    if save_to_db:
                                        pk = db.insert_db('tb_Test', 'test_pk', table_fields, record)
                                        # update tb_Test_Angle where test_fk = 0 to pk
                                        db.update_db('tb_Test_Angle', 'test_fk', 0, ('test_fk',), (pk,))

                            print('-----------------')
                            quit()
                            
elapsed_time = monotonic() - start_program_time
print(f"time taken (all tests) Run time {format_timespan(elapsed_time)}")
