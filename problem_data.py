from enum import Enum
import re
import itertools
import math
from sympy import symbols, expand, sympify
import qubovert
from evan_library import info

# Define the symbols
Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17, Z18, Z19, Z20, Z21, Z22, Z23, Z24 \
    = symbols('Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17, Z18, Z19, Z20, Z21, Z22, Z23, Z24')


class Action(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


class Qubo:
    def __init__(self, Z, vertices: int, ancillary: int):
        self._Z = Z
        self._vertices = vertices
        self._ancillary = ancillary

    @property
    def Z(self):
        return self._Z

    @property
    def vertices(self):
        return self._vertices

    @property
    def ancillary(self):
        return self._ancillary


class Pubo:
    def __init__(self, Z, vertices: int):
        self._Z = Z
        self._vertices = vertices

    @property
    def Z(self):
        return self._Z

    @property
    def vertices(self):
        return self._vertices


class Problem:
    def __init__(self, x_problem, z_problem, pubo, qubo, action: Action, update_or_insert_field):
        C: qubovert.PUBO = qubovert.PUBO()
        self._update_or_insert_field = update_or_insert_field

        if x_problem is not None:
            C = self.transform(x_problem)
            C_problem = C
            C = -C if action == Action.MAXIMIZE else C

            self._cost_function = C
            self._cost_problem = C_problem

            self._x_problem = C_problem.pretty_str()
            info("'x' Problem", C_problem.pretty_str())
            info("'x' Expression", C.pretty_str())

        num_variables = 0
        if z_problem is None and x_problem is not None:
            # create z_problem from x_problem
            num_variables, pubo_string = self.x_pubo(C)
            z_problem = self.replace_x(pubo_string)
            z_problem = z_problem.strip()
            z_problem = '+ 1 ' + z_problem if z_problem[0] == '*' else z_problem
            z_problem = str(expand(sympify(z_problem)))
            z_problem = self.replace_Zn_with_Zn_brackets(z_problem)
            info('Z Expression', z_problem)
            info('Variable Count', num_variables)
            info('Cost Function', z_problem)
            update_or_insert_field('cost_function', (z_problem[:497] + '...') if len(z_problem) > 500 else z_problem)
        else:
            z_variables = re.findall(r'Z\d+', str(z_problem))
            num_variables = len(set(z_variables))

        self._num_variables = num_variables
        self._z_problem = z_problem

        if pubo is None and z_problem is not None:
            if num_variables > 2:
                pubo = Pubo(z_problem, num_variables)
        self._pubo = pubo
        info('PUBO variables', pubo.vertices)
        update_or_insert_field('pubo_variables', pubo.vertices)
        #info("'ZZ'", self.count_occurrences(z_problem, 'Z', pubo.vertices, pairs_int))

        if qubo is None and z_problem is not None:
            # transform PUBO to QUBO
            pairs, pairs_int = self.make_pairs(self._num_variables)
            #print(pairs)
            qubo_num_variables, qubo_string = self.x_pubo_qubo(C, pairs)
            info('QUBO variables', qubo_num_variables)
            self._update_or_insert_field('qubo_variables', qubo_num_variables)
            qubo_final = self.replace_x(qubo_string)
            qubo_final = self.replace_Zn_with_Zn_brackets(str(expand(sympify(qubo_final))))
            info("'x' QUBO Expr.", qubo_string)
            info("'xx'", self.count_occurrences(qubo_string, 'x', self._num_variables, pairs_int))
            info('QUBO Expression', qubo_final)
            qubo = Qubo(qubo_final, qubo_num_variables, 0)
        else:
            qubo_num_variables = re.findall(r'Z\d+', str(pubo))
        self._qubo = qubo
        #self.qubo_num_variables = qubo_num_variables

    @property
    def x_problem(self):
        return self._x_problem

    @property
    def cost_function(self):
        return self._cost_function

    @property
    def cost_problem(self):
        return self._cost_problem

    @property
    def num_variables(self):
        return self._num_variables

    @property
    def z_problem(self):
        return self._z_problem

    @property
    def pubo(self):
        return self._pubo

    @property
    def qubo(self):
        return self._qubo

    def transform(self, x_problem: str):
        # seperate x_problem into an array by +, -
        coefficients, x_values = self.expression_dissect(x_problem)

        C: qubovert.PUBO = qubovert.PUBO()
        x = [qubovert.boolean_var("x%d" % i) for i in range(50)]

        for ctr in range(len(coefficients)):
            x_indices = x_values[ctr]
            if len(x_indices) == 3:
                C += coefficients[ctr] * (x[x_indices[0]] * x[x_indices[1]] * x[x_indices[2]])
            elif len(x_indices) == 2:
                C += coefficients[ctr] * (x[x_indices[0]] * x[x_indices[1]])
            elif len(x_indices) == 1:
                C += coefficients[ctr] * (x[x_indices[0]])
            elif len(x_indices) == 0:
                C += coefficients[ctr]

        return C

    def expression_dissect(self, expression: str):
        # Remove all spaces
        expression = expression.replace(" ", "").replace("x", " x")

        # Regex pattern to match terms in the expression
        regex_pattern = r'(?<=\d)\s*(?=[+-])'

        # Find all terms matching the pattern
        terms = re.split(regex_pattern, expression)

        # Initialize arrays for coefficients and x-value tuples
        coefficients = []
        x_values = []

        # Process each term
        for term in terms:
            # Split the term into parts
            parts = term.split()

            # Extract and process the coefficient
            if parts[0] == '+':
                # Coefficient is implied as 1
                coefficients.append(1.0)
                parts.pop(0)
            elif parts[0] == '-':
                # Coefficient is implied as -1
                coefficients.append(-1.0)
                parts.pop(0)
            else:
                # Explicit coefficient
                coeff = parts.pop(0).replace('+', '')
                coefficients.append(float(coeff))

            # Extract x-values
            x_tuple = tuple(int(x[1:]) for x in parts)
            x_values.append(x_tuple)

        # Output results
        return coefficients, x_values

    def x_pubo(self, C: qubovert.PUBO) -> (int, str):
        input_expression = C.pretty_str()
        output_expression = input_expression.replace('x(x', 'x(')
        return C.num_binary_variables, output_expression

    def x_pubo_qubo(self, C: qubovert.PUBO, pairs: set) -> (int, str):
        # create the variables
        #print('C__________')
        # print(type(C))
        #print(C)
        #print(C.degree)

        '''
        #Q:qubovert.utils._qubomatrix.QUBOMatrix = C.to_qubo()
        #TODO: worked for others
        #Q: qubovert.utils._qubomatrix.QUBOMatrix = C.to_qubo(pairs={('x0', 'x1'), ('x4', 'x5'), ('x0', 'x2'), ('x3', 'x5'), ('x1', 'x2'), ('x3', 'x4')})
        #TODO: 4 qubits > 6
        #Q: qubovert.utils._qubomatrix.QUBOMatrix = C.to_qubo(pairs={('x0', 'x1'), ('x2', 'x3')})

        pairs = {}

        match n:
            case 3:
                pairs = {('x0', 'x1')}
            case 4:
                pairs = {('x0', 'x1'), ('x2', 'x3')}
            case 5:
                pairs = {('x0', 'x1'), ('x0', 'x2'), ('x1', 'x2'), ('x3', 'x4')}
            case 6:
                pairs = {('x0', 'x1'), ('x4', 'x5'), ('x0', 'x2'), ('x3', 'x5'), ('x1', 'x2'), ('x3', 'x4')}
            case 7:
                pairs = {('x0', 'x1'), ('x0', 'x2'), ('x0', 'x3'), ('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3'), ('x4', 'x5'), ('x4', 'x6'), ('x5', 'x6')}
            case 8:
                pairs = {('x0', 'x1'), ('x0', 'x2'), ('x0', 'x3'), ('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3'), ('x4', 'x5'), ('x4', 'x6'), ('x4', 'x7'), ('x5', 'x6'), ('x5', 'x7'), ('x6', 'x7')}
        '''

        info('Suggested PAIRS', pairs)
        Q: qubovert.utils._qubomatrix.QUBOMatrix = C.to_qubo(pairs=pairs)

        #print(Q)
        #print(type(Q))
        #print(Q.pretty_str())

        return Q.num_binary_variables, Q.pretty_str()

    def replace_x(self, expression):
        def replace(match):
            n = int(match.group(1))
            return f"*0.5*(1-Z{n})"

        pattern = r'x\((\d+)\)'
        replaced_expression = re.sub(pattern, replace, expression)
        replaced_expression = replaced_expression.replace('+*', '+').replace('-*', '-')
        replaced_expression = replaced_expression.replace('+ *', '+').replace('- *', '-')
        return replaced_expression

    def make_pairs(self, n):
        left = math.ceil(n / 2) - 1
        right = math.floor(n / 2) - 1
        pairs_left = set(itertools.combinations(range(left + 1), 2))
        pairs_right = set(itertools.combinations(range(left + 1, n), 2))
        pairs_int = pairs_left.union(pairs_right)
        # Transform the set into a set of tuples with string values
        pairs = {('x' + str(item[0]), 'x' + str(item[1])) for item in pairs_int}
        return pairs, pairs_int

    def count_occurrences(self, expression: str, searchChar: str, n: int, pairs: set):
        # Split the expression using '+' and '-' as separators
        terms = re.split(r'[-+]', expression)

        # Initialize variables
        count = 0
        pattern = r'[()\[](\d+)[()\]]'
        q_q_p = ''
        q_q_p_count = 0
        q_q_np = ''
        q_q_np_count = 0
        q_a = ''
        q_a_count = 0
        a_a = ''
        a_a_count = 0
        found_pairs: set = set()

        # Iterate through the terms
        for term in terms:
            # Count the occurrences of searchChar in the current term
            term_count = term.count(searchChar)

            # If searchChar appears at least twice in the term, add to the total count
            if term_count >= 2:
                count += 1

                matches = re.findall(pattern, term)
                q0 = int(matches[0])  # Extract the first number
                q1 = int(matches[1])  # Extract the second number
                if q0 < q1:
                    found_pairs.add((q0, q1))
                else:
                    found_pairs.add((q1, q0))

                if q0 < n and q1 < n:
                    if (q0, q1) or (q1, q0) in pairs:
                        q_q_p_count += 1
                        q_q_p += f'({q0},{q1})'
                    else:
                        q_q_np_count += 1
                        q_q_np += f'({q0},{q1})'
                else:
                    if q0 >= n and q1 >= n:
                        a_a_count += 1
                        a_a += f'({q0},{q1})'
                    else:
                        q_a_count += 1
                        q_a += f'({q0},{q1})'

        ancillary_count = math.ceil(n*(n-2)/4)

        info('Max Ancillary Count', ancillary_count)
        self._update_or_insert_field('ancillary_count', ancillary_count)

        ancillary_list = {(i, j) for i in range(n) for j in range(n, n+ancillary_count)}

        q_q =    f'Q-Q<p> {q_q_p_count}: {q_q_p}' if q_q_p_count > 0 else ''
        q_q += f' | Q-Q<np> {q_q_np_count}: {q_q_np}' if q_q_np_count > 0 else ''
        q_q += f' | Q-A {q_a_count}: {q_a}' if q_a_count > 0 else ''
        q_q += f' | A-A {a_a_count}: {a_a}' if a_a_count > 0 else ''
        q_q += f' | Q-Q?? {len(pairs - found_pairs)}: {pairs - found_pairs}' if len(pairs - found_pairs) > 0 else ''
        q_q += f' | Q/A-A?? {len(ancillary_list - found_pairs)}: {ancillary_list - found_pairs}' if len(ancillary_list - found_pairs) > 0 else ''
        return count, q_q

    def replace_Zn_with_Zn_brackets(self, input_string):
        # Define a regular expression pattern to find 'Zn' where n is an integer
        pattern = r'Z(\d+)'

        # Use re.sub() to replace all matches with 'Z[n]'
        replaced_string = re.sub(pattern, r'Z[\1]', input_string)

        return replaced_string
