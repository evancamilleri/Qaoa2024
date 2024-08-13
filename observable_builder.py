from collections import OrderedDict
import re
from enum import Enum
from functools import reduce
from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
from circuit import Circuit


class TokenType(Enum):
    Variable = 1
    Constant = 2
    Operator = 3


class ObservableBuilder:

    def __init__(self, variable_limits: OrderedDict, cost_function: str, change_sign: bool, weight: float):
        self._variable_limits = variable_limits
        self._cost_function = cost_function.replace(" ", "")
        self.change_cost_function_sign() if change_sign else None
        self._weight = weight
        self._tokens = []
        self._final_observable = ''
        self._circuit_list = []
        self.parse_cost_function()

    def _debug_tokens(self):
        print(self._tokens)

    def _debug_final_observable(self):
        print(self._final_observable)

    def get_final_observable(self):
        return self._final_observable

    def clear_observable(self):
        self._final_observable = ''

    def circuit_list(self):
        return self._circuit_list

    def change_cost_function_sign(self):
        trans_table = str.maketrans("-+", "+-")
        self._cost_function = f'+ {self._cost_function}' if not self._cost_function.startswith(('+', '-')) else self._cost_function
        self._cost_function = self._cost_function.translate(trans_table)

    def parse_cost_function(self):
        # Split the formula based on operators (+, -)
        tokens = re.split(r'([-+])', self._cost_function)
        # Remove empty tokens
        self._tokens = [token for token in tokens if token]

    def add_to_observable(self, _variables: dict, **kwargs):
        # Process the tokens and generate the desired output
        output = ''
        for token in self._tokens:

            # replace the tags in token using kwargs
            for tag, value in kwargs.items():
                token = token.replace("{" + tag + "}", str(value))

            # now process the token
            if '*' in token:
                product_tokens = token.split('*')
                # make sure we always have a constant prefix per product_token
                product_tokens.insert(0, '1') if ObservableBuilder.get_token_type(product_tokens[0]) != TokenType.Constant else None
                for ptoken in product_tokens:
                    output += self.process_single_token(ptoken, self._weight, _variables) + ' '
                output = output.strip()
            else:
                if ObservableBuilder.get_token_type(token) != TokenType.Variable:
                    output += self.process_single_token(token, self._weight, _variables)
                else:
                    output += "{:.3f}".format(self._weight) + ' ' + self.process_single_token(token, self._weight, _variables)

        self._final_observable += output.strip() if self._final_observable == '' else ' + ' + output.strip()

    def process_single_token(self, token: str, weight: float, _dict: dict):
        output = ''
        token_type = ObservableBuilder.get_token_type(token)
        # print(token_type, token)
        match token_type:
            case TokenType.Operator:
                output += f" {token} "

            case TokenType.Variable:
                output += self.pauliz_token_to_qubit_index(token, _dict)

            case TokenType.Constant:
                output += "{:.3f}".format(float(token) * weight)

        return output

    def pauliz_token_to_qubit_index(self, token: str, _dict: dict):
        # from Z[a,b',c] get a, b', c
        values = token[token.index('[') + 1:token.index(']')]
        indices = values.split(',')

        # get dict of indexes and their value
        _index_dict: dict = {}
        for i, _index in enumerate(indices):
            if _index.isdigit():
                key_index = list(self._variable_limits.keys())[i].lower()
                _index_dict[key_index] = int(_index)
            else:
                _index_dict[_index.replace("'", "")] = _dict[_index]

        return 'Z ' + str(self.qubit_index(_index_dict))

    # determine the qubit number from all the indexes
    def qubit_index(self, _index_dict: dict):
        index = 0
        multiplier = 1

        for variable, limit in reversed(self._variable_limits.items()):
            value = _index_dict[variable.lower()]
            index += value * multiplier
            multiplier *= limit

        return index

    # Helper function to determine the type of token
    @staticmethod
    def get_token_type(token):
        token = token.strip()
        if token.startswith("Z["):
            return TokenType.Variable
        elif ObservableBuilder.is_number(token):
            return TokenType.Constant
        elif token in ['+', '-']:
            return TokenType.Operator

    @staticmethod
    def is_number(token):
        if token.startswith('-'):
            token = token[1:]  # Remove the negative sign for further checks
        # Check if the remaining token consists of only digits or a decimal point
        if token.isdigit() or (token.count('.') == 1 and token.replace('.', '').isdigit()):
            return True
        else:
            return False

    def get_qubits(self) -> int:
        return reduce(lambda x, y: x * y, map(int, self._variable_limits.values()))

    def get_observable(self) -> (Observable, list):
        self._circuit_list = Circuit.observable_to_circuit(self._final_observable)
        # [print(instance) for instance in circuit_list]
        n = self.get_qubits()
        cost_observable = Observable(n)
        for _c in self._circuit_list:
            cost_observable.add_operator(PauliOperator(_c.get_pauli_matrices(), _c.get_coefficient()))
        return cost_observable, self._circuit_list

    def get_observable2(self, final_observable) -> (Observable, list):
        self._circuit_list = Circuit.observable_to_circuit(final_observable)
        # [print(instance) for instance in circuit_list]
        n = self.get_qubits()
        cost_observable = Observable(n)
        for _c in self._circuit_list:
            cost_observable.add_operator(PauliOperator(_c.get_pauli_matrices(), _c.get_coefficient()))
        return cost_observable, self._circuit_list
