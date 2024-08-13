import re

class Circuit:
    _pauli_matrices: str
    _coefficient: float

    def __init__(self, pauli_matrices: str, coefficient: float):
        self._pauli_matrices = pauli_matrices
        self._coefficient = coefficient

    def __str__(self):
        return f"Circuit: Pauli Matrices = {self._pauli_matrices}, Coefficient = {self._coefficient}"

    def get_pauli_matrices(self):
        return self._pauli_matrices

    def get_coefficient(self):
        return self._coefficient

    @staticmethod
    def observable_to_circuit(observable: str):
        circuit = []
        # Split the formula based on operators (+, -)
        tokens = [token.strip() for token in re.split(r'([-+])', observable) if token.strip() != '']
        _sign = 1
        for token in tokens:
            if token in ['+', '-']:
                _sign = -1 if token == '-' else 1
            else:
                parts = token.strip().split()
                coefficient = float(parts[0])
                pauli_matrices = ' '.join(parts[1:])
                circuit.append(Circuit(pauli_matrices, coefficient * _sign))
        return circuit
