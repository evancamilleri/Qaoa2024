from enum import Enum

TITLE_LENGHT = 20

class AngleStudy(Enum):
    default = 0
    multi_angle = 1
    just_one = 2
    polynomial = 3
    xautomorphism = 4
    qubits = 5
    automorphism_global = 6
    automorphism_local = 7
    ka = 8


class ClassicalOptimiser(Enum):
    BFGS = 'BFGS'
    Powell = 'Powell'
    COBYLA = 'COBYLA'
    Nelder_Mead = 'Nelder-Mead'
    # SPSA = 'SPSA'


class GraphType(Enum):
    na = 0
    path = 1
    cyclic = 2

def info(title: str, value):
    print(title + ' ' * (TITLE_LENGHT - len(title)) + ': ' + str(value))


def insert_with_padding(lst, index, part, padding=None):
    # Extend the list with the padding if the index is greater than the list length
    while len(lst) < index:
        lst.append(padding)
    # Now insert the part at the specified index
    lst.insert(index, part)
