from qulacs import QuantumCircuit
from qulacsvis.qulacs.circuit import to_model
from qulacsvis import circuit_drawer
from qulacsvis.visualization import MPLCircuitlDrawer
import matplotlib.pyplot as plt

# Build a quantum circuit
circuit = QuantumCircuit(3)
circuit.add_X_gate(0)
circuit.add_Y_gate(1)
circuit.add_Z_gate(2)
circuit.add_dense_matrix_gate(
    [0, 1], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
)
circuit.add_CNOT_gate(2, 0)
circuit.add_X_gate(2)

# Draw a quantum circuit
#circuit_drawer(circuit)

drawer = MPLCircuitlDrawer(to_model(circuit))
drawer.draw()
plt.show()

#circuit_drawer(circuit, "latex")
