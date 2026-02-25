import jax
import jax.numpy as jnp
import pennylane as qml

def create_quantum_attention_layer(n_qubits=4):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def quantum_attention_layer(inputs, weights):
        # Data Encoding
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # Trainable Variational Layers (Entanglement)
        # This acts as the 'Attention Weight' learner
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Measure expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Vectorize the circuit to handle the whole sequence at once
    vnode = jax.vmap(quantum_attention_layer, in_axes=(0, None))
    return vnode

def initialize_quantum_weights(key, n_layers=2, n_qubits=4):
    return jax.random.normal(key, (n_layers, n_qubits, 3))