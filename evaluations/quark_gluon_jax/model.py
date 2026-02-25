import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
import matplotlib.pyplot as plt

class QuarkGluonClassifier(nn.Module):
    """A standard MLP for binary classification of particle jets."""
    hidden_dims: int = 128

    @nn.compact
    def __call__(self, x):
        # Input features (e.g., [pt, eta, phi, mass, etc.])
        x = nn.Dense(self.hidden_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dims // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)  # Single output for binary classification
        return x

def binary_cross_entropy_loss(params, apply_fn, x, y):
    logits = apply_fn({'params': params}, x)
    # Use sigmoid binary cross-entropy for a single output binary classification
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

@jax.jit
def train_step(state, x, y):
    """A single vectorized training step using JAX JIT."""
    # Inner loss function to pass only parameters for gradient computation
    def loss_fn(params):
        return binary_cross_entropy_loss(params, state.apply_fn, x, y)

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Apply gradients to update model parameters
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def initialize_model(key, hidden_dims=128, learning_rate=1e-3):
    # Create an instance of the QuarkGluonClassifier model
    model = QuarkGluonClassifier(hidden_dims=hidden_dims)

    # Initialize the model's parameters using a dummy input
    dummy_x = jnp.ones([1, 5])  # A single dummy input with 5 features
    params = model.init(key, dummy_x)['params']

    # Initialize an Adam optimizer
    tx = optax.adam(learning_rate)

    # Create an optax.TrainState
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    return state

def train_and_plot(key, num_steps=100, log_interval=10):
    """Train the model and plot the loss curve."""
    state = initialize_model(key)

    # Generate dummy data
    key, subkey_x, subkey_y = jax.random.split(key, 3)
    batch_size = 32
    num_features = 5
    x = jax.random.normal(subkey_x, (batch_size, num_features))
    y = jax.random.bernoulli(subkey_y, shape=(batch_size, 1)).astype(jnp.float32)

    losses = []
    for i in range(num_steps):
        state, loss = train_step(state, x, y)
        losses.append(loss)

    # Plot the loss curve
    plt.plot(losses)
    plt.title("Quark-Gluon Classifier Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig('results.png')
    plt.show()

    return losses