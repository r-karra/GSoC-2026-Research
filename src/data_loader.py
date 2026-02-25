import jax
import jax.numpy as jnp

def get_dummy_embeddings(num_sequences, dim=4, key=None):
    """Generate dummy embeddings for sequences, normalized for quantum gates."""
    if key is None:
        key = jax.random.PRNGKey(42)
    raw_data = jax.random.normal(key, (num_sequences, dim))
    # Normalize to [0, pi] for Angle Embedding
    normalized = jnp.pi * (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())
    return normalized

# Example usage for KJV verses or HEP data
# verses = ["In the beginning...", ...]
# sequence_data = get_dummy_embeddings(len(verses))