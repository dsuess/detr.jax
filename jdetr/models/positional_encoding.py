import jax.numpy as jnp

from jdetr._typing import JaxArray

__all__ = ["sinusoidal_encoding"]


# TODO: Replace by encodings which are padding-aware
def sinusoidal_encoding(
    grid_size: int, num_pos_features: int, temperature: float = 10000
) -> JaxArray:
    """
    >>> y = sinusoidal_encoding(7, 64)
    >>> tuple(y.shape)
    (7, 7, 128)
    """
    # pylint: disable=invalid-name
    xs = 1 + jnp.arange(grid_size, dtype=jnp.float32)
    ks = jnp.arange(num_pos_features, dtype=jnp.float32)
    ks = jnp.power(temperature, -2 * (ks // 2) / num_pos_features)
    ts = xs[:, None] * ks[None, :]

    pos_x = jnp.tile(ts[None, :, :], (grid_size, 1, 1))
    features_x = jnp.stack(
        (jnp.sin(pos_x[:, :, 0::2]), jnp.cos(pos_x[:, :, 1::2])), axis=3
    )
    pos_y = jnp.tile(ts[:, None, :], (1, grid_size, 1))
    features_y = jnp.stack(
        (jnp.sin(pos_y[:, :, 0::2]), jnp.cos(pos_y[:, :, 1::2])), axis=3
    )
    return jnp.concatenate([features_y, features_x], axis=3).reshape(
        (grid_size, grid_size, 2 * num_pos_features)
    )
