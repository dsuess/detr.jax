import haiku as hk
import jax
import jax.numpy as jnp

from jdetr._typing import JaxArray
from jdetr.utils import maybe

__all__ = ["Transformer"]


class MultiHeadAttentionLayer(hk.Module):
    def __init__(
        self,
        feature_dim: int,
        value_dim: int,
        num_heads: int,
        key_query_dim: int = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.key_query_dim = maybe(key_query_dim, value_dim)

    @hk.transparent
    def multi_head_linear(self, x: JaxArray, dim: int) -> JaxArray:
        """
        >>> from jdetr.utils import Init
        >>> x = jnp.zeros((2, 3, 4))
        >>> y = (
        ...     Init(MultiHeadAttentionLayer, feature_dim=5, value_dim=6, num_heads=7)
        ...     .multi_head_linear(x, dim=6)
        ... )
        >>> tuple(y.shape)
        (2, 3, 7, 6)
        """
        y = hk.Linear(dim * self.num_heads)(x)
        # (batch_idx, seq_idx, head_idx, hidden_dim)
        return y.reshape((*x.shape[:-1], self.num_heads, dim))

    # pylint: disable=invalid-name
    @hk.transparent
    def _multihead_attention(self, k: JaxArray, q: JaxArray, v: JaxArray) -> JaxArray:
        attn = jnp.einsum("btij,bsij->btsi", q, k) / jnp.sqrt(self.key_query_dim)
        attn = jax.nn.softmax(attn, axis=2)
        z = jnp.einsum("btsi,bsij->btij", attn, v).reshape(
            q.shape[0], q.shape[1], self.num_heads * self.value_dim
        )
        return hk.Linear(self.feature_dim)(z)

    def __call__(self, key: JaxArray, query: JaxArray, value: JaxArray) -> JaxArray:
        """
        >>> from jdetr.utils import Init
        >>> x = jnp.zeros((2, 3, 4))
        >>> x_ = jnp.zeros((2, 4, 4))
        >>> y = (
        ...     Init(MultiHeadAttentionLayer, feature_dim=5, value_dim=6, num_heads=7)
        ...     .__call__(key=x, query=x_, value=x)
        ... )
        >>> tuple(y.shape)
        (2, 4, 5)
        """
        k = self.multi_head_linear(key, self.key_query_dim)
        q = self.multi_head_linear(query, self.key_query_dim)
        v = self.multi_head_linear(value, self.value_dim)
        return self._multihead_attention(k, q, v)


class DetrMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    """
    >>> from jdetr.utils import Init
    >>> from jdetr.models.positional_encoding import sinusoidal_encoding
    >>> x = jnp.zeros((2, 16, 4))
    >>> pos_encoding = sinusoidal_encoding(4, 2).reshape(1, 16, 4)
    >>> y = (
    ...     Init(DetrMultiHeadAttentionLayer, 5, 6, 7, 32)
    ...     .__call__(x, pos_encoding)
    ... )
    >>> tuple(y.shape)
    (2, 16, 5)
    """

    def __call__(self, x: JaxArray, pos_encoding: JaxArray) -> JaxArray:
        # pylint: disable=invalid-name
        # Add dimension for head-index
        k = self.multi_head_linear(x + pos_encoding, self.key_query_dim)
        q = self.multi_head_linear(x + pos_encoding, self.key_query_dim)
        v = self.multi_head_linear(x, self.value_dim)
        return self._multihead_attention(k, q, v)


class DropoutLayer(hk.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate

    def __call__(self, x: JaxArray, is_training: bool) -> JaxArray:
        rng = hk.next_rng_key()
        return hk.dropout(rng, self.dropout_rate, x)


class EncoderLayer(hk.Module):
    """
    >>> from jdetr.utils import Init
    >>> from jdetr.models.positional_encoding import sinusoidal_encoding
    >>> x = jnp.zeros((2, 16, 4))
    >>> pos_encoding = sinusoidal_encoding(4, 2).reshape(1, 16, 4)
    >>> y = (
    ...     Init(EncoderLayer, feature_dim=4, num_heads=2)
    ...     .__call__(x, pos_encoding, True)
    ... )
    >>> tuple(y.shape)
    (2, 16, 4)
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        feedforward_dim: int = 2048,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.feedforward_dim = feedforward_dim

    def __call__(
        self, x: JaxArray, pos_encoding: JaxArray, is_training: bool
    ) -> JaxArray:
        y = DetrMultiHeadAttentionLayer(
            self.feature_dim, self.feature_dim, self.num_heads
        )(x, pos_encoding)
        y = x + DropoutLayer(self.dropout_rate)(y, is_training)
        # TODO Try out with batchnorm as well
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(y)

        y = hk.Linear(self.feedforward_dim)(x)
        y = jax.nn.relu6(y)
        y = DropoutLayer(self.dropout_rate)(y, is_training)
        y = hk.Linear(self.feature_dim)(y)
        y = x + DropoutLayer(self.dropout_rate)(y, is_training)
        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(y)


class DecoderLayer(hk.Module):
    """
    >>> from jdetr.utils import Init
    >>> from jdetr.models.positional_encoding import sinusoidal_encoding
    >>> x = jnp.zeros((2, 16, 4))
    >>> pos_encoding = sinusoidal_encoding(4, 2).reshape(1, 16, 4)
    >>> y = (
    ...     Init(DecoderLayer, feature_dim=4, num_heads=2)
    ...     .__call__(x, x, pos_encoding, pos_encoding, True)
    ... )
    >>> tuple(y.shape)
    (2, 16, 4)
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        feedforward_dim: int = 2048,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.feedforward_dim = feedforward_dim

    def __call__(
        self,
        encoder_features: JaxArray,
        decoder_features: JaxArray,
        pos_encoding: JaxArray,
        query_encoding: JaxArray,
        is_training: bool,
    ) -> JaxArray:
        y = DetrMultiHeadAttentionLayer(
            self.feature_dim, self.feature_dim, self.num_heads
        )(decoder_features, query_encoding)
        y = decoder_features + DropoutLayer(self.dropout_rate)(y, is_training)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(y)

        y = MultiHeadAttentionLayer(self.feature_dim, self.feature_dim, self.num_heads)(
            key=encoder_features + pos_encoding,
            query=x + query_encoding,
            value=encoder_features,
        )
        y = x + DropoutLayer(self.dropout_rate)(y, is_training)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(y)

        y = hk.Linear(self.feedforward_dim)(x)
        y = jax.nn.relu6(y)
        y = DropoutLayer(self.dropout_rate)(y, is_training)
        y = hk.Linear(self.feature_dim)(y)
        y = x + DropoutLayer(self.dropout_rate)(y, is_training)
        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(y)


class Transformer(hk.Module):
    """
    >>> from jdetr.utils import Init
    >>> from jdetr.models.positional_encoding import sinusoidal_encoding
    >>> x = jnp.zeros((2, 16, 4))
    >>> pos_encoding = sinusoidal_encoding(4, 2).reshape(1, 16, 4)
    >>> query_encoding = jnp.zeros((2, 10, 4))
    >>> y = Init(Transformer, 4, 2, 1, 1).__call__(x, pos_encoding, query_encoding, True)
    >>> tuple(y.shape)
    (2, 10, 4)
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout_rate: float = 0.1,
        feedforward_dim: int = 2048,
    ):
        assert feature_dim % 2 == 0

        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_rate = dropout_rate
        self.feedforward_dim = feedforward_dim

    def __call__(
        self,
        x: JaxArray,
        pos_encoding: JaxArray,
        query_encoding: JaxArray,
        is_training: bool,
    ) -> JaxArray:
        encoder_features = x
        for _ in range(self.num_encoder_layers):
            encoder_features = EncoderLayer(
                self.feature_dim,
                self.num_heads,
                self.dropout_rate,
                feedforward_dim=self.feedforward_dim,
            )(encoder_features, pos_encoding, is_training)

        decoder_features = jnp.zeros_like(query_encoding)
        for _ in range(self.num_decoder_layers):
            decoder_features = DecoderLayer(
                self.feature_dim,
                self.num_heads,
                self.dropout_rate,
                feedforward_dim=self.feedforward_dim,
            )(
                encoder_features,
                decoder_features,
                pos_encoding,
                query_encoding,
                is_training,
            )

        return decoder_features
