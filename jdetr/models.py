import haiku as hk
import jax
from jax import numpy as jnp
from jdetr._typing import JaxArray


class HeadlessResnet(hk.nets.ResNet):
    def __call__(
        self, inputs: JaxArray, is_training: bool, test_local_stats: bool = False
    ) -> JaxArray:
        out = inputs
        out = self.initial_conv(out)
        if not self.resnet_v2:
            out = self.initial_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)

        out = hk.max_pool(
            out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
        )

        for block_group in self.block_groups:
            out = block_group(out, is_training, test_local_stats)

        return out


class ResNet50(HeadlessResnet, hk.nets.ResNet50):
    """
    >>> from jdetr.utils import Init
    >>> x = jnp.zeros((1, 224, 224, 3))
    >>> y = Init(ResNet50, 0).__call__(x, is_training=True)
    >>> y.shape
    (1, 7, 7, 2048)
    """

    ...


# TODO: Replace by encodings which are padding-aware
def get_positional_encoding2d(
    grid_size: int, num_pos_features: int, temperature: float = 10000
) -> JaxArray:
    """
    >>> y = get_positional_encoding2d(7, 64)
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


class DETR(hk.Module):
    """
    >>> from jdetr.utils import Init
    >>> x = jnp.zeros((1, 224, 224, 3))
    >>> y = Init(DETR, 2).__call__(x, is_training=True)

    """

    def __init__(self, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = hk.Sequential(
            [ResNet50(0, resnet_v2=True), hk.Conv2D(hidden_dim, kernel_shape=1)]
        )
        self.num_pos_features = 64

    def __call__(self, image: JaxArray, is_training: bool):
        # FIXME Make work in channel_first
        assert image.shape[1] == image.shape[2]

        cnn_features = self.backbone(image, is_training=is_training)
        _, grid_size, _, _ = cnn_features.shape
        cnn_features = jnp.reshape(
            cnn_features, (cnn_features.shape[0], -1, cnn_features.shape[-1])
        )
        posititional_encoding = get_positional_encoding2d(
            grid_size, self.num_pos_features
        )
        posititional_encoding = jnp.tile(
            posititional_encoding.reshape(1, -1, posititional_encoding.shape[-1]),
            (cnn_features.shape[0], 1, 1),
        )
        features = jnp.concatenate([cnn_features, posititional_encoding], axis=-1)
        __import__("pdb").set_trace()  # FIXME PDB
